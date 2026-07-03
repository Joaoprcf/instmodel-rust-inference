//! wgpu-backed evaluation host (`gpu-runtime` feature).
//!
//! [`GpuContext`] owns the adapter/device/queue; [`PopulationEvaluator`]
//! wraps a [`PopulationPack`] with persistent GPU buffers and a prebuilt
//! pipeline so an ES loop is: write candidates → [`evaluate`] → fitness →
//! repeat, with one dispatch and one synchronous readback per step.

use pollster::FutureExt;

use crate::gpu::errors::{GpuRuntimeError, PopulationError};
use crate::gpu::population::PopulationPack;
use crate::gpu::shaders::get_instmodel_wgsl;
use crate::graph::ModelGraph;
use crate::instruction_model_info::InstructionModelInfo;

const POPULATION_ENTRY_TEMPLATE: &str = include_str!("shaders/population_entry.wgsl");
const WORKGROUP_SIZE: u32 = 64;

/// Options for [`GpuContext::new`].
#[derive(Debug, Clone, Default)]
pub struct GpuContextOptions {
    /// Accept software/CPU adapters (llvmpipe, SwiftShader, ...). Off by
    /// default so training loops fail loudly instead of silently running on
    /// a CPU rasterizer.
    pub allow_software_adapter: bool,
    /// Ask wgpu for its fallback (software) adapter explicitly — mainly for
    /// tests.
    pub force_fallback_adapter: bool,
}

/// An initialized wgpu device/queue pair.
pub struct GpuContext {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// Requests a high-performance adapter and device.
    pub fn new(options: &GpuContextOptions) -> Result<Self, GpuRuntimeError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: None,
            })
            .block_on()
            .ok_or(GpuRuntimeError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        if adapter_info.device_type == wgpu::DeviceType::Cpu && !options.allow_software_adapter {
            return Err(GpuRuntimeError::SoftwareAdapterRejected {
                name: adapter_info.name,
            });
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .block_on()
            .map_err(|source| GpuRuntimeError::DeviceRequestFailed {
                message: source.to_string(),
            })?;

        Ok(GpuContext {
            adapter_info,
            device,
            queue,
        })
    }

    /// Information about the selected adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// The wgpu device (cloneable handle).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue (cloneable handle).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Evaluates every candidate of a [`PopulationPack`] against a shared input
/// batch in a single compute dispatch.
///
/// All GPU buffers are allocated once at construction and reused by every
/// [`evaluate`](Self::evaluate) call.
pub struct PopulationEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pack: PopulationPack,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    model_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    batch_size: usize,
    workgroups: u32,
}

impl PopulationEvaluator {
    /// Builds an evaluator for `n_candidates` copies of the model described
    /// by `info`, each evaluated over `batch_size` samples per dispatch.
    pub fn new(
        context: &GpuContext,
        info: &InstructionModelInfo,
        n_candidates: usize,
        batch_size: usize,
    ) -> Result<Self, GpuRuntimeError> {
        if batch_size == 0 {
            return Err(GpuRuntimeError::ZeroBatchSize);
        }
        let pack = PopulationPack::new(info, n_candidates)?;
        Self::from_pack(context, pack, batch_size)
    }

    /// Compiles `graph` with zeroed parameters and builds an evaluator from
    /// it.
    pub fn from_graph(
        context: &GpuContext,
        graph: &ModelGraph,
        n_candidates: usize,
        batch_size: usize,
    ) -> Result<Self, GpuRuntimeError> {
        let info = graph.compile_zeroed().map_err(PopulationError::from)?;
        Self::new(context, &info, n_candidates, batch_size)
    }

    fn from_pack(
        context: &GpuContext,
        pack: PopulationPack,
        batch_size: usize,
    ) -> Result<Self, GpuRuntimeError> {
        let device = context.device().clone();
        let queue = context.queue().clone();
        let limits = device.limits();

        let total_threads = (pack.n_candidates() * batch_size) as u32;
        let workgroups = total_threads.div_ceil(WORKGROUP_SIZE);
        if workgroups > limits.max_compute_workgroups_per_dimension {
            return Err(GpuRuntimeError::DispatchTooLarge {
                workgroups,
                max: limits.max_compute_workgroups_per_dimension,
            });
        }

        let model_bytes = pack.as_bytes().len() as u64;
        let output_bytes =
            (pack.n_candidates() * batch_size * pack.output_size() * std::mem::size_of::<f32>())
                as u64;
        let max_binding = u64::from(limits.max_storage_buffer_binding_size);
        for required in [model_bytes, output_bytes] {
            if required > max_binding {
                return Err(GpuRuntimeError::BufferLimitExceeded {
                    required,
                    max: max_binding,
                });
            }
        }

        let pipeline = Self::build_pipeline(&device, &pack, batch_size)?;

        let input_bytes = (batch_size * pack.feature_size() * std::mem::size_of::<f32>()) as u64;
        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population Models"),
            size: model_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population Inputs"),
            size: input_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population Outputs"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population Staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Population Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let evaluator = PopulationEvaluator {
            device,
            queue,
            pack,
            pipeline,
            bind_group,
            model_buffer,
            input_buffer,
            output_buffer,
            staging_buffer,
            batch_size,
            workgroups,
        };
        evaluator.warmup();
        Ok(evaluator)
    }

    fn build_pipeline(
        device: &wgpu::Device,
        pack: &PopulationPack,
        batch_size: usize,
    ) -> Result<wgpu::ComputePipeline, GpuRuntimeError> {
        let dims = format!(
            "const N_CANDIDATES: u32 = {}u;\nconst BATCH_SIZE: u32 = {}u;\nconst MODEL_STRIDE: u32 = {}u;",
            pack.n_candidates(),
            batch_size,
            pack.model_stride()
        );
        let source = POPULATION_ENTRY_TEMPLATE
            .replace(
                "// POPULATION_DIMS - replaced at build time with N_CANDIDATES/BATCH_SIZE/MODEL_STRIDE constants",
                &dims,
            )
            .replace(
                "// INSTMODEL_FUNCTIONS - replaced at build time with the generated inference library",
                &get_instmodel_wgsl(pack.compute_buffer_size() as u32),
            );

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Population Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Population Pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = device.pop_error_scope().block_on() {
            return Err(GpuRuntimeError::ShaderCompilationFailed {
                message: error.to_string(),
            });
        }
        Ok(pipeline)
    }

    /// One throwaway dispatch so pipeline/driver warmup cost is paid at
    /// construction instead of inside the first training step. The buffers
    /// are still zeroed, which the interpreter reads as an empty model.
    fn warmup(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Overwrites one candidate's weights (canonical flat-θ order).
    pub fn write_candidate(
        &mut self,
        candidate: usize,
        theta: &[f32],
    ) -> Result<(), GpuRuntimeError> {
        Ok(self.pack.write_candidate(candidate, theta)?)
    }

    /// Overwrites one candidate's weights from f64 optimizer state.
    pub fn write_candidate_f64(
        &mut self,
        candidate: usize,
        theta: &[f64],
    ) -> Result<(), GpuRuntimeError> {
        Ok(self.pack.write_candidate_f64(candidate, theta)?)
    }

    /// Writes a whole population at once. `population` is candidate-major
    /// `[n_candidates * theta_len]` — exactly the buffer returned by
    /// [`EsOptimizer::ask`](crate::evolution::EsOptimizer::ask) when the
    /// evaluator was sized with `n_candidates == population_size()`.
    pub fn write_population_f64(&mut self, population: &[f64]) -> Result<(), GpuRuntimeError> {
        let theta_len = self.pack.theta_len();
        let expected = self.pack.n_candidates() * theta_len;
        if population.len() != expected {
            return Err(GpuRuntimeError::PopulationLengthMismatch {
                expected,
                got: population.len(),
            });
        }
        if theta_len == 0 {
            return Ok(());
        }
        for (candidate, theta) in population.chunks_exact(theta_len).enumerate() {
            self.pack.write_candidate_f64(candidate, theta)?;
        }
        Ok(())
    }

    /// Runs every candidate over `inputs` (row-major
    /// `[batch_size * feature_size]`, shared by all candidates) and fills
    /// `outputs` candidate-major
    /// (`[n_candidates * batch_size * output_size]`): one buffer upload, one
    /// dispatch, one synchronous readback.
    pub fn evaluate(&mut self, inputs: &[f32], outputs: &mut [f32]) -> Result<(), GpuRuntimeError> {
        let expected_inputs = self.batch_size * self.pack.feature_size();
        if inputs.len() != expected_inputs {
            return Err(GpuRuntimeError::InputLengthMismatch {
                expected: expected_inputs,
                got: inputs.len(),
            });
        }
        let expected_outputs = self.pack.n_candidates() * self.batch_size * self.pack.output_size();
        if outputs.len() != expected_outputs {
            return Err(GpuRuntimeError::OutputLengthMismatch {
                expected: expected_outputs,
                got: outputs.len(),
            });
        }

        self.queue
            .write_buffer(&self.model_buffer, 0, self.pack.as_bytes());
        self.queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(inputs));

        let output_bytes = (expected_outputs * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            output_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..output_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        {
            let data = slice.get_mapped_range();
            outputs.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_buffer.unmap();
        Ok(())
    }

    /// Number of candidates per dispatch.
    pub fn n_candidates(&self) -> usize {
        self.pack.n_candidates()
    }

    /// Samples per dispatch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Flat parameter count per candidate.
    pub fn theta_len(&self) -> usize {
        self.pack.theta_len()
    }

    /// Model input size.
    pub fn feature_size(&self) -> usize {
        self.pack.feature_size()
    }

    /// Model output size.
    pub fn output_size(&self) -> usize {
        self.pack.output_size()
    }

    /// The underlying host-side pack.
    pub fn pack(&self) -> &PopulationPack {
        &self.pack
    }

    /// Mutable access to the pack for direct candidate manipulation; the
    /// next [`evaluate`](Self::evaluate) uploads whatever the pack holds.
    pub fn pack_mut(&mut self) -> &mut PopulationPack {
        &mut self.pack
    }
}
