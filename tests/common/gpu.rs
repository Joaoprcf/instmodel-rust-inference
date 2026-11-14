use instmodel_inference::gpu::{GpuModel, get_instmodel_wgsl};
use pollster::FutureExt;
use wgpu::util::DeviceExt;

const TEST_SHADER_TEMPLATE: &str = include_str!("../shaders/test_shader.wgsl");

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Runs one prediction through the embeddable WGSL interpreter; `None` when no adapter is available.
pub fn gpu_predict(gpu_model: &GpuModel, input: &[f32]) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .block_on()
        .ok()?;

    let size = gpu_model.compute_buffer_size();
    let source = TEST_SHADER_TEMPLATE
        .replace(
            "// INSTMODEL_WGSL_PLACEHOLDER - replaced at runtime with actual instmodel code",
            &get_instmodel_wgsl(size as u32),
        )
        .replace(
            "// COMPUTE_BUFFER_DECLARATION - replaced at runtime with actual size",
            &format!("var compute_buffer: array<f32, {size}>;"),
        );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let init = |contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let model_buffer = init(gpu_model.as_bytes());
    let input_buffer = init(bytemuck::cast_slice(input));
    let output_bytes = (gpu_model.output_size() * std::mem::size_of::<f32>()) as u64;
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output_buffer = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let staging_buffer = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[&model_buffer, &input_buffer, &output_buffer]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging_buffer.unmap();
    Some(result)
}
