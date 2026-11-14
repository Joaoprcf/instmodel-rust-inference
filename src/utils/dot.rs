//! SIMD dot-product kernels and the dense (matrix-vector + bias) layer shared by DOT and ATTENTION.

use crate::errors::InstructionModelError;
use crate::instructions::split_src_dst;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DotKernel {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512Fma,
}

impl DotKernel {
    #[inline(always)]
    pub(crate) fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
                return DotKernel::Avx512Fma;
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return DotKernel::Avx2Fma;
            }
        }
        DotKernel::Scalar
    }
}

/// Dot product over the first `min(a.len(), b.len())` elements.
#[inline(always)]
pub(crate) fn dot(kernel: DotKernel, a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (a.as_ptr(), b.as_ptr());
    // SAFETY: both slices hold at least `len` elements, and the SIMD variants are only
    // selected by `DotKernel::detect` when the CPU supports their target features.
    unsafe {
        match kernel {
            DotKernel::Scalar => dot_scalar(a, b, len),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            DotKernel::Avx2Fma => simd::dot_avx2_fma(a, b, len),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            DotKernel::Avx512Fma => simd::dot_avx512_fma(a, b, len),
        }
    }
}

#[inline(always)]
unsafe fn tail(a: *const f32, b: *const f32, mut i: usize, len: usize, mut sum: f32) -> f32 {
    while i < len {
        sum = unsafe { (*a.add(i)).mul_add(*b.add(i), sum) };
        i += 1;
    }
    sum
}

#[inline(always)]
unsafe fn dot_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut s = [0.0f32; 4];
    let mut i = 0usize;
    while i + 4 <= len {
        for (lane, acc) in s.iter_mut().enumerate() {
            *acc = unsafe { (*a.add(i + lane)).mul_add(*b.add(i + lane), *acc) };
        }
        i += 4;
    }
    unsafe { tail(a, b, i, len, (s[0] + s[1]) + (s[2] + s[3])) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod simd {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    fn hsum(lanes: &[f32]) -> f32 {
        lanes[1..].iter().fold(lanes[0], |acc, &v| acc + v)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= len {
            let (va, vb) = unsafe { (_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i))) };
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        unsafe { super::tail(a, b, i, len, hsum(&lanes)) }
    }

    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn dot_avx512_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= len {
            let (va, vb) = unsafe { (_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i))) };
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }
        let mut lanes = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        unsafe { super::tail(a, b, i, len, hsum(&lanes)) }
    }
}

/// Row-major weight matrix plus bias: `out[r] = dot(weights[r], input) + bias[r]`.
#[derive(Clone)]
pub(crate) struct Dense {
    weights: Vec<f32>,
    bias: Vec<f32>,
    cols: usize,
    kernel: DotKernel,
}

impl Dense {
    pub(crate) fn new(rows: &[Vec<f32>], bias: &[f32]) -> Self {
        Self {
            weights: rows.concat(),
            bias: bias.to_vec(),
            cols: rows.first().map_or(0, Vec::len),
            kernel: DotKernel::detect(),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.bias.len()
    }

    /// Writes the layer output for `buf[src..src + cols]` into `buf[dst..dst + rows]`
    /// and returns the output slice. Overlapping ranges read a copy of the input.
    #[inline(always)]
    pub(crate) fn forward<'b>(&self, buf: &'b mut [f32], src: usize, dst: usize) -> &'b mut [f32] {
        let (cols, rows) = (self.cols, self.rows());
        if let Some((input, out)) = split_src_dst(buf, src, cols, dst, rows) {
            self.fill(input, out);
        } else {
            let input = buf[src..src + cols].to_vec();
            self.fill(&input, &mut buf[dst..dst + rows]);
        }
        &mut buf[dst..dst + rows]
    }

    #[inline(always)]
    fn fill(&self, input: &[f32], out: &mut [f32]) {
        let cols = self.cols;
        for (r, (o, &b)) in out.iter_mut().zip(&self.bias).enumerate() {
            *o = dot(self.kernel, &self.weights[r * cols..(r + 1) * cols], input) + b;
        }
    }

    fn check_len(&self, weights: usize, bias: usize) -> Result<(), InstructionModelError> {
        if weights != self.weights.len() || bias != self.bias.len() {
            return Err(InstructionModelError::ThetaLengthMismatch {
                expected: self.weights.len() + self.bias.len(),
                got: weights + bias,
            });
        }
        Ok(())
    }

    pub(crate) fn set(
        &mut self,
        weights: &[f32],
        bias: &[f32],
    ) -> Result<(), InstructionModelError> {
        self.check_len(weights.len(), bias.len())?;
        self.weights.copy_from_slice(weights);
        self.bias.copy_from_slice(bias);
        Ok(())
    }

    pub(crate) fn read(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
    ) -> Result<(), InstructionModelError> {
        self.check_len(weights.len(), bias.len())?;
        weights.copy_from_slice(&self.weights);
        bias.copy_from_slice(&self.bias);
        Ok(())
    }
}
