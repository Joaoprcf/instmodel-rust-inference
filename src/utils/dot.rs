//! SIMD-aware dot-product kernels.

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

#[inline(always)]
pub(crate) fn dot(kernel: DotKernel, a: *const f32, b: *const f32, len: usize) -> f32 {
    match kernel {
        DotKernel::Scalar => unsafe { dot_scalar(a, b, len) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        DotKernel::Avx2Fma => unsafe { dot_avx2_fma_impl(a, b, len) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        DotKernel::Avx512Fma => unsafe { dot_avx512_fma_impl(a, b, len) },
    }
}

#[inline(always)]
unsafe fn dot_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut i = 0usize;
    while i + 4 <= len {
        let a0 = unsafe { *a.add(i) };
        let b0 = unsafe { *b.add(i) };
        let a1 = unsafe { *a.add(i + 1) };
        let b1 = unsafe { *b.add(i + 1) };
        let a2 = unsafe { *a.add(i + 2) };
        let b2 = unsafe { *b.add(i + 2) };
        let a3 = unsafe { *a.add(i + 3) };
        let b3 = unsafe { *b.add(i + 3) };

        sum0 = a0.mul_add(b0, sum0);
        sum1 = a1.mul_add(b1, sum1);
        sum2 = a2.mul_add(b2, sum2);
        sum3 = a3.mul_add(b3, sum3);
        i += 4;
    }

    let mut sum = (sum0 + sum1) + (sum2 + sum3);

    while i < len {
        let av = unsafe { *a.add(i) };
        let bv = unsafe { *b.add(i) };
        sum = av.mul_add(bv, sum);
        i += 1;
    }

    sum
}

#[cfg(target_arch = "x86")]
mod x86 {
    use core::arch::x86::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2_fma_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe fn hsum256(v: __m256) -> f32 {
            let mut tmp = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), v) };
            tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
        }

        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;

        while i + 8 <= len {
            let va = unsafe { _mm256_loadu_ps(a.add(i)) };
            let vb = unsafe { _mm256_loadu_ps(b.add(i)) };
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }

        let mut sum = unsafe { hsum256(acc) };
        while i < len {
            let av = unsafe { *a.add(i) };
            let bv = unsafe { *b.add(i) };
            sum = av.mul_add(bv, sum);
            i += 1;
        }

        sum
    }

    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn dot_avx512_fma_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe fn hsum512(v: __m512) -> f32 {
            let mut tmp = [0.0f32; 16];
            unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), v) };
            tmp[0]
                + tmp[1]
                + tmp[2]
                + tmp[3]
                + tmp[4]
                + tmp[5]
                + tmp[6]
                + tmp[7]
                + tmp[8]
                + tmp[9]
                + tmp[10]
                + tmp[11]
                + tmp[12]
                + tmp[13]
                + tmp[14]
                + tmp[15]
        }

        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;

        while i + 16 <= len {
            let va = unsafe { _mm512_loadu_ps(a.add(i)) };
            let vb = unsafe { _mm512_loadu_ps(b.add(i)) };
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }

        let mut sum = unsafe { hsum512(acc) };
        while i < len {
            let av = unsafe { *a.add(i) };
            let bv = unsafe { *b.add(i) };
            sum = av.mul_add(bv, sum);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2_fma_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe fn hsum256(v: __m256) -> f32 {
            let mut tmp = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), v) };
            tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
        }

        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;

        while i + 8 <= len {
            let va = unsafe { _mm256_loadu_ps(a.add(i)) };
            let vb = unsafe { _mm256_loadu_ps(b.add(i)) };
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }

        let mut sum = unsafe { hsum256(acc) };
        while i < len {
            let av = unsafe { *a.add(i) };
            let bv = unsafe { *b.add(i) };
            sum = av.mul_add(bv, sum);
            i += 1;
        }

        sum
    }

    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn dot_avx512_fma_impl(a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe fn hsum512(v: __m512) -> f32 {
            let mut tmp = [0.0f32; 16];
            unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), v) };
            tmp[0]
                + tmp[1]
                + tmp[2]
                + tmp[3]
                + tmp[4]
                + tmp[5]
                + tmp[6]
                + tmp[7]
                + tmp[8]
                + tmp[9]
                + tmp[10]
                + tmp[11]
                + tmp[12]
                + tmp[13]
                + tmp[14]
                + tmp[15]
        }

        let mut acc = _mm512_setzero_ps();
        let mut i = 0usize;

        while i + 16 <= len {
            let va = unsafe { _mm512_loadu_ps(a.add(i)) };
            let vb = unsafe { _mm512_loadu_ps(b.add(i)) };
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }

        let mut sum = unsafe { hsum512(acc) };
        while i < len {
            let av = unsafe { *a.add(i) };
            let bv = unsafe { *b.add(i) };
            sum = av.mul_add(bv, sum);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86")]
use x86::{dot_avx2_fma_impl, dot_avx512_fma_impl};

#[cfg(target_arch = "x86_64")]
use x86_64::{dot_avx2_fma_impl, dot_avx512_fma_impl};
