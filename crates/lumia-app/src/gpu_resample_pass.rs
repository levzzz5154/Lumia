//! Per-pass parameters of the GPU resampler, as laid out in `gpu_resample.wgsl`.

use lumia_core::ResampleFilter;

pub(crate) const PARAMS_SIZE: u64 = 48;

pub(crate) fn kernel(filter: ResampleFilter) -> (u32, f64) {
    match filter {
        ResampleFilter::NearestNeighbor => (0, 0.5),
        ResampleFilter::Bilinear => (1, 1.0),
        ResampleFilter::Lanczos => (2, 3.0),
        ResampleFilter::MagicKernelSharp => (3, 4.5),
    }
}

/// Source texels `[start, end)` that destination texels `[first, first + len)`
/// can read on one axis, padded by one for float rounding in the shader.
pub(crate) fn source_span(
    source: u32,
    target: u32,
    first: u32,
    len: u32,
    radius: f64,
) -> (u32, u32) {
    if source == target {
        return (first.min(source - 1), (first + len).min(source));
    }
    let scale = f64::from(source) / f64::from(target);
    let reach = radius * scale.max(1.0) + 1.0;
    let center = |index: u32| (f64::from(index) + 0.5) * scale - 0.5;
    let start = (center(first) - reach).floor().max(0.0) as u32;
    let end = ((center(first + len - 1) + reach).ceil() as u32 + 1).min(source);
    (start.min(source - 1), end)
}

pub(crate) struct Pass {
    pub(crate) source_len: u32,
    pub(crate) target_len: u32,
    pub(crate) target_offset: u32,
    pub(crate) tap_origin: u32,
    pub(crate) cross_offset: u32,
    pub(crate) kernel: u32,
    pub(crate) axis: u32,
    pub(crate) premultiply: u32,
    /// `(source_len, target_len, target_offset)` of the other axis when the
    /// pixel grid is drawn.
    pub(crate) grid: Option<(u32, u32, u32)>,
}

impl Pass {
    pub(crate) fn bytes(&self) -> [u8; PARAMS_SIZE as usize] {
        let mut bytes = [0; PARAMS_SIZE as usize];
        let floats = [
            self.source_len,
            self.target_len,
            self.target_offset,
            self.tap_origin,
            self.cross_offset,
        ];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip(floats) {
            chunk.copy_from_slice(&(value as f32).to_le_bytes());
        }
        for (chunk, value) in
            bytes[20..]
                .chunks_exact_mut(4)
                .zip([self.kernel, self.axis, self.premultiply])
        {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        if let Some((source_len, target_len, target_offset)) = self.grid {
            for (chunk, value) in
                bytes[32..]
                    .chunks_exact_mut(4)
                    .zip([source_len, target_len, target_offset])
            {
                chunk.copy_from_slice(&(value as f32).to_le_bytes());
            }
            bytes[44..].copy_from_slice(&1_u32.to_le_bytes());
        }
        bytes
    }
}
