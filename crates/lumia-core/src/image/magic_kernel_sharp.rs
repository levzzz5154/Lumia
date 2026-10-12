//! Separable Magic Kernel Sharp 2021 display resampling.

use super::{DecodedImage, ImageLoadError};

// Full direct MKS2021 kernel (support radius 4.5), including all Sharp lobes.
// https://johncostella.com/magic/
// Also independently implemented as MagicKernelSharp2021 in ImageMagick.
// Widen the entire kernel when reducing to suppress aliasing.
fn kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 0.5 {
        577.0 / 576.0 - 239.0 / 144.0 * x * x
    } else if x < 1.5 {
        35.0 / 36.0 * (x - 1.0) * (x - 239.0 / 140.0)
    } else if x < 2.5 {
        (x - 2.0) * (65.0 / 24.0 - x) / 6.0
    } else if x < 3.5 {
        (x - 3.0) * (x - 3.75) / 36.0
    } else if x < 4.5 {
        -(x - 4.5).powi(2) / 288.0
    } else {
        0.0
    }
}

/// Normalized kernel weights for one output sample, over the contiguous
/// source run starting at `first`.
struct Taps {
    first: usize,
    weights: Vec<f32>,
}

fn weights(source_len: u32, target_len: u32, target: u32) -> Taps {
    // An unchanged axis must not acquire the small residual blur/sharpening
    // of this finite approximation to the exact interpolating kernel.
    if source_len == target_len {
        return Taps {
            first: target as usize,
            weights: vec![1.0],
        };
    }
    let scale = f64::from(source_len) / f64::from(target_len);
    let stretch = scale.max(1.0);
    let center = (f64::from(target) + 0.5) * scale - 0.5;
    let radius = 4.5 * stretch;
    let first = ((center - radius).ceil() as i64).max(0);
    let last = ((center + radius).floor() as i64).min(i64::from(source_len) - 1);
    let taps: Vec<f64> = (first..=last)
        .map(|i| kernel((i as f64 - center) / stretch))
        .collect();
    let sum: f64 = taps.iter().sum();
    Taps {
        first: first as usize,
        weights: taps.iter().map(|weight| (weight / sum) as f32).collect(),
    }
}

// Below this many output rows a band is not worth its own thread.
const MIN_BAND_ROWS: usize = 32;

/// Filter `y_taps.len()` output rows into `output`. Horizontally filtered
/// source rows are kept only while a later output row still needs them.
fn filter_band(
    pixels: &[u8],
    width: usize,
    x_taps: &[Taps],
    y_taps: &[Taps],
    output: &mut [u8],
) -> Result<(), ImageLoadError> {
    let row_len = x_taps.len() * 4;
    let span_start = x_taps.iter().map(|t| t.first).min().unwrap_or(0);
    let span_end = x_taps
        .iter()
        .map(|t| t.first + t.weights.len())
        .max()
        .unwrap_or(0);
    // One source row, premultiplied, so the horizontal taps are plain sums.
    let mut line = allocate::<f32>((span_end - span_start) * 4)?;
    let mut rows = std::collections::VecDeque::<Vec<f32>>::new();
    let mut spare = Vec::new();
    let mut base = 0;
    let mut sums = allocate::<f32>(row_len)?;
    for (taps, out) in y_taps.iter().zip(output.chunks_exact_mut(row_len)) {
        while base < taps.first {
            match rows.pop_front() {
                Some(row) => spare.push(row),
                None => {
                    base = taps.first;
                    break;
                }
            }
            base += 1;
        }
        for src_y in base + rows.len()..taps.first + taps.weights.len() {
            let start = (src_y * width + span_start) * 4;
            for (dst, src) in line
                .chunks_exact_mut(4)
                .zip(pixels[start..].chunks_exact(4))
            {
                let alpha = f32::from(src[3]) / 255.0;
                for c in 0..3 {
                    dst[c] = f32::from(src[c]) * alpha;
                }
                dst[3] = f32::from(src[3]);
            }
            let mut row = match spare.pop() {
                Some(row) => row,
                None => allocate::<f32>(row_len)?,
            };
            for (dst, horizontal) in row.chunks_exact_mut(4).zip(x_taps) {
                let mut channels = [0.0_f32; 4];
                let start = (horizontal.first - span_start) * 4;
                for (src, &weight) in line[start..].chunks_exact(4).zip(&horizontal.weights) {
                    for c in 0..4 {
                        channels[c] += src[c] * weight;
                    }
                }
                dst.copy_from_slice(&channels);
            }
            rows.push_back(row);
        }
        sums.fill(0.0);
        for (row, &weight) in rows.iter().skip(taps.first - base).zip(&taps.weights) {
            for (sum, value) in sums.iter_mut().zip(row) {
                *sum += value * weight;
            }
        }
        for (dst, channels) in out.chunks_exact_mut(4).zip(sums.chunks_exact(4)) {
            dst[3] = channels[3].clamp(0.0, 255.0).round() as u8;
            if dst[3] > 0 {
                for c in 0..3 {
                    dst[c] = (channels[c] * 255.0 / channels[3])
                        .clamp(0.0, 255.0)
                        .round() as u8;
                }
            }
        }
    }
    Ok(())
}

/// Split the output rows into `bands` and filter them on separate threads.
/// Bands share no state, so the result does not depend on the band count.
fn filter_rows(
    pixels: &[u8],
    width: usize,
    x_taps: &[Taps],
    y_taps: &[Taps],
    output: &mut [u8],
    bands: usize,
) -> Result<(), ImageLoadError> {
    let band_rows = y_taps.len().div_ceil(bands.max(1)).max(1);
    let row_len = x_taps.len() * 4;
    std::thread::scope(|scope| {
        let jobs: Vec<_> = y_taps
            .chunks(band_rows)
            .zip(output.chunks_mut(band_rows * row_len))
            .map(|(taps, out)| scope.spawn(move || filter_band(pixels, width, x_taps, taps, out)))
            .collect();
        jobs.into_iter()
            .try_for_each(|job| job.join().expect("Magic Kernel Sharp band panicked"))
    })
}

pub(super) fn resize(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<DecodedImage, ImageLoadError> {
    resize_magic_kernel_sharp_region(
        pixels,
        (width, height),
        (target_width, target_height),
        super::ImagePixelRect::new(0, 0, target_width, target_height),
    )
}

/// Resample a rectangle in the full output coordinate system. Kernel phase and
/// support are identical to a full resize; source pixels outside the visible
/// rectangle still contribute. Scratch storage is bounded to filtered rows.
pub fn resize_magic_kernel_sharp_region(
    pixels: &[u8],
    source: (u32, u32),
    target: (u32, u32),
    region: super::ImagePixelRect,
) -> Result<DecodedImage, ImageLoadError> {
    let (width, height) = source;
    let (target_width, target_height) = target;
    let valid = width > 0
        && height > 0
        && target_width > 0
        && target_height > 0
        && region.width > 0
        && region.height > 0
        && region
            .x
            .checked_add(region.width)
            .is_some_and(|v| v <= target_width)
        && region
            .y
            .checked_add(region.height)
            .is_some_and(|v| v <= target_height)
        && super::checked_bgra8_len(width, height) == Some(pixels.len());
    if !valid {
        return Err(ImageLoadError::HeifMetadata {
            path: "(memory)".into(),
            message: "invalid Magic Kernel Sharp source or output region".into(),
        });
    }
    let output_len = super::checked_bgra8_len(region.width, region.height).ok_or(
        ImageLoadError::MemoryLimit {
            bytes: u64::MAX,
            limit: isize::MAX as u64,
        },
    )?;
    let mut output = allocate::<u8>(output_len)?;
    if source == target {
        for y in 0..region.height as usize {
            let src = ((region.y as usize + y) * width as usize + region.x as usize) * 4;
            let dst = y * region.width as usize * 4;
            output[dst..dst + region.width as usize * 4]
                .copy_from_slice(&pixels[src..src + region.width as usize * 4]);
        }
    } else {
        let x_taps: Vec<_> = (region.x..region.x + region.width)
            .map(|x| weights(width, target_width, x))
            .collect();
        let y_taps: Vec<_> = (region.y..region.y + region.height)
            .map(|y| weights(height, target_height, y))
            .collect();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let bands = threads.min(y_taps.len() / MIN_BAND_ROWS).max(1);
        filter_rows(pixels, width as usize, &x_taps, &y_taps, &mut output, bands)?;
    }
    Ok(DecodedImage {
        pixels_bgra8: output,
        width: region.width,
        height: region.height,
    })
}

// Reserve fallibly, including the element size in the overflow check.
fn allocate<T: Default + Clone>(len: usize) -> Result<Vec<T>, ImageLoadError> {
    let bytes = (len as u64).saturating_mul(std::mem::size_of::<T>() as u64);
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| ImageLoadError::MemoryLimit {
            bytes,
            limit: isize::MAX as u64,
        })?;
    buffer.resize(len, T::default());
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regional_output_matches_full_resize_at_every_scale() {
        let pixels: Vec<u8> = (0..13 * 11 * 4).map(|i| ((i * 71) % 256) as u8).collect();
        for (w, h) in [(7, 5), (31, 29), (13, 11), (13, 31), (31, 11)] {
            let full = resize(&pixels, 13, 11, w, h).unwrap();
            for (x, y) in [(0, 0), (w / 2, h / 2), (w - 1, h - 1)] {
                let region =
                    super::super::ImagePixelRect::new(x, y, (w - x).min(5), (h - y).min(4));
                let crop =
                    resize_magic_kernel_sharp_region(&pixels, (13, 11), (w, h), region).unwrap();
                for row in 0..region.height as usize {
                    let start = ((y as usize + row) * w as usize + x as usize) * 4;
                    let dst = row * region.width as usize * 4;
                    assert_eq!(
                        &crop.pixels_bgra8[dst..dst + region.width as usize * 4],
                        &full.pixels_bgra8[start..start + region.width as usize * 4]
                    );
                }
            }
        }
    }

    #[test]
    fn banded_output_matches_a_single_band_when_enlarging_and_reducing() {
        let pixels: Vec<u8> = (0..40 * 50 * 4).map(|i| ((i * 37) % 256) as u8).collect();
        for (w, h) in [(90u32, 131u32), (17, 23)] {
            let x_taps: Vec<_> = (0..w).map(|x| weights(40, w, x)).collect();
            let y_taps: Vec<_> = (0..h).map(|y| weights(50, h, y)).collect();
            let mut single = vec![0; (w * h * 4) as usize];
            let mut banded = single.clone();
            filter_rows(&pixels, 40, &x_taps, &y_taps, &mut single, 1).unwrap();
            filter_rows(&pixels, 40, &x_taps, &y_taps, &mut banded, 5).unwrap();
            assert_eq!(single, banded);
        }
    }

    #[test]
    fn invalid_regions_and_sources_return_errors() {
        for region in [
            super::super::ImagePixelRect::new(9, 0, 2, 1),
            super::super::ImagePixelRect::new(u32::MAX, 0, 2, 1),
            super::super::ImagePixelRect::new(0, 0, 0, 1),
        ] {
            assert!(resize_magic_kernel_sharp_region(&[0; 4], (1, 1), (10, 10), region).is_err());
        }
        assert!(resize(&[], 0, 0, 1, 1).is_err());
    }

    #[test]
    fn full_kernel_matches_magic_convolved_with_seven_tap_sharp() {
        // Independent construction: quadratic B-spline plus the 2021
        // discrete Sharp coefficients, rather than the piecewise polynomial.
        let sharp = [-1.0, 6.0, -35.0, 204.0, -35.0, 6.0, -1.0];
        for sample in -500..=500 {
            let x = f64::from(sample) / 100.0;
            let expected: f64 = sharp
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let t = (x - (i as f64 - 3.0)).abs();
                    let magic = if t < 0.5 {
                        0.75 - t * t
                    } else if t < 1.5 {
                        0.5 * (t - 1.5).powi(2)
                    } else {
                        0.0
                    };
                    c / 144.0 * magic
                })
                .sum();
            assert!((kernel(x) - expected).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn identity_preserves_all_bytes_including_hidden_colors() {
        let pixels = [255, 7, 19, 0, 30, 90, 170, 128, 0, 255, 0, 255];
        assert_eq!(resize(&pixels, 3, 1, 3, 1).unwrap().pixels_bgra8, pixels);
    }

    #[test]
    fn alpha_ringing_preserves_visible_color() {
        let pixels = [40, 80, 120, 0, 40, 80, 120, 255];
        let result = resize(&pixels, 2, 1, 32, 1).unwrap();
        for p in result.pixels_bgra8.chunks_exact(4) {
            if p[3] != 0 {
                assert_eq!(&p[..3], &[40, 80, 120]);
            } else {
                assert_eq!(&p[..3], &[0, 0, 0]);
            }
        }
    }

    #[test]
    fn reduction_suppresses_checkerboard_aliasing() {
        let pixels: Vec<u8> = (0..64)
            .flat_map(|i| {
                let value = if i % 2 == 0 { 0 } else { 255 };
                [value, value, value, 255]
            })
            .collect();
        let result = resize(&pixels, 64, 1, 8, 1).unwrap();
        for p in result.pixels_bgra8.chunks_exact(4).skip(2).take(4) {
            assert!((i32::from(p[0]) - 128).abs() <= 1);
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn constant_image_stays_constant_at_both_scales() {
        let pixels = [40, 80, 160, 255].repeat(16);
        for (width, height) in [(2, 2), (9, 7)] {
            let resized = resize(&pixels, 4, 4, width, height).unwrap();
            assert!(resized
                .pixels_bgra8
                .chunks_exact(4)
                .all(|p| p == [40, 80, 160, 255]));
        }
    }

    #[test]
    fn transparent_colors_do_not_bleed_into_opaque_pixels() {
        let pixels = [255, 0, 0, 0, 0, 0, 255, 255];
        let resized = resize(&pixels, 2, 1, 7, 1).unwrap();
        for pixel in resized.pixels_bgra8.chunks_exact(4) {
            if pixel[3] > 0 {
                assert_eq!(pixel[0], 0);
                assert_eq!(pixel[1], 0);
            }
        }
    }
}
