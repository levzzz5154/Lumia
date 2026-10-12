use std::path::Path;

use lumia_core::DecodedImage;

use super::*;
use crate::load_state::PreparedImage;

#[test]
fn source_span_covers_every_tap_and_stays_in_bounds() {
    // Enlarging 100 -> 1000 with a 4.5 radius: texel 500 centres on 49.75.
    let (start, end) = source_span(100, 1000, 500, 10, 4.5);
    assert!(start <= 45 && end >= 56);
    // Reducing 1000 -> 100 widens the kernel tenfold.
    let (start, end) = source_span(1000, 100, 50, 1, 4.5);
    assert!(start <= 460 && end >= 550);
    assert_eq!(source_span(100, 1000, 0, 1000, 4.5), (0, 100));
    assert_eq!(source_span(64, 64, 10, 20, 4.5), (10, 30));
}

#[test]
fn pass_parameters_match_the_shader_layout() {
    let bytes = Pass {
        source_len: 3,
        target_len: 7,
        target_offset: 1,
        tap_origin: 2,
        cross_offset: 5,
        kernel: 3,
        axis: 1,
        premultiply: 1,
    }
    .bytes();
    assert_eq!(bytes[0..4], 3.0_f32.to_le_bytes());
    assert_eq!(bytes[16..20], 5.0_f32.to_le_bytes());
    assert_eq!(bytes[20..24], 3_u32.to_le_bytes());
    assert_eq!(bytes[28..32], 1_u32.to_le_bytes());
}

/// A real device without a window, or `None` on machines that have no GPU.
fn headless() -> Option<GpuResampler> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = futures::executor::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .ok()?;
    eprintln!("GPU resample tests run on {}", adapter.get_info().name);
    let (device, queue) =
        futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .ok()?;
    Some(GpuResampler::new(Arc::new(device), Arc::new(queue)))
}

const SOURCE: (u32, u32) = (61, 47);

/// Smooth gradients with hard edges and noise, so every kernel lobe matters.
fn source(translucent: bool) -> DecodedImage {
    let mut pixels = Vec::new();
    for y in 0..SOURCE.1 {
        for x in 0..SOURCE.0 {
            let noise = (x * 73 + y * 151 + x * y * 7) % 256;
            let edge = if (x / 9 + y / 7) % 2 == 0 { 230 } else { 20 };
            let alpha = if translucent {
                40 + (x * 3 + y * 2) % 216
            } else {
                255
            };
            pixels.extend([noise as u8, edge as u8, (x * 4) as u8, alpha as u8]);
        }
    }
    DecodedImage {
        pixels_bgra8: pixels,
        width: SOURCE.0,
        height: SOURCE.1,
    }
}

fn key(filter: ResampleFilter, target: (u32, u32), region: ImagePixelRect) -> ResampleKey {
    ResampleKey {
        path: Path::new("a.png").to_path_buf(),
        filter,
        width: target.0,
        height: target.1,
        region: Some(region),
        source_dimensions: Some(SOURCE),
    }
}

fn read_back(gpu: &GpuResampler, texture: &wgpu::Texture) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let stride = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(stride) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    gpu.queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |result| result.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = buffer.get_mapped_range(..);
    mapped
        .chunks_exact(stride as usize)
        .flat_map(|row| row[..width as usize * 4].to_vec())
        .collect()
}

fn on_gpu(gpu: &mut GpuResampler, image: &DecodedImage, key: ResampleKey) -> Vec<u8> {
    let source = ResampleSource::Bitmap(PreparedImage::from_decoded(image.clone()).render_image());
    assert!(gpu
        .render(
            ResampleSlot::MainDisplay,
            ResampleRequest {
                key: key.clone(),
                source
            }
        )
        .is_ok());
    let (ready, texture) = gpu.slots[0].output.clone().unwrap();
    assert_eq!(ready, key);
    read_back(gpu, &texture)
}

fn crop(full: &DecodedImage, region: ImagePixelRect) -> Vec<u8> {
    (region.y..region.y + region.height)
        .flat_map(|y| {
            let start = ((y * full.width + region.x) * 4) as usize;
            full.pixels_bgra8[start..start + region.width as usize * 4].to_vec()
        })
        .collect()
}

fn worst(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap()
}

// Targets and regions cover enlarging and reducing, both pass orders (wide and
// tall regions), offsets, and an axis left at its original length.
const CASES: [((u32, u32), (u32, u32, u32, u32)); 5] = [
    ((244, 188), (30, 20, 160, 40)),
    ((244, 188), (100, 10, 30, 170)),
    ((23, 17), (2, 1, 20, 15)),
    ((61, 141), (5, 30, 50, 90)),
    ((150, 47), (0, 0, 150, 47)),
];

#[test]
fn magic_kernel_sharp_matches_the_cpu_filter() {
    let Some(mut gpu) = headless() else { return };
    let image = source(false);
    for (target, (x, y, width, height)) in CASES {
        let region = ImagePixelRect::new(x, y, width, height);
        let cpu = lumia_core::resize_magic_kernel_sharp_region(
            &image.pixels_bgra8,
            SOURCE,
            target,
            region,
        )
        .unwrap();
        let gpu_pixels = on_gpu(
            &mut gpu,
            &image,
            key(ResampleFilter::MagicKernelSharp, target, region),
        );
        let worst = worst(&gpu_pixels, &cpu.pixels_bgra8);
        assert!(
            worst <= 1,
            "target {target:?} region {region:?}: off by {worst}"
        );
    }
}

#[test]
fn other_filters_match_the_cpu_resize() {
    let Some(mut gpu) = headless() else { return };
    let image = source(false);
    for filter in [
        ResampleFilter::NearestNeighbor,
        ResampleFilter::Bilinear,
        ResampleFilter::Lanczos,
    ] {
        for (target, (x, y, width, height)) in CASES {
            let region = ImagePixelRect::new(x, y, width, height);
            let full =
                lumia_core::resize_decoded_image(&image, filter, target.0, target.1).unwrap();
            let gpu_pixels = on_gpu(&mut gpu, &image, key(filter, target, region));
            let worst = worst(&gpu_pixels, &crop(&full, region));
            assert!(
                worst <= 2,
                "{filter:?} target {target:?} region {region:?}: off by {worst}"
            );
        }
    }
}

#[test]
fn translucent_pixels_match_the_cpu_filter_once_premultiplied() {
    let Some(mut gpu) = headless() else { return };
    let image = source(true);
    let target = (244, 188);
    let region = ImagePixelRect::new(30, 20, 160, 40);
    let cpu =
        lumia_core::resize_magic_kernel_sharp_region(&image.pixels_bgra8, SOURCE, target, region)
            .unwrap();
    let gpu_pixels = on_gpu(
        &mut gpu,
        &image,
        key(ResampleFilter::MagicKernelSharp, target, region),
    );
    let premultiplied = |pixels: &[u8]| -> Vec<u8> {
        pixels
            .chunks_exact(4)
            .flat_map(|p| {
                let alpha = u32::from(p[3]);
                [0, 1, 2]
                    .map(|c| ((u32::from(p[c]) * alpha + 127) / 255) as u8)
                    .into_iter()
                    .chain([p[3]])
            })
            .collect()
    };
    let worst = worst(
        &premultiplied(&gpu_pixels),
        &premultiplied(&cpu.pixels_bgra8),
    );
    assert!(worst <= 2, "off by {worst}");
}

#[test]
fn unscaled_regions_are_copied_exactly() {
    let Some(mut gpu) = headless() else { return };
    let image = source(true);
    let region = ImagePixelRect::new(7, 5, 40, 30);
    for filter in [ResampleFilter::Lanczos, ResampleFilter::MagicKernelSharp] {
        let gpu_pixels = on_gpu(&mut gpu, &image, key(filter, SOURCE, region));
        assert_eq!(gpu_pixels, crop(&image, region));
    }
}
