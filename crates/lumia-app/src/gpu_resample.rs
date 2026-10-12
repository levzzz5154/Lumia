//! Display resampling on the window's own wgpu device (gpui-ce on Linux).
//!
//! The source image is uploaded once; each visible region is then filtered in
//! two passes straight into a texture that GPUI composites, with no readback.

use std::sync::Arc;

use lumia_core::ImagePixelRect;

use crate::display_resample::{ResampleKey, ResampleRequest, ResampleSlot, ResampleSource};
use crate::gpu_resample_pass::{kernel, source_span, Pass, PARAMS_SIZE};

// Half floats keep the signed, premultiplied sums between the two passes.
const INTERMEDIATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
// Physical pixels per source pixel from which a grid line does not crowd it.
const MIN_GRID_PIXEL: u32 = 8;

struct Source {
    /// Address and length of the pixels the texture was uploaded from.
    identity: (usize, usize),
    dimensions: (u32, u32),
    view: wgpu::TextureView,
    /// Keeps the pixels alive so `identity` cannot be reused by another image.
    _pixels: ResampleSource,
}

#[derive(Default)]
struct Slot {
    source: Option<Source>,
    output: Option<(ResampleKey, Arc<wgpu::Texture>)>,
}

pub(crate) struct GpuResampler {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    first_pass: wgpu::RenderPipeline,
    second_pass: wgpu::RenderPipeline,
    params: [wgpu::Buffer; 2],
    max_dimension: u32,
    slots: [Slot; 3],
}

fn slot_index(slot: ResampleSlot) -> usize {
    match slot {
        ResampleSlot::MainDisplay => 0,
        ResampleSlot::ComparisonLeft => 1,
        ResampleSlot::ComparisonRight => 2,
    }
}

fn source_pixels(source: &ResampleSource) -> Option<(&[u8], (u32, u32))> {
    match source {
        ResampleSource::Raster(raster) => Some((raster.pixels(), raster.dimensions())),
        ResampleSource::Bitmap(image) => {
            let size = image.size(0);
            let dimensions = (
                u32::try_from(size.width.0).ok()?,
                u32::try_from(size.height.0).ok()?,
            );
            Some((image.as_bytes(0)?, dimensions))
        }
    }
}

impl GpuResampler {
    pub(crate) fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lumia_resample"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_resample.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lumia_resample_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lumia_resample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let params = [(); 2].map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lumia_resample_params"),
                size: PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Self {
            first_pass: pipeline("fs_first", INTERMEDIATE_FORMAT),
            second_pass: pipeline("fs_second", DISPLAY_FORMAT),
            max_dimension: device.limits().max_texture_dimension_2d,
            layout,
            params,
            slots: Default::default(),
            device,
            queue,
        }
    }

    /// The slot's display texture, when it holds exactly `key`.
    pub(crate) fn output(
        &self,
        slot: ResampleSlot,
        key: &ResampleKey,
    ) -> Option<(Arc<wgpu::Texture>, ImagePixelRect)> {
        let (ready, texture) = self.slots[slot_index(slot)].output.as_ref()?;
        (ready == key).then(|| Some((texture.clone(), key.region?)))?
    }

    pub(crate) fn has_output(&self, slot: ResampleSlot) -> bool {
        self.slots[slot_index(slot)].output.is_some()
    }

    pub(crate) fn clear(&mut self, slot: ResampleSlot) {
        self.slots[slot_index(slot)] = Slot::default();
    }

    fn texture(
        &self,
        label: &str,
        (width, height): (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn upload(&self, pixels: &[u8], dimensions: (u32, u32)) -> wgpu::TextureView {
        let texture = self.texture(
            "lumia_resample_source",
            dimensions,
            DISPLAY_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions.0 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        params: &wgpu::Buffer,
        pass: Pass,
        input: &wgpu::TextureView,
        target: &wgpu::TextureView,
    ) {
        self.queue.write_buffer(params, 0, &pass.bytes());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lumia_resample_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lumia_resample_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Filter `request`'s region into the slot's display texture. The request
    /// is handed back when this device cannot hold the image, so the caller
    /// can fall back to the CPU path.
    pub(crate) fn render(
        &mut self,
        slot: ResampleSlot,
        request: ResampleRequest,
    ) -> Result<(), ResampleRequest> {
        let index = slot_index(slot);
        let max = self.max_dimension;
        let (Some(region), Some((pixels, dimensions))) =
            (request.key.region, source_pixels(&request.source))
        else {
            return Err(request);
        };
        let fits = [dimensions.0, dimensions.1, region.width, region.height]
            .iter()
            .all(|side| (1..=max).contains(side))
            && pixels.len() == dimensions.0 as usize * dimensions.1 as usize * 4;
        if !fits {
            self.clear(slot);
            return Err(request);
        }

        let identity = (pixels.as_ptr() as usize, pixels.len());
        let same_source = self.slots[index]
            .source
            .as_ref()
            .is_some_and(|source| source.identity == identity);
        if same_source
            && self.slots[index]
                .output
                .as_ref()
                .is_some_and(|(ready, _)| *ready == request.key)
        {
            return Ok(());
        }
        if !same_source {
            let view = self.upload(pixels, dimensions);
            self.slots[index] = Slot {
                source: Some(Source {
                    identity,
                    dimensions,
                    view,
                    _pixels: request.source,
                }),
                output: None,
            };
        }
        let ResampleKey { width, height, .. } = request.key;
        // Pixel view only makes sense while every source pixel is enlarged.
        let pixel_view = request
            .key
            .pixels
            .filter(|_| width >= dimensions.0 && height >= dimensions.1);
        let (kernel, radius) = match pixel_view {
            Some(_) => (4, 1.0),
            None => kernel(request.key.filter),
        };
        let grid = pixel_view.is_some_and(|view| {
            view.grid
                && width / dimensions.0 >= MIN_GRID_PIXEL
                && height / dimensions.1 >= MIN_GRID_PIXEL
        });

        // Filter first along the axis that leaves the smaller intermediate.
        let span_x = source_span(dimensions.0, width, region.x, region.width, radius);
        let span_y = source_span(dimensions.1, height, region.y, region.height, radius);
        let x_first = u64::from(region.width) * u64::from(span_y.1 - span_y.0)
            <= u64::from(span_x.1 - span_x.0) * u64::from(region.height);
        let (intermediate_size, first, second) = if x_first {
            (
                (region.width, span_y.1 - span_y.0),
                (dimensions.0, width, region.x, span_y.0),
                (dimensions.1, height, region.y, span_y.0),
            )
        } else {
            (
                (span_x.1 - span_x.0, region.height),
                (dimensions.1, height, region.y, span_x.0),
                (dimensions.0, width, region.x, span_x.0),
            )
        };
        let first_axis = u32::from(!x_first);

        let attachment =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let intermediate = self
            .texture(
                "lumia_resample_intermediate",
                intermediate_size,
                INTERMEDIATE_FORMAT,
                attachment,
            )
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Reuse the display texture while the region keeps its size.
        let output = self.slots[index]
            .output
            .take()
            .map(|(_, texture)| texture)
            .filter(|texture| (texture.width(), texture.height()) == (region.width, region.height))
            .unwrap_or_else(|| {
                Arc::new(self.texture(
                    "lumia_resample_output",
                    (region.width, region.height),
                    DISPLAY_FORMAT,
                    attachment | wgpu::TextureUsages::COPY_SRC,
                ))
            });
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lumia_resample"),
            });
        let source = self.slots[index].source.as_ref().expect("uploaded source");
        debug_assert_eq!(source.dimensions, dimensions);
        self.draw(
            &mut encoder,
            &self.first_pass,
            &self.params[0],
            Pass {
                source_len: first.0,
                target_len: first.1,
                target_offset: first.2,
                tap_origin: 0,
                cross_offset: first.3,
                kernel,
                axis: first_axis,
                premultiply: 1,
                grid: None,
            },
            &source.view,
            &intermediate,
        );
        self.draw(
            &mut encoder,
            &self.second_pass,
            &self.params[1],
            Pass {
                source_len: second.0,
                target_len: second.1,
                target_offset: second.2,
                tap_origin: second.3,
                cross_offset: 0,
                kernel,
                axis: 1 - first_axis,
                premultiply: 0,
                grid: grid.then_some((first.0, first.1, first.2)),
            },
            &intermediate,
            &output_view,
        );
        self.queue.submit([encoder.finish()]);
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            eprintln!(
                "lumia: first GPU region {}x{} of a {width}x{height} grid from {dimensions:?}, {:?}",
                region.width, region.height, request.key.filter
            );
        });
        self.slots[index].output = Some((request.key, output));
        Ok(())
    }
}

#[cfg(test)]
#[path = "gpu_resample_tests.rs"]
mod tests;
