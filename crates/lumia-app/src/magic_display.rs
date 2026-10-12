//! Strict MKS display path: filter only visible output pixels, never an
//! unfiltered preview or an oversized full-image texture as a substitute for
//! that output. Only earlier MKS output may stand in while a job is running.
use std::path::Path;

use gpui_kit::{
    canvas, div, point, px, size, AnyElement, Bounds, Context, IntoElement, ParentElement, Styled,
    Window,
};
use lumia_core::{ImagePixelRect, ResampleFilter};

use crate::{
    app::LumiaApp,
    display_resample::{
        ResampleKey, ResamplePreview, ResampleRequest, ResampleSlot, ResampleSource,
    },
};

// The filtered region extends past the viewport so a pan is served from
// pixels that already exist. Its edges sit on a coarse grid: the request only
// changes when the view crosses a grid line, and by then the previous region
// still has `OVERSCAN - SNAP` physical pixels of travel left in it.
const SNAP: u32 = 256;
const OVERSCAN: u32 = 512;

#[derive(Clone)]
struct Geometry {
    key: ResampleKey,
    /// Top-left of the visible part of the image, in logical pixels.
    visible: (f32, f32),
}

fn geometry(
    path: &Path,
    filter: ResampleFilter,
    dimensions: (u32, u32),
    display: (f32, f32),
    viewport: (f32, f32),
    pan: (f32, f32),
    density: f32,
) -> Option<Geometry> {
    if ![display.0, display.1, viewport.0, viewport.1, density]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
    {
        return None;
    }
    let width = (display.0 * density).round().max(1.0) as u32;
    let height = (display.1 * density).round().max(1.0) as u32;
    // Work in the full destination's pixel grid, preserving filter phase on pan.
    let left = (viewport.0 - display.0) * 0.5 + pan.0;
    let top = (viewport.1 - display.1) * 0.5 + pan.1;
    let x_scale = width as f32 / display.0;
    let y_scale = height as f32 / display.1;
    let x = ((-left * x_scale).floor().max(0.0) as u32).min(width);
    let y = ((-top * y_scale).floor().max(0.0) as u32).min(height);
    let right = (((viewport.0 - left) * x_scale).ceil().max(0.0) as u32).min(width);
    let bottom = (((viewport.1 - top) * y_scale).ceil().max(0.0) as u32).min(height);
    if x >= right || y >= bottom {
        return None;
    }
    let visible = (x as f32 / x_scale, y as f32 / y_scale);
    let grow = |start: u32, end: u32, limit: u32| {
        (
            (start / SNAP * SNAP).saturating_sub(OVERSCAN),
            (end.div_ceil(SNAP) * SNAP)
                .saturating_add(OVERSCAN)
                .min(limit),
        )
    };
    let (x, right) = grow(x, right, width);
    let (y, bottom) = grow(y, bottom, height);
    Some(Geometry {
        key: ResampleKey {
            path: path.to_path_buf(),
            filter,
            width,
            height,
            region: Some(ImagePixelRect::new(x, y, right - x, bottom - y)),
            source_dimensions: Some(dimensions),
        },
        visible,
    })
}

impl LumiaApp {
    fn magic_geometry(&self, slot: ResampleSlot, window: &Window) -> Option<Geometry> {
        let main = !matches!(slot, ResampleSlot::ComparisonRight);
        let state = self.comparison.as_ref();
        let (path, dimensions, display, viewport_size, viewport) = if main {
            let dimensions = self.viewer.display_dimensions()?;
            let viewport = self.viewer.viewport();
            if matches!(slot, ResampleSlot::ComparisonLeft) {
                let bounds = state?.left_pane_bounds?;
                (
                    self.image_path()?,
                    dimensions,
                    crate::comparison::pane_display_size(viewport, dimensions, Some(bounds.size))?,
                    (f32::from(bounds.size.width), f32::from(bounds.size.height)),
                    viewport,
                )
            } else {
                (
                    self.image_path()?,
                    dimensions,
                    self.scaled_image_size(window)?,
                    self.viewer_available_size(window),
                    viewport,
                )
            }
        } else {
            let state = state?;
            let dimensions = state.display_image()?.dimensions();
            let bounds = state.right_pane_bounds?;
            (
                &*state.path,
                dimensions,
                crate::comparison::pane_display_size(
                    &state.viewport,
                    dimensions,
                    Some(bounds.size),
                )?,
                (f32::from(bounds.size.width), f32::from(bounds.size.height)),
                &state.viewport,
            )
        };
        let source_dimensions = self
            .magic_source(slot)
            .map(|source| match source {
                ResampleSource::Raster(raster) => raster.dimensions(),
                ResampleSource::Bitmap(image) => {
                    let size = image.size(0);
                    (size.width.0 as u32, size.height.0 as u32)
                }
            })
            .unwrap_or(dimensions);
        geometry(
            path,
            self.settings.resample_filter,
            source_dimensions,
            display,
            viewport_size,
            (viewport.pan_x, viewport.pan_y),
            window.scale_factor(),
        )
    }

    fn magic_source(&self, slot: ResampleSlot) -> Option<ResampleSource> {
        if matches!(slot, ResampleSlot::ComparisonRight) {
            return Some(ResampleSource::Bitmap(
                self.comparison.as_ref()?.display_image()?.render_image(),
            ));
        }
        if self.viewer.rotation_quarter_turns() == 0 {
            if let Some(raster) = self.large_image.raster() {
                return Some(ResampleSource::Raster(raster));
            }
            // Wait for full-resolution pixels rather than magnifying the
            // low-resolution large-image preview.
            if self
                .image_path()
                .is_some_and(|p| self.large_image.is_active(p))
            {
                return None;
            }
        }
        self.loads
            .display_image(self.viewer.rotation_quarter_turns())
            .map(|image| ResampleSource::Bitmap(image.render_image()))
    }

    pub(crate) fn schedule_magic_resamples(&mut self, window: &Window, cx: &mut Context<Self>) {
        if self.viewer.rotation_quarter_turns() == 0 && self.large_image.begin_raster_build() {
            self.start_large_image_raster_build(cx);
        }
        for slot in [
            ResampleSlot::MainDisplay,
            ResampleSlot::ComparisonLeft,
            ResampleSlot::ComparisonRight,
        ] {
            let active = matches!(slot, ResampleSlot::MainDisplay) == self.comparison.is_none();
            let request = active
                .then(|| {
                    let geometry = self.magic_geometry(slot, window)?;
                    Some(ResampleRequest {
                        key: geometry.key,
                        source: self.magic_source(slot)?,
                    })
                })
                .flatten();
            // The GPU filters the region within this frame; anything it
            // cannot hold falls through to the background CPU job.
            let request = match (request, self.gpu_display.prepare(window)) {
                (Some(request), Some(gpu)) => gpu.render(slot, request).err(),
                (request, gpu) => {
                    if let Some(gpu) = gpu {
                        gpu.clear(slot);
                    }
                    request
                }
            };
            let on_gpu = request.is_none() && self.gpu_display.output_ready(slot);
            let handle = self.self_handle.clone();
            let preview = match slot {
                ResampleSlot::MainDisplay => &mut self.display_resample,
                ResampleSlot::ComparisonLeft => {
                    let Some(state) = self.comparison.as_mut() else {
                        continue;
                    };
                    &mut state.left_resample
                }
                ResampleSlot::ComparisonRight => {
                    let Some(state) = self.comparison.as_mut() else {
                        continue;
                    };
                    &mut state.right_resample
                }
            };
            if on_gpu {
                // Nothing is owed by the CPU cache for this slot any more.
                preview.invalidate();
            } else {
                preview.request(request, slot, handle, cx);
            }
        }
    }

    /// Always returns an MKS surface while MKS is selected. Missing or failed
    /// jobs cannot fall through to GPUI's default image sampling; a stale MKS
    /// output covers the gap while a pan or zoom is being refiltered.
    pub(crate) fn render_magic_content(
        &self,
        slot: ResampleSlot,
        window: &Window,
    ) -> Option<AnyElement> {
        if !self.gpu_display.is_active()
            && self.settings.resample_filter != ResampleFilter::MagicKernelSharp
        {
            return None;
        }
        let mut content = div().relative().size_full();
        let preview: Option<&ResamplePreview> = match slot {
            ResampleSlot::MainDisplay => Some(&self.display_resample),
            ResampleSlot::ComparisonLeft => self.comparison.as_ref().map(|s| &s.left_resample),
            ResampleSlot::ComparisonRight => self.comparison.as_ref().map(|s| &s.right_resample),
        };
        let geometry = self.magic_geometry(slot, window);
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        if let Some((texture, region)) = geometry
            .as_ref()
            .and_then(|geometry| self.gpu_display.output(slot, &geometry.key))
        {
            let density = window.scale_factor();
            // One texel per physical pixel: the compositor's sampler then
            // returns the filtered texels unchanged.
            content = content.child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        let origin = point(
                            px((f32::from(bounds.origin.x) * density).round() / density),
                            px((f32::from(bounds.origin.y) * density).round() / density),
                        );
                        let target = Bounds::new(
                            origin
                                + point(
                                    px(region.x as f32 / density),
                                    px(region.y as f32 / density),
                                ),
                            size(
                                px(region.width as f32 / density),
                                px(region.height as f32 / density),
                            ),
                        );
                        window.paint_surface(
                            target,
                            texture,
                            size(
                                gpui_kit::DevicePixels(region.width as i32),
                                gpui_kit::DevicePixels(region.height as i32),
                            ),
                        );
                    },
                )
                .absolute()
                .size_full(),
            );
            return Some(content.into_any_element());
        }
        if let (Some(geometry), Some(preview)) = (geometry, preview) {
            // Between jobs the last output stays up: unchanged after a pan,
            // stretched onto the new grid after a zoom.
            if let Some((image, region, grid)) = preview.latest_region(&geometry.key) {
                let x_scale = geometry.key.width as f32 / grid.0 as f32;
                let y_scale = geometry.key.height as f32 / grid.1 as f32;
                let image = image.render_image();
                let density = window.scale_factor();
                // Place MKS output texels directly on physical screen pixels
                // (scale 1 once the job for this grid is ready). Fractional
                // layout sizes must not trigger a second GPU resize.
                content = content.child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            let origin = point(
                                px((f32::from(bounds.origin.x) * density).round() / density),
                                px((f32::from(bounds.origin.y) * density).round() / density),
                            );
                            let target = Bounds::new(
                                origin
                                    + point(
                                        px(region.x as f32 * x_scale / density),
                                        px(region.y as f32 * y_scale / density),
                                    ),
                                size(
                                    px(region.width as f32 * x_scale / density),
                                    px(region.height as f32 * y_scale / density),
                                ),
                            );
                            if let Err(error) = window.paint_image(
                                target,
                                target,
                                Default::default(),
                                image,
                                0,
                                false,
                            ) {
                                eprintln!("Magic Kernel Sharp display failed: {error}");
                            }
                        },
                    )
                    .absolute()
                    .size_full(),
                );
            } else {
                let message = preview.error.clone().unwrap_or_else(|| {
                    crate::i18n::tr(self.settings.language, crate::i18n::TextKey::LoadingImage)
                        .to_string()
                });
                content = content.child(
                    div()
                        .absolute()
                        .left(px(geometry.visible.0))
                        .top(px(geometry.visible.1))
                        .child(message),
                );
            }
        }
        Some(content.into_any_element())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_zoom_keeps_full_scale_but_allocates_only_visible_pixels() {
        let g = geometry(
            Path::new("a.png"),
            ResampleFilter::MagicKernelSharp,
            (4000, 3000),
            (40000.0, 30000.0),
            (1200.0, 800.0),
            (0.0, 0.0),
            2.0,
        )
        .unwrap();
        assert_eq!((g.key.width, g.key.height), (80000, 60000));
        // Visible 38800..41200 x 29200..30800, snapped out and overscanned.
        assert_eq!(
            g.key.region,
            Some(ImagePixelRect::new(38144, 28672, 3584, 2816))
        );
        assert_eq!(g.visible, (19400.0, 14600.0));
    }

    #[test]
    fn panning_reuses_the_region_until_the_view_crosses_a_snap_line() {
        let at = |pan_x: f32| {
            geometry(
                Path::new("a.png"),
                ResampleFilter::MagicKernelSharp,
                (100, 100),
                (10000.0, 10000.0),
                (1000.0, 800.0),
                (pan_x, 0.0),
                1.0,
            )
            .unwrap()
        };
        // Visible x is 4500..5500 at rest: snapped to 4352..5632, then overscanned.
        let rest = at(0.0);
        assert_eq!(rest.key.region.unwrap().x, 3840);
        assert_eq!(rest.key.region.unwrap().width, 2304);
        assert_eq!(rest.key, at(100.0).key);
        assert_eq!(rest.key, at(-100.0).key);

        // Past a snap line the region moves by whole steps of the same grid,
        // and the old one still covers the new view.
        let moved = at(200.0);
        assert_eq!(moved.key.width, rest.key.width);
        assert_eq!(moved.key.region.unwrap().x, 3840 - SNAP);
        assert!(rest.key.region.unwrap().x <= 4300);
    }

    #[test]
    fn fitting_and_offscreen_images_have_bounded_regions() {
        let g = geometry(
            Path::new("a.png"),
            ResampleFilter::MagicKernelSharp,
            (100, 100),
            (200.0, 200.0),
            (1000.0, 800.0),
            (0.0, 0.0),
            1.5,
        )
        .unwrap();
        assert_eq!(g.key.region, Some(ImagePixelRect::new(0, 0, 300, 300)));
        assert!(geometry(
            Path::new("a.png"),
            ResampleFilter::MagicKernelSharp,
            (100, 100),
            (200.0, 200.0),
            (1000.0, 800.0),
            (2000.0, 0.0),
            1.0
        )
        .is_none());
    }
}
