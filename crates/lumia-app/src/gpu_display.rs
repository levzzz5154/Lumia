//! Owns the GPU resampler of one viewer window.

use std::sync::Arc;

use gpui_kit::Window;
use lumia_core::ImagePixelRect;

use crate::display_resample::{ResampleKey, ResampleSlot};
use crate::gpu_resample::GpuResampler;

/// The resampler for one viewer window, created on first use.
#[derive(Default)]
pub(crate) struct GpuDisplay {
    tried: bool,
    resampler: Option<GpuResampler>,
}

impl GpuDisplay {
    /// Acquire the window's device once; drop everything if it was lost so the
    /// next frame starts over on the recovered device.
    pub(crate) fn prepare(&mut self, window: &Window) -> Option<&mut GpuResampler> {
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        {
            if window.gpu_device_lost() == Some(true) {
                *self = Self::default();
                return None;
            }
            if !self.tried {
                self.tried = true;
                let disabled = std::env::var_os("LUMIA_GPU_RESAMPLE").is_some_and(|v| v == "0");
                self.resampler = (!disabled)
                    .then(|| window.gpu_context())
                    .flatten()
                    .and_then(|context| {
                        context
                            .downcast::<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>()
                            .ok()
                    })
                    .map(|context| GpuResampler::new(context.0, context.1));
                eprintln!(
                    "lumia: display resampling on {}",
                    if self.resampler.is_some() {
                        "GPU"
                    } else {
                        "CPU"
                    }
                );
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
        let _ = window;
        self.resampler.as_mut()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.resampler.is_some()
    }

    pub(crate) fn output_ready(&self, slot: ResampleSlot) -> bool {
        self.resampler
            .as_ref()
            .is_some_and(|resampler| resampler.has_output(slot))
    }

    pub(crate) fn output(
        &self,
        slot: ResampleSlot,
        key: &ResampleKey,
    ) -> Option<(Arc<wgpu::Texture>, ImagePixelRect)> {
        self.resampler.as_ref()?.output(slot, key)
    }
}
