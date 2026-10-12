//! Runs one display resample on a background thread.

use lumia_core::{resize_decoded_image, DecodedImage, ImageLoadError};

use crate::display_resample::{ResampleKey, ResampleSource};
use crate::load_state::PreparedImage;

pub(crate) fn run_resample_job(
    source: ResampleSource,
    key: &ResampleKey,
) -> Result<Option<PreparedImage>, ImageLoadError> {
    let source = match source {
        ResampleSource::Raster(raster) => {
            return if let Some(region) = key.region {
                raster.resize_magic_kernel_sharp_region((key.width, key.height), region)
            } else {
                raster.resize(key.filter, key.width, key.height)
            }
            .map(PreparedImage::from_decoded)
            .map(Some)
        }
        ResampleSource::Bitmap(source) => source,
    };
    let Some(pixels) = source.as_bytes(0) else {
        return Ok(None);
    };
    let size = source.size(0);
    let Ok(source_width) = u32::try_from(size.width.0) else {
        return Ok(None);
    };
    let Ok(source_height) = u32::try_from(size.height.0) else {
        return Ok(None);
    };
    if source_width == 0 || source_height == 0 {
        return Ok(None);
    }
    if let Some(region) = key.region {
        return lumia_core::resize_magic_kernel_sharp_region(
            pixels,
            (source_width, source_height),
            (key.width, key.height),
            region,
        )
        .map(PreparedImage::from_decoded)
        .map(Some);
    }
    let input = DecodedImage {
        pixels_bgra8: pixels.to_vec(),
        width: source_width,
        height: source_height,
    };
    resize_decoded_image(&input, key.filter, key.width, key.height)
        .map(|resized| Some(PreparedImage::from_decoded(resized)))
}
