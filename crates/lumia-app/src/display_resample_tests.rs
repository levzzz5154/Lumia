use super::*;
use lumia_core::DecodedImage;

fn key(path: &str, size: u32) -> ResampleKey {
    ResampleKey {
        path: Path::new(path).to_path_buf(),
        filter: ResampleFilter::Lanczos,
        width: size,
        height: size,
        region: None,
        source_dimensions: None,
    }
}

fn finish(preview: &mut ResamplePreview, key: ResampleKey) {
    preview.key = Some(key.clone());
    preview.pending = true;
    // Key size is irrelevant here; a 1x1 bitmap keeps the fixture small.
    let image = PreparedImage::from_decoded(DecodedImage {
        pixels_bgra8: vec![0, 0, 0, 255],
        width: 1,
        height: 1,
    });
    assert!(preview.complete(preview.generation, key, Ok(Some(image))));
}

#[test]
fn latest_for_keeps_the_previous_scale_of_the_same_document() {
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));

    // Still the same file, only the scale changed: the stale bitmap is a
    // valid stopgap while the new one is being produced.
    preview.key = Some(key("a.png", 200));
    preview.pending = true;
    assert!(preview.ready(&key("a.png", 200)).is_none());
    assert!(preview.latest_for(Path::new("a.png")).is_some());
}

#[test]
fn invalidate_drops_the_bitmap_and_makes_a_late_completion_a_no_op() {
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));
    assert!(preview.latest_for(Path::new("a.png")).is_some());

    let generation_before = preview.generation;
    preview.invalidate();

    assert!(preview.latest_for(Path::new("a.png")).is_none());
    assert!(preview.ready(&key("a.png", 100)).is_none());

    // The navigation case: the pane already reports the new path, so a job
    // scheduled against it before the decode landed must not repopulate
    // the cache with the outgoing image's pixels.
    let stale = key("b.png", 100);
    let image = PreparedImage::from_decoded(DecodedImage {
        pixels_bgra8: vec![1, 2, 3, 255],
        width: 1,
        height: 1,
    });
    assert!(!preview.complete(generation_before, stale.clone(), Ok(Some(image))));
    assert!(preview.latest_for(Path::new("b.png")).is_none());
    assert!(preview.image.is_none());
}

#[test]
fn latest_for_withholds_the_outgoing_documents_bitmap() {
    // After navigating, the previous file's bitmap must not be stretched
    // into the incoming image's frame.
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));
    assert!(preview.latest_for(Path::new("a.png")).is_some());
    assert!(preview.latest_for(Path::new("b.png")).is_none());
}

#[test]
fn latest_for_is_empty_before_any_resample_finishes() {
    let preview = ResamplePreview::default();
    assert!(preview.latest_for(Path::new("a.png")).is_none());
}

#[test]
fn failed_resize_does_not_label_previous_scale_as_ready() {
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));
    let requested = key("a.png", 200);
    preview.key = Some(requested.clone());
    preview.pending = true;
    assert!(preview.complete(
        preview.generation,
        requested.clone(),
        Err(ImageLoadError::MemoryLimit {
            bytes: 100,
            limit: 50
        })
    ));
    assert!(preview.ready(&requested).is_none());
    assert!(preview.error.is_some());
}

#[test]
fn mks_job_resamples_visible_region_above_old_full_image_limit() {
    let input = PreparedImage::from_decoded(DecodedImage {
        pixels_bgra8: [40, 80, 120, 255].repeat(4),
        width: 2,
        height: 2,
    });
    let key = ResampleKey {
        path: "a.png".into(),
        filter: ResampleFilter::MagicKernelSharp,
        width: 100_000,
        height: 100_000,
        region: Some(lumia_core::ImagePixelRect::new(50_000, 50_000, 64, 48)),
        source_dimensions: Some((2, 2)),
    };
    let output = run_resample_job(ResampleSource::Bitmap(input.render_image()), &key)
        .unwrap()
        .unwrap();
    assert_eq!(output.dimensions(), (64, 48));
    assert!(output
        .render_image()
        .as_bytes(0)
        .unwrap()
        .chunks_exact(4)
        .all(|p| p == [40, 80, 120, 255]));
}

#[test]
fn latest_region_survives_pan_and_zoom_but_not_a_document_change() {
    let region_key = |x: u32, width: u32| ResampleKey {
        filter: ResampleFilter::MagicKernelSharp,
        region: Some(lumia_core::ImagePixelRect::new(x, 0, 10, 10)),
        source_dimensions: Some((50, 50)),
        ..key("a.png", width)
    };
    let mut preview = ResamplePreview::default();
    finish(&mut preview, region_key(5, 100));

    // Panned: the exact key is not ready, but the old region still applies.
    assert!(preview.ready(&region_key(9, 100)).is_none());
    let (_, region, grid) = preview.latest_region(&region_key(9, 100)).unwrap();
    assert_eq!((region.x, grid), (5, (100, 100)));

    // Zoomed: reported with its own grid so the caller can rescale it.
    let (_, _, grid) = preview.latest_region(&region_key(5, 200)).unwrap();
    assert_eq!(grid, (100, 100));

    assert!(preview
        .latest_region(&ResampleKey {
            path: Path::new("b.png").to_path_buf(),
            ..region_key(5, 100)
        })
        .is_none());
}
