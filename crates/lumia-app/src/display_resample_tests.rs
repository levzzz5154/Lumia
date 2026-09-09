use super::*;

fn key(path: &str, size: u32) -> ResampleKey {
    ResampleKey {
        path: Path::new(path).to_path_buf(),
        rotation_quarter_turns: 0,
        filter: ResampleFilter::Lanczos,
        width: size,
        height: size,
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
    assert!(preview.latest_for(Path::new("a.png"), 0).is_some());
}

#[test]
fn invalidate_drops_the_bitmap_and_makes_a_late_completion_a_no_op() {
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));
    assert!(preview.latest_for(Path::new("a.png"), 0).is_some());

    let generation_before = preview.generation;
    preview.invalidate();

    assert!(preview.latest_for(Path::new("a.png"), 0).is_none());
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
    assert!(preview.latest_for(Path::new("b.png"), 0).is_none());
    assert!(preview.image.is_none());
}

#[test]
fn latest_for_withholds_the_outgoing_documents_bitmap() {
    // After navigating, the previous file's bitmap must not be stretched
    // into the incoming image's frame.
    let mut preview = ResamplePreview::default();
    finish(&mut preview, key("a.png", 100));
    assert!(preview.latest_for(Path::new("a.png"), 0).is_some());
    assert!(preview.latest_for(Path::new("b.png"), 0).is_none());
}

#[test]
fn latest_for_is_empty_before_any_resample_finishes() {
    let preview = ResamplePreview::default();
    assert!(preview.latest_for(Path::new("a.png"), 0).is_none());
}

fn check_rotation_resamples(width: u32, height: u32, rotations: &[u8]) {
    let source = DecodedImage {
        pixels_bgra8: (1..=width * height)
            .flat_map(|value| [value as u8, 0, 0, 255])
            .collect(),
        width,
        height,
    };
    let path = Path::new("rotation.png");
    let display_size = ((width * 2) as f32, (height * 2) as f32);
    let mut preview = ResamplePreview::default();
    let mut previous_rotation = 0;

    for &rotation in rotations {
        let rotated =
            lumia_core::rotate_bgra8(&source.pixels_bgra8, width, height, rotation).unwrap();
        let expected_corner = rotated.pixels_bgra8[..4].to_vec();
        let prepared = PreparedImage::from_decoded(rotated);
        let request = plan_display_resample(
            Some(&prepared),
            Some(path),
            rotation,
            ResampleFilter::NearestNeighbor,
            Some(display_size),
        )
        .unwrap();

        // A half turn (or any square-image turn) keeps the path, filter and
        // dimensions unchanged. The new orientation must still get a job.
        if let Some(previous) = preview.key.as_ref() {
            assert_eq!(
                (previous.width, previous.height),
                (request.key.width, request.key.height)
            );
            assert_ne!(previous, &request.key);
            assert!(preview.ready(&request.key).is_none());
            assert!(preview.latest_for(path, rotation).is_none());
        }

        let stale_key = preview.key.clone();
        let stale_generation = preview.generation;
        preview.generation += 1;
        preview.key = Some(request.key.clone());
        preview.pending = true;
        assert!(preview.latest_for(path, rotation).is_none());
        if let Some(stale_key) = stale_key {
            // A superseded rotation job cannot replace the current request.
            assert!(!preview.complete(stale_generation, stale_key, Ok(Some(prepared.clone()))));
        }
        let result = run_resample_job(request.source, &request.key);
        assert!(preview.complete(preview.generation, request.key.clone(), result));
        let image = preview.ready(&request.key).unwrap();
        assert_eq!(image.dimensions(), (width * 2, height * 2));
        assert_eq!(&image.pixels_bgra8().unwrap()[..4], expected_corner.as_slice());
        assert!(preview.latest_for(path, rotation).is_some());
        if previous_rotation != rotation {
            assert!(preview.latest_for(path, previous_rotation).is_none());
        }
        previous_rotation = rotation;
    }
}

#[test]
fn rectangular_half_turn_resamples_and_can_return_to_zero() {
    check_rotation_resamples(3, 2, &[0, 2, 0]);
}

#[test]
fn square_quarter_turns_resample_and_can_return_to_zero() {
    check_rotation_resamples(2, 2, &[0, 1, 2, 3, 0]);
}

#[test]
fn unsuccessful_rotation_does_not_relabel_the_previous_bitmap() {
    for result in [Ok(None), Err(ImageLoadError::Cancelled)] {
        let mut preview = ResamplePreview::default();
        finish(&mut preview, key("a.png", 100));
        let rotated_key = ResampleKey {
            rotation_quarter_turns: 2,
            ..key("a.png", 100)
        };
        preview.key = Some(rotated_key.clone());
        preview.pending = true;
        let should_notify = result.is_ok();
        assert_eq!(
            preview.complete(preview.generation, rotated_key.clone(), result),
            should_notify
        );

        assert!(preview.ready(&rotated_key).is_none());
        assert!(preview.latest_for(Path::new("a.png"), 2).is_none());
        assert!(preview.latest_for(Path::new("a.png"), 0).is_some());
    }
}
