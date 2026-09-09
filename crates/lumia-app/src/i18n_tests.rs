use super::*;

#[test]
fn translations_preserve_language_selection_and_scaling_labels() {
    for (key, english, chinese) in [
        (TextKey::Open, "Open", "打开"),
        (TextKey::Confirm, "Open", "确认"),
        (TextKey::ScalingAlgorithm, "Scaling Algorithm", "缩放算法"),
        (TextKey::FilterNearestNeighbor, "Nearest Neighbor", "最近邻"),
        (TextKey::FilterBilinear, "Bilinear", "双线性"),
        (TextKey::FilterLanczos, "Lanczos", "Lanczos"),
        (TextKey::English, "English", "English"),
        (TextKey::Chinese, "中文", "中文"),
    ] {
        assert_eq!(tr(Language::English, key), english);
        assert_eq!(tr(Language::Chinese, key), chinese);
    }
}

#[test]
fn empty_state_preserves_the_platform_shortcut_in_both_languages() {
    let (english, chinese) = if cfg!(target_os = "macos") {
        (
            "Drop an image here, or open one with Cmd+O",
            "将图片拖到这里，或按 Cmd+O 打开",
        )
    } else {
        (
            "Drop an image here, or open one with Ctrl+O",
            "将图片拖到这里，或按 Ctrl+O 打开",
        )
    };
    assert_eq!(tr(Language::English, TextKey::EmptyState), english);
    assert_eq!(tr(Language::Chinese, TextKey::EmptyState), chinese);
}
