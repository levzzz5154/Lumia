// Separable display resampling. Each pass filters one axis of `input` and is
// drawn as a full-target triangle; taps are fetched with textureLoad, so no
// sampler filtering is involved and the weights below are the whole filter.

struct Params {
    // Length of the filtered axis in the source image and in the full
    // destination grid. Equal lengths mean the axis is copied unfiltered.
    source_len: f32,
    target_len: f32,
    // Destination coordinate of this target's first texel on the filtered axis.
    target_offset: f32,
    // Source coordinate of `input`'s first texel on the filtered axis.
    tap_origin: f32,
    // Source coordinate of `input`'s texel that this target's first texel
    // reads on the other axis.
    cross_offset: f32,
    // 0 nearest, 1 bilinear, 2 Lanczos3, 3 Magic Kernel Sharp 2021,
    // 4 pixel view: nearest with edges blended by coverage (enlarging only).
    kernel: u32,
    // 0 filters x, 1 filters y.
    axis: u32,
    // 1 when `input` holds straight-alpha colour that must be premultiplied.
    premultiply: u32,
    // Second pass only: the lengths and offset of the other axis, and whether
    // to draw the pixel grid.
    grid_source_len: f32,
    grid_target_len: f32,
    grid_target_offset: f32,
    grid: u32,
}

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

const PI: f32 = 3.14159265358979;

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-6) {
        return 1.0;
    }
    let t = PI * x;
    return sin(t) / t;
}

fn magic_kernel_sharp(x: f32) -> f32 {
    if (x < 0.5) {
        return 577.0 / 576.0 - 239.0 / 144.0 * x * x;
    } else if (x < 1.5) {
        return 35.0 / 36.0 * (x - 1.0) * (x - 239.0 / 140.0);
    } else if (x < 2.5) {
        return (x - 2.0) * (65.0 / 24.0 - x) / 6.0;
    } else if (x < 3.5) {
        return (x - 3.0) * (x - 3.75) / 36.0;
    } else if (x < 4.5) {
        return -(x - 4.5) * (x - 4.5) / 288.0;
    }
    return 0.0;
}

fn radius() -> f32 {
    switch params.kernel {
        case 1u: { return 1.0; }
        case 2u: { return 3.0; }
        default: { return 4.5; }
    }
}

fn weight(distance: f32) -> f32 {
    let x = abs(distance);
    switch params.kernel {
        case 1u: { return max(0.0, 1.0 - x); }
        case 2u: {
            if (x >= 3.0) {
                return 0.0;
            }
            return sinc(x) * sinc(x / 3.0);
        }
        default: { return magic_kernel_sharp(x); }
    }
}

fn fetch(tap: i32, cross_index: i32) -> vec4<f32> {
    let along = tap - i32(params.tap_origin);
    var texel: vec4<f32>;
    if (params.axis == 0u) {
        texel = textureLoad(input, vec2<i32>(along, cross_index), 0);
    } else {
        texel = textureLoad(input, vec2<i32>(cross_index, along), 0);
    }
    if (params.premultiply != 0u) {
        return vec4<f32>(texel.rgb * texel.a, texel.a);
    }
    return texel;
}

// Premultiplied, normalized result for the target texel at `position`.
fn filtered(position: vec4<f32>) -> vec4<f32> {
    let texel = vec2<i32>(position.xy);
    var target_index: f32;
    var cross_index: i32;
    if (params.axis == 0u) {
        target_index = f32(texel.x) + params.target_offset;
        cross_index = texel.y + i32(params.cross_offset);
    } else {
        target_index = f32(texel.y) + params.target_offset;
        cross_index = texel.x + i32(params.cross_offset);
    }
    let last_source = i32(params.source_len) - 1;
    if (params.source_len == params.target_len) {
        return fetch(clamp(i32(target_index), 0, last_source), cross_index);
    }
    let scale = params.source_len / params.target_len;
    if (params.kernel == 0u) {
        let nearest = i32(floor((target_index + 0.5) * scale));
        return fetch(clamp(nearest, 0, last_source), cross_index);
    }
    if (params.kernel == 4u) {
        // The target texel spans `center ± half` in source texels and touches
        // at most two of them; weight each by the share it covers.
        let half = 0.5 * scale;
        let middle = (target_index + 0.5) * scale;
        let low = clamp(i32(floor(middle - half)), 0, last_source);
        let high = clamp(i32(floor(middle + half)), 0, last_source);
        if (low == high) {
            return fetch(low, cross_index);
        }
        let share = clamp((middle + half - f32(high)) / scale, 0.0, 1.0);
        return mix(fetch(low, cross_index), fetch(high, cross_index), share);
    }
    // Widen the whole kernel when reducing, to suppress aliasing.
    let stretch = max(scale, 1.0);
    let center = (target_index + 0.5) * scale - 0.5;
    let reach = radius() * stretch;
    let first = max(i32(ceil(center - reach)), 0);
    let last = min(i32(floor(center + reach)), last_source);
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var tap = first; tap <= last; tap++) {
        let w = weight((f32(tap) - center) / stretch);
        sum += fetch(tap, cross_index) * w;
        total += w;
    }
    if (total == 0.0) {
        return fetch(clamp(i32(round(center)), 0, last_source), cross_index);
    }
    return sum / total;
}

@fragment
fn fs_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return filtered(position);
}

// The second pass also returns to straight alpha, which is what the display
// surface is composited with.
@fragment
fn fs_second(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let color = filtered(position);
    let alpha = clamp(color.a, 0.0, 1.0);
    if (alpha <= 0.0) {
        return vec4<f32>(0.0);
    }
    var rgb = clamp(color.rgb / color.a, vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.grid != 0u) {
        let texel = floor(position.xy);
        var along = texel.x;
        var across = texel.y;
        if (params.axis == 1u) {
            along = texel.y;
            across = texel.x;
        }
        let on_line = ends_source_texel(
            along + params.target_offset,
            params.source_len / params.target_len,
        ) || ends_source_texel(
            across + params.grid_target_offset,
            params.grid_source_len / params.grid_target_len,
        );
        if (on_line) {
            // Pull towards the opposite of the pixel's own brightness so the
            // line reads on both light and dark content.
            let luminance = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
            rgb = mix(rgb, vec3<f32>(step(luminance, 0.5)), 0.28);
        }
    }
    return vec4<f32>(rgb, alpha);
}

// Whether `target_index` is the last target texel of a source texel. Adjacent
// texels evaluate the same product for their shared edge, so every boundary
// lands on exactly one of them. The nudge absorbs the GPU's inexact division
// in `scale`; the grid is only drawn while texels are far wider than it.
fn ends_source_texel(target_index: f32, scale: f32) -> bool {
    return floor((target_index + 1.0) * scale + 0.01) != floor(target_index * scale + 0.01);
}
