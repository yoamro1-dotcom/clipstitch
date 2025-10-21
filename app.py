import os
import io
import math
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    clips_array,
    concatenate_videoclips,
    ImageClip,
    AudioFileClip
)


# --------------------
# Helpers
# --------------------
def _safe_int_even(x: int) -> int:
    """Make sure dimension is an even integer for H.264 compatibility."""
    x = int(max(2, round(x)))
    return x if x % 2 == 0 else x - 1

def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Attempt to load a decent font; fall back to default if not available."""
    # Common fonts that are often present on Linux/Streamlit Cloud
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    # Fallback
    return ImageFont.load_default()

def _render_center_text_image(
    text: str,
    frame_size: Tuple[int, int],
    font_size: int = 48,
    text_color: str = "#FFFFFF",
    bg_color: str = "#000000",
    bg_opacity: float = 0.45,
    max_width_ratio: float = 0.8,
    padding_px: int = 24,
    line_spacing: float = 1.1,
) -> Image.Image:
    """Create an RGBA image with centered text and semi-transparent rounded rect."""
    w, h = frame_size
    W = int(w * max_width_ratio)

    font = _load_font(font_size)

    # Wrap text to fit width W
    # Approximate wrap width by measuring average char width
    avg_char_w = font.getbbox("M")[2]  # width of "M"
    if avg_char_w <= 0:
        avg_char_w = font_size * 0.6
    max_chars = max(1, int(W / avg_char_w))
    wrapped = []
    for para in text.split("\n"):
        wrapped.extend(textwrap.wrap(para, width=max_chars) or [""])

    # Measure text block
    draw_dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_heights = []
    line_widths = []
    for line in wrapped:
        bbox = draw_dummy.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    if not line_heights:
        line_heights = [font_size]
        line_widths = [font_size]

    text_block_w = max(line_widths)
    # Add spacing between lines
    text_block_h = sum(line_heights) + int((len(line_heights) - 1) * font_size * (line_spacing - 1.0))

    # Create transparent canvas
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle
    rect_w = text_block_w + padding_px * 2
    rect_h = text_block_h + padding_px * 2
    rect_x0 = (w - rect_w) // 2
    rect_y0 = (h - rect_h) // 2
    rect_x1 = rect_x0 + rect_w
    rect_y1 = rect_y0 + rect_h
    radius = int(min(rect_w, rect_h) * 0.12)

    # Convert bg_color hex to RGBA with opacity
    bg_color = bg_color.lstrip("#")
    r = int(bg_color[0:2], 16)
    g = int(bg_color[2:4], 16)
    b = int(bg_color[4:6], 16)
    a = int(max(0, min(255, round(255 * bg_opacity))))

    # Rounded rectangle
    try:
        draw.rounded_rectangle([rect_x0, rect_y0, rect_x1, rect_y1], radius=radius, fill=(r, g, b, a))
    except Exception:
        # Fallback to normal rectangle if rounded not supported
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(r, g, b, a))

    # Draw wrapped text
    text_color = text_color.lstrip("#")
    tr = int(text_color[0:2], 16)
    tg = int(text_color[2:4], 16)
    tb = int(text_color[4:6], 16)
    tx = rect_x0 + padding_px
    ty = rect_y0 + padding_px

    for i, line in enumerate(wrapped):
        draw.text((tx, ty), line, font=font, fill=(tr, tg, tb, 255))
        # Move down for next line
        line_h = line_heights[i]
        ty += int(line_h * line_spacing)

    return img

@dataclass
class ExportOptions:
    layout: str               # 'side_by_side' | 'stacked' | 'sequential'
    target_height: int        # final output height (auto width), or vice versa
    max_width: Optional[int]  # optional cap on width
    bg_color: str
    show_divider: bool
    divider_color: str
    overlay_text: str
    font_size: int
    text_color: str
    text_bg_color: str
    text_bg_opacity: float
    audio_source: str         # 'pre' | 'post' | 'none'
    fps: Optional[int]


def make_composite_video(
    pre_path: str,
    post_path: str,
    opts: ExportOptions,
    tmp_dir: str
) -> str:
    pre = VideoFileClip(pre_path)
    post = VideoFileClip(post_path)

    # Normalize fps if requested (helps with odd encodes)
    if opts.fps:
        pre = pre.set_fps(opts.fps)
        post = post.set_fps(opts.fps)

    # Decide layout
    if opts.layout == "side_by_side":
        # Match heights, stack horizontally
        target_h = opts.target_height
        pre_resized = pre.resize(height=target_h)
        post_resized = post.resize(height=target_h)

        # Ensure even widths
        pre_w = _safe_int_even(pre_resized.w)
        post_w = _safe_int_even(post_resized.w)
        h = _safe_int_even(target_h)
        pre_resized = pre_resized.resize(newsize=(pre_w, h))
        post_resized = post_resized.resize(newsize=(post_w, h))

        comp = clips_array([[pre_resized, post_resized]])
        # Optional divider
        if opts.show_divider:
            divider_w = max(2, int(0.004 * comp.w))  # ~0.4% of width
            divider = ImageClip(np.full((h, divider_w, 3), 0, dtype=np.uint8), duration=comp.duration)
            # Colorize divider
            dc = tuple(int(opts.divider_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            divider = divider.on_color(size=(divider_w, h), color=dc, pos=('center', 'center'))
            # Overlay divider at center x
            comp = CompositeVideoClip([
                comp,
                divider.set_position((comp.w // 2 - divider_w // 2, 0))
            ], size=(comp.w, comp.h), bg_color=None)

    elif opts.layout == "stacked":
        # Match widths, stack vertically
        target_h = opts.target_height
        # First, set a working width from pre clip to maintain aspect
        working_w = pre.w
        pre_resized = pre.resize(width=working_w)
        post_resized = post.resize(width=working_w)
        comp = clips_array([[pre_resized], [post_resized]])

        # Then scale final composite to target height if provided
        scale_ratio = target_h / comp.h
        new_w = _safe_int_even(comp.w * scale_ratio)
        new_h = _safe_int_even(target_h)
        comp = comp.resize(newsize=(new_w, new_h))

        if opts.show_divider:
            divider_h = max(2, int(0.004 * comp.h))
            dc = tuple(int(opts.divider_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            divider = ImageClip(np.full((divider_h, comp.w, 3), 0, dtype=np.uint8), duration=comp.duration)
            divider = divider.on_color(size=(comp.w, divider_h), color=dc, pos=('center', 'center'))
            comp = CompositeVideoClip([
                comp,
                divider.set_position((0, comp.h // 2 - divider_h // 2))
            ], size=(comp.w, comp.h), bg_color=None)

    elif opts.layout == "sequential":
        # Play pre then post
        # Resize both to same height
        target_h = opts.target_height
        pre_resized = pre.resize(height=target_h)
        post_resized = post.resize(height=target_h)
        # Ensure even dims
        pre_resized = pre_resized.resize(newsize=(_safe_int_even(pre_resized.w), _safe_int_even(pre_resized.h)))
        post_resized = post_resized.resize(newsize=(_safe_int_even(post_resized.w), _safe_int_even(post_resized.h)))
        comp = concatenate_videoclips([pre_resized, post_resized], method="compose")

    else:
        raise ValueError("Unknown layout")

    # Cap width if requested (for performance)
    if opts.max_width and comp.w > opts.max_width:
        scale = opts.max_width / comp.w
        comp = comp.resize(scale)

    # Ensure even final dims
    final_w = _safe_int_even(comp.w)
    final_h = _safe_int_even(comp.h)
    if (final_w != comp.w) or (final_h != comp.h):
        comp = comp.resize(newsize=(final_w, final_h))

    # Overlay center text (full duration)
    if opts.overlay_text.strip():
        img = _render_center_text_image(
            text=opts.overlay_text,
            frame_size=(comp.w, comp.h),
            font_size=opts.font_size,
            text_color=opts.text_color,
            bg_color=opts.text_bg_color,
            bg_opacity=opts.text_bg_opacity
        )
        text_clip = ImageClip(np.array(img)).set_duration(comp.duration).set_position(("center", "center"))
        comp = CompositeVideoClip([comp, text_clip], size=(comp.w, comp.h))

    # Audio handling
    if opts.audio_source == "none":
        comp = comp.without_audio()
    elif opts.audio_source == "pre":
        comp = comp.set_audio(pre.audio if pre.audio else None)
    elif opts.audio_source == "post":
        comp = comp.set_audio(post.audio if post.audio else None)

    # Export
    out_path = os.path.join(tmp_dir, "export.mp4")
    comp.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac" if opts.audio_source != "none" else None,
        fps=opts.fps or comp.fps or 30,
        threads=os.cpu_count() or 4,
        temp_audiofile=os.path.join(tmp_dir, "temp-audio.m4a"),
        remove_temp=True,
        preset="medium"
    )
    # Close clips to free resources
    comp.close(); pre.close(); post.close()
    return out_path


# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Pre/Post Echo Composer", page_icon="🎬", layout="centered")

st.title("🎬 Pre/Post Echo Video Composer")
st.caption("Upload your Pre Echo and Post Echo videos, choose a layout, add centered text, and export a shareable MP4.")

with st.expander("How it works"):
    st.write(
        "- Upload a **Pre** video and a **Post** video.\n"
        "- Choose a layout: **Side‑by‑Side**, **Stacked**, or **Sequential (Before→After)**.\n"
        "- Enter overlay text to appear **centered** with a semi‑transparent background.\n"
        "- Optionally select which audio to keep.\n"
        "- Export to MP4 and download."
    )

col1, col2 = st.columns(2)
with col1:
    pre_file = st.file_uploader("Upload **Pre Echo** video", type=["mp4", "mov", "m4v", "avi", "webm"])
with col2:
    post_file = st.file_uploader("Upload **Post Echo** video", type=["mp4", "mov", "m4v", "avi", "webm"])

layout = st.radio(
    "Layout",
    options=["Side‑by‑Side", "Stacked", "Sequential (Before→After)"],
    horizontal=True
)
map_layout = {
    "Side‑by‑Side": "side_by_side",
    "Stacked": "stacked",
    "Sequential (Before→After)": "sequential"
}

overlay_text = st.text_area("Center overlay text", value="PRE vs POST", placeholder="Type the text shown in the middle…")

c1, c2 = st.columns(2)
with c1:
    font_size = st.slider("Font size", min_value=24, max_value=96, value=48, step=2)
    text_color = st.color_picker("Text color", value="#FFFFFF")
    fps = st.number_input("Output FPS (optional)", min_value=1, max_value=120, value=30, step=1)
with c2:
    text_bg_color = st.color_picker("Text background color", value="#000000")
    text_bg_opacity = st.slider("Background opacity", 0.0, 1.0, 0.45, 0.05)

c3, c4 = st.columns(2)
with c3:
    show_divider = st.checkbox("Show divider (for side‑by‑side/stacked)", value=True)
    divider_color = st.color_picker("Divider color", value="#FFFFFF")
with c4:
    audio_source = st.selectbox("Audio", options=["Keep Pre audio", "Keep Post audio", "Mute"], index=2)

target_height = st.select_slider("Target height (px)", options=[360, 480, 720, 1080, 1440], value=720)
max_width = st.select_slider("Max output width (cap)", options=[None, 640, 960, 1280, 1920, 2560], value=1280, format_func=lambda x: "No cap" if x is None else f"{x}px")

export_btn = st.button("🔄 Generate & Export MP4", type="primary", use_container_width=True)

if export_btn:
    if not pre_file or not post_file:
        st.error("Please upload both **Pre** and **Post** videos.")
        st.stop()

    # Persist uploads to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path = os.path.join(tmpdir, f"pre_{pre_file.name}")
        post_path = os.path.join(tmpdir, f"post_{post_file.name}")
        with open(pre_path, "wb") as f:
            f.write(pre_file.read())
        with open(post_path, "wb") as f:
            f.write(post_file.read())

        opts = ExportOptions(
            layout=map_layout[layout],
            target_height=int(target_height),
            max_width=None if max_width is None else int(max_width),
            bg_color="#000000",
            show_divider=bool(show_divider),
            divider_color=divider_color,
            overlay_text=overlay_text or "",
            font_size=int(font_size),
            text_color=text_color,
            text_bg_color=text_bg_color,
            text_bg_opacity=float(text_bg_opacity),
            audio_source={"Keep Pre audio": "pre", "Keep Post audio": "post", "Mute": "none"}[audio_source],
            fps=int(fps) if fps else None
        )

        st.info("Processing video… this can take a moment for large files.")
        try:
            out_path = make_composite_video(pre_path, post_path, opts, tmpdir)
        except Exception as e:
            st.exception(e)
            st.stop()

        # Load and offer download
        with open(out_path, "rb") as f:
            data = f.read()

        st.success("Export complete!")
        st.video(data)

        st.download_button(
            label="⬇️ Download MP4",
            data=data,
            file_name="pre-post-export.mp4",
            mime="video/mp4",
            use_container_width=True
        )

st.markdown("---")
st.caption("Tip: For faster exports, choose 720p height and cap width to 1280px. You can always re-export at 1080p if needed.")
``
