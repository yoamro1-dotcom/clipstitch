import os
import json
import math
import shlex
import tempfile
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
import imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


# --------------------------
# FFmpeg utilities
# --------------------------
def get_ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def get_ffprobe_path() -> Optional[str]:
    ffmpeg_path = get_ffmpeg_path()
    d = os.path.dirname(ffmpeg_path)
    candidates = [os.path.join(d, "ffprobe"), os.path.join(d, "ffprobe.exe")]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Some builds put ffprobe in PATH; try bare name
    return "ffprobe"

def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def probe_video_dims(path: str) -> Tuple[int, int]:
    """
    Returns (width, height) of the first video stream using ffprobe, falling back to imageio.
    """
    ffprobe = get_ffprobe_path()
    try:
        # Try ffprobe JSON
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path
        ]
        proc = run_cmd(cmd)
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            if streams:
                w = int(streams[0]["width"])
                h = int(streams[0]["height"])
                return w, h
    except Exception:
        pass

    # Fallback to imageio
    try:
        rdr = imageio.get_reader(path)
        meta = rdr.get_meta_data()
        size = meta.get("size")
        if size:
            w, h = size
            rdr.close()
            return int(w), int(h)
        rdr.close()
    except Exception:
        pass

    # Worst-case fallback
    return 1920, 1080

def even(x: int) -> int:
    x = int(round(x))
    return x if x % 2 == 0 else x - 1

def hex_to_rgb_tuple(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def rgb_to_ffmpeg_hex(rgb: Tuple[int, int, int]) -> str:
    # FFmpeg color format e.g., 0xRRGGBB
    return "0x{0:02x}{1:02x}{2:02x}".format(*rgb)


# --------------------------
# Overlay text image (Pillow)
# --------------------------
def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_center_text_overlay_png(
    text: str,
    frame_w: int,
    frame_h: int,
    font_size: int,
    text_color: str,
    bg_color: str,
    bg_opacity: float,
    max_width_ratio: float = 0.8,
    padding_px: int = 24,
    line_spacing: float = 1.1,
) -> Image.Image:
    """
    Returns a small RGBA image (only the pill + text) to be overlaid at center.
    Its width is capped at max_width_ratio * frame_w.
    """
    max_w = max(50, int(frame_w * max_width_ratio))
    font = load_font(font_size)

    # Wrap text using a rough char-width estimate
    try:
        bbox_M = font.getbbox("M")
        avg_char_w = max(1, bbox_M[2] - bbox_M[0])
    except Exception:
        avg_char_w = max(1, int(font_size * 0.6))
    max_chars = max(1, int(max_w / avg_char_w))

    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=max_chars)
        lines.extend(wrapped if wrapped else [""])

    # Measure lines
    dummy = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy)
    line_sizes = [draw_dummy.textbbox((0, 0), ln, font=font) for ln in lines]
    line_widths = [(b[2] - b[0]) for b in line_sizes]
    line_heights = [(b[3] - b[1]) for b in line_sizes]
    text_w = max(line_widths) if line_widths else font_size
    text_h = sum(line_heights) if line_heights else font_size
    if len(line_heights) > 1:
        text_h += int((len(line_heights) - 1) * font_size * (line_spacing - 1.0))

    # Canvas for pill + text (tight bounding box)
    rect_w = text_w + padding_px * 2
    rect_h = text_h + padding_px * 2

    img = Image.new("RGBA", (rect_w, rect_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle
    bg_rgb = hex_to_rgb_tuple(bg_color)
    a = max(0, min(255, round(255 * float(bg_opacity))))
    radius = int(min(rect_w, rect_h) * 0.12)
    try:
        draw.rounded_rectangle([0, 0, rect_w, rect_h], radius=radius, fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], a))
    except Exception:
        draw.rectangle([0, 0, rect_w, rect_h], fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], a))

    # Text
    txt_rgb = hex_to_rgb_tuple(text_color)
    x = padding_px
    y = padding_px
    for i, line in enumerate(lines):
        draw.text((x, y), line, font=font, fill=(txt_rgb[0], txt_rgb[1], txt_rgb[2], 255))
        y += int(line_heights[i] * line_spacing)

    return img


# --------------------------
# Filter builders
# --------------------------
@dataclass
class ExportOptions:
    layout: str                # 'sbs' | 'stack' | 'seq'
    target_height: int         # used by 'sbs' & 'seq'
    target_width: int          # used by 'stack'
    show_divider: bool
    divider_color: str
    overlay_text: str
    font_size: int
    text_color: str
    text_bg_color: str
    text_bg_opacity: float
    audio_mode: str            # 'pre' | 'post' | 'concat' | 'none'
    fps: int

def build_filters_side_by_side(
    pre_w, pre_h, post_w, post_h, H, show_divider, divider_hex
):
    preW = even(pre_w * H / pre_h)
    postW = even(post_w * H / post_h)
    preW, postW, H = int(preW), int(postW), even(H)
    div_w = int(max(2, round(0.004 * (preW + postW)))) if show_divider else 0

    parts = []
    parts.append(f"[0:v]scale={preW}:{H}:flags=lanczos,setsar=1[v0]")
    parts.append(f"[1:v]scale={postW}:{H}:flags=lanczos,setsar=1[v1]")
    if show_divider:
        parts.append(f"color=c={divider_hex}:s={div_w}x{H}[div]")
        parts.append(f"[v0][div][v1]hstack=inputs=3[base]")
    else:
        parts.append(f"[v0][v1]hstack=inputs=2[base]")
    out_w = preW + postW + (div_w if show_divider else 0)
    out_h = H
    # overlay later: [base][2:v]overlay=...
    return ";".join(parts), out_w, out_h

def build_filters_stacked(
    pre_w, pre_h, post_w, post_h, W, show_divider, divider_hex
):
    W = even(W)
    preH = even(pre_h * W / pre_w)
    postH = even(post_h * W / post_w)
    preH, postH = int(preH), int(postH)
    div_h = int(max(2, round(0.004 * (preH + postH)))) if show_divider else 0

    parts = []
    parts.append(f"[0:v]scale={W}:{preH}:flags=lanczos,setsar=1[v0]")
    parts.append(f"[1:v]scale={W}:{postH}:flags=lanczos,setsar=1[v1]")
    if show_divider:
        parts.append(f"color=c={divider_hex}:s={W}x{div_h}[div]")
        parts.append(f"[v0][div][v1]vstack=inputs=3[base]")
    else:
        parts.append(f"[v0][v1]vstack=inputs=2[base]")
    out_w = W
    out_h = preH + postH + (div_h if show_divider else 0)
    return ";".join(parts), out_w, out_h

def build_filters_sequential(
    pre_w, pre_h, post_w, post_h, H, audio_mode
):
    H = even(H)
    preW = even(pre_w * H / pre_h)
    postW = even(post_w * H / post_h)
    finalW = int(max(preW, postW))

    parts = []
    parts.append(f"[0:v]scale={int(preW)}:{H}:flags=lanczos,setsar=1,pad={finalW}:{H}:(ow-iw)/2:(oh-ih)/2[v0]")
    parts.append(f"[1:v]scale={int(postW)}:{H}:flags=lanczos,setsar=1,pad={finalW}:{H}:(ow-iw)/2:(oh-ih)/2[v1]")

    if audio_mode == "concat":
        # concat video and audio
        parts.append(f"[v0][0:a?][v1][1:a?]concat=n=2:v=1:a=1[basev][basea]")
        out_v = "[basev]"
        out_a = "[basea]"
    else:
        # concat video only (a=0)
        parts.append(f"[v0][v1]concat=n=2:v=1:a=0[base]")
        out_v = "[base]"
        out_a = None

    return ";".join(parts), finalW, H, out_v, out_a


# --------------------------
# Build and run FFmpeg
# --------------------------
def assemble_and_run_ffmpeg(
    pre_path: str,
    post_path: str,
    opts: ExportOptions,
    tmpdir: str
) -> str:
    ffmpeg = get_ffmpeg_path()

    # Probe input sizes
    pre_w, pre_h = probe_video_dims(pre_path)
    post_w, post_h = probe_video_dims(post_path)

    # Build base filter by layout
    divider_hex = rgb_to_ffmpeg_hex(hex_to_rgb_tuple(opts.divider_color))

    if opts.layout == "sbs":
        base_filters, out_w, out_h = build_filters_side_by_side(
            pre_w, pre_h, post_w, post_h, opts.target_height, opts.show_divider, divider_hex
        )
        out_v_label = "[base]"
        out_a_label = None  # audio mapped later

    elif opts.layout == "stack":
        base_filters, out_w, out_h = build_filters_stacked(
            pre_w, pre_h, post_w, post_h, opts.target_width, opts.show_divider, divider_hex
        )
        out_v_label = "[base]"
        out_a_label = None

    elif opts.layout == "seq":
        base_filters, out_w, out_h, out_v_label, out_a_label = build_filters_sequential(
            pre_w, pre_h, post_w, post_h, opts.target_height, opts.audio_mode
       
