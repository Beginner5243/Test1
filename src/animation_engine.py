from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _speech_energy_per_frame(wav_path: Path, total_frames: int) -> list[float]:
    if not wav_path.exists():
        return [0.0] * total_frames
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            n_ch = max(1, wf.getnchannels())
            sw = wf.getsampwidth()
            raw = wf.readframes(n_frames)
        if n_frames == 0 or sw != 2:
            return [0.0] * total_frames
        samples = [
            int.from_bytes(raw[i:i + 2], byteorder="little", signed=True)
            for i in range(0, len(raw), 2 * n_ch)
        ]
        peak = max((abs(s) for s in samples), default=1) or 1
        spf = max(1, len(samples) // max(1, total_frames))
        out: list[float] = []
        for idx in range(total_frames):
            chunk = samples[idx * spf:(idx + 1) * spf]
            avg = (sum(abs(x) for x in chunk) / max(1, len(chunk))) / peak
            out.append(min(1.0, avg * 2.5))
        return out
    except Exception:
        return [0.0] * total_frames


def render_scene_animation_clip(
    scene: dict[str, Any],
    characters: list[dict[str, Any]],
    backdrop_path: Path,
    voice_wav_path: Path,
    work_dir: Path,
    clip_out_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> Path:
    """
    Procedural motion engine with camera tracking, character movement,
    and mouth/expression animation synced to voice energy.
    """
    frames_dir = work_dir / f"frames_s{scene['scene_number']:02d}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip_out_path.parent.mkdir(parents=True, exist_ok=True)

    dur = float(scene["duration_sec"])
    total_frames = max(15, int(round(dur * fps)))
    energies = _speech_energy_per_frame(voice_wav_path, total_frames)
    bg_base = Image.open(backdrop_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    active_chars = [c for c in characters if c["id"] in scene.get("active_character_ids", [])] or characters[:1]
    sprites = [
        Image.open(c["sprite_path"]).convert("RGBA").resize((340, 340), Image.Resampling.LANCZOS)
        for c in active_chars
    ]

    movement = scene.get("movement", "stride_advance")
    speaker_id = scene.get("speaker_id", active_chars[0]["id"])

    for f_idx in range(total_frames):
        prog = f_idx / max(1, total_frames - 1)
        frame = bg_base.copy()

        # Camera tracking pan + subtle zoom
        zoom = 1.0 + 0.04 * prog
        zw, zh = int(width * zoom), int(height * zoom)
        zoomed = frame.resize((zw, zh), Image.Resampling.BILINEAR)
        pan_x = int((zw - width) * (0.35 + 0.3 * math.sin(prog * math.pi)))
        top_y = (zh - height) // 2
        frame = zoomed.crop((pan_x, top_y, pan_x + width, top_y + height))

        canvas = frame.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        amp = energies[f_idx] if f_idx < len(energies) else 0.0

        n_act = len(sprites)
        for idx, (char_info, sp) in enumerate(zip(active_chars, sprites)):
            slot_x = 120 + idx * 380 if n_act > 1 else 370
            dx = 0
            dy = 0
            if movement == "stride_advance":
                dx = int(60 * prog)
                dy = -int(abs(math.sin(prog * math.pi * 4)) * 24)
            elif movement == "jump_arc":
                dy = -int(math.sin(prog * math.pi) * 110)
            elif movement == "dash_dodge":
                dx = int(90 * math.sin(prog * math.pi * 2)) * (1 if idx == 0 else -1)
            elif movement == "celebrate_hop":
                dy = -int(abs(math.sin(prog * math.pi * 6)) * 45)

            px = max(50, min(width - sp.width - 50, slot_x + dx))
            py = max(400, min(height - sp.height - 260, 1140 + dy))
            tilt = 8 * math.sin(prog * math.pi * 4) if char_info["id"] == speaker_id else 0
            rotated = sp.rotate(tilt, expand=True, resample=Image.BICUBIC)
            canvas.alpha_composite(rotated, (px, py))

            # Mouth movement cue for active speaker
            if char_info["id"] == speaker_id and amp > 0.12:
                mx = px + rotated.width // 2
                my = py + int(rotated.height * 0.46)
                m_h = int(6 + amp * 16)
                draw.ellipse(
                    [mx - 14, my - m_h // 2, mx + 14, my + m_h // 2],
                    fill="#C62828",
                    outline="#111111",
                    width=2,
                )

        canvas.convert("RGB").save(frames_dir / f"f_{f_idx:04d}.png", format="PNG")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "f_%04d.png"),
            "-i", str(voice_wav_path),
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "128k",
            str(clip_out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return clip_out_path
