from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

from src.animation_engine import render_scene_animation_clip
from src.audio_mixer import mix_episode_audio
from src.character_manager import build_character_dna
from src.quality_checker import validate_reel_output
from src.scene_planner import plan_scenes
from src.story_generator import generate_story, load_recent_signatures, record_story_signature
from src.subtitle_generator import generate_srt_subtitles
from src.universe_builder import build_universe, render_universe_backdrop
from src.video_generator import generate_image_to_video_clip
from src.video_renderer import concatenate_and_burn_subtitles
from src.voice_generator import generate_scene_voice


def main() -> int:
    parser = argparse.ArgumentParser(description="CuteVerse Auto-Only Pipeline")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--language", type=str, default="en", choices=["en", "hi", "gu"])
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    mode = "auto"
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "output" / "auto_reels" / f"ep_{args.episode:03d}_{args.language}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("cuteverse.auto")

    try:
        settings = json.loads((repo_root / "config" / "settings.json").read_text(encoding="utf-8"))
        duration = max(30, min(60, int(args.duration)))
        rng = random.Random(args.episode * 1009 + len(args.language))
        num_chars = rng.choice([1, 2, 3])

        history_path = repo_root / "output" / "history" / "recent_runs.json"
        recent_sigs = load_recent_signatures(history_path)

        universe_data = build_universe(mode=mode, custom_inputs={}, rng=rng)
        characters = build_character_dna(
            mode=mode,
            num_characters=num_chars,
            custom_inputs={},
            saved_profiles_path=repo_root / "config" / "saved_profiles.json",
            assets_dir=out_dir / "assets",
            rng=rng,
        )
        story = generate_story(
            mode=mode,
            language=args.language,
            characters=characters,
            universe_data=universe_data,
            custom_inputs={},
            rng=rng,
            recent_signatures=recent_sigs,
        )
        scenes = plan_scenes(story, characters, universe_data, duration, rng)

        story_json = out_dir / "story.json"
        chars_json = out_dir / "characters.json"
        univ_json = out_dir / "universe.json"
        scenes_json = out_dir / "scenes.json"
        story_json.write_text(json.dumps(story, indent=2, ensure_ascii=False), encoding="utf-8")
        chars_json.write_text(json.dumps(characters, indent=2, ensure_ascii=False), encoding="utf-8")
        univ_json.write_text(json.dumps(universe_data, indent=2, ensure_ascii=False), encoding="utf-8")
        scenes_json.write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")

        lang_cfg = settings.get("languages", {}).get(args.language, {
            "voice_char_a": "en-IN-NeerjaNeural",
            "voice_char_b": "en-IN-PrabhatNeural",
        })

        clip_paths: list[Path] = []
        voice_wavs: list[Path] = []
        render_mode = "FALLBACK_RENDER"

        for sc in scenes:
            s_num = sc["scene_number"]
            voice_name = lang_cfg["voice_char_a"] if sc["speaker_id"] == "char_1" else lang_cfg["voice_char_b"]
            tts_res = generate_scene_voice(
                text=sc["dialogue"],
                voice=voice_name,
                output_wav=out_dir / "work" / f"voice_{s_num:02d}.wav",
                duration_sec=float(sc["duration_sec"]),
                speaker_id=sc["speaker_id"],
            )
            wav_p = Path(tts_res["wav_path"]) if isinstance(tts_res, dict) else Path(tts_res)
            voice_wavs.append(wav_p)

            remote_clip, mode_tag = generate_image_to_video_clip(
                scene=sc,
                output_mp4=out_dir / "work" / f"remote_{s_num:02d}.mp4",
                remote_cfg=settings.get("remote_video", {}),
            )
            if remote_clip and remote_clip.exists():
                clip_paths.append(remote_clip)
                render_mode = mode_tag
            else:
                bg_p = render_universe_backdrop(
                    universe_data=universe_data,
                    scene_index=s_num,
                    out_path=out_dir / "work" / f"bg_{s_num:02d}.png",
                    width=1080,
                    height=1920,
                )
                clip_p = render_scene_animation_clip(
                    scene=sc,
                    characters=characters,
                    backdrop_path=bg_p,
                    voice_wav_path=wav_p,
                    work_dir=out_dir / "work",
                    clip_out_path=out_dir / "work" / f"clip_{s_num:02d}.mp4",
                    width=1080,
                    height=1920,
                    fps=30,
                )
                clip_paths.append(clip_p)

        mixed_audio = mix_episode_audio(
            scenes=scenes,
            voice_wavs=voice_wavs,
            output_audio_path=out_dir / "work" / "mixed.m4a",
            total_duration_sec=float(duration),
        )
        srt_path = generate_srt_subtitles(scenes, out_dir / "work" / "captions.srt")
        final_reel = out_dir / "final_reel.mp4"
        concatenate_and_burn_subtitles(
            clip_paths=clip_paths,
            mixed_audio_path=mixed_audio,
            srt_path=srt_path,
            final_mp4_path=final_reel,
            total_duration_sec=float(duration),
            language=args.language,
            width=1080,
            height=1920,
            fps=30,
        )

        gen_report = out_dir / "generation_report.json"
        gen_report.write_text(
            json.dumps({
                "mode": mode,
                "render_mode": render_mode,
                "episode": args.episode,
                "language": args.language,
                "duration_sec": duration,
                "universe": universe_data["universe"],
                "character_count": len(characters),
                "scene_count": len(scenes),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        qual_report = out_dir / "quality_report.json"
        qc = validate_reel_output(
            reel_path=final_reel,
            required_artifacts=[story_json, chars_json, univ_json, scenes_json, gen_report],
            expected_duration_sec=float(duration),
            render_mode=render_mode,
            report_path=qual_report,
        )
        if not qc["passed"]:
            raise RuntimeError(f"Quality check failed: {qc}")

        record_story_signature(history_path, story["signature"])
        logger.info("Auto reel completed (%s): %s", render_mode, final_reel)
        return 0
    except Exception as exc:
        logger.exception("Auto pipeline error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
