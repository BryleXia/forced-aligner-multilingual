"""
路线A 多进程并行版：多个音频文件同时 CTC 对齐

核心思路：
- 每个子进程独立加载 wav2vec2 + VAD 模型到 GPU
- wav2vec2 ~1.2GB × N 进程，RTX 5090 (32GB) 轻松容纳 5-8 个
- 所有核心对齐逻辑复用 align_srt_routeA.py，本脚本只负责并行调度

用法：
  python align_srt_routeA_multi.py --lang es --audio-dir /root/input --output-dir /root/aligned_routeA --workers 5
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
import argparse

# ──────────────────────── 环境变量（必须在 import whisperx 前设置） ────────────────────────

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch"

# 复用原脚本的全部核心函数
import align_srt_routeA as routeA

# ──────────────────────── SRT 文件匹配 ────────────────────────


def find_srt_for_audio(audio_path):
    """
    按优先级匹配 SRT 文件：
      1. {stem}_tgt.asr.qc.srt  （生产格式：seg001.m4a → seg001_tgt.asr.qc.srt）
      2. {stem}.asr.qc.srt       （原格式：seg001.m4a → seg001.asr.qc.srt）
    """
    stem = audio_path.stem
    parent = audio_path.parent
    for pattern in [f"{stem}_tgt.asr.qc.srt", f"{stem}.asr.qc.srt"]:
        p = parent / pattern
        if p.exists():
            return p
    return None


# ──────────────────────── 子进程 Worker ────────────────────────


def worker_fn(task):
    """
    子进程入口：独立加载模型 → 处理一个文件 → 写出结果。

    参数 task 是一个 dict:
      audio_path, srt_path, output_path, language, device
    返回 (文件名, 状态, 耗时秒, 消息)
    """
    audio_path = Path(task["audio_path"])
    srt_path = Path(task["srt_path"])
    output_path = Path(task["output_path"])
    language = task["language"]
    device = task["device"]
    worker_id = task["worker_id"]

    t0 = time.time()
    prefix = f"[W{worker_id}]"

    try:
        # ── 1. 设置全局 LANGUAGE（normalize_for_dp 依赖它）──
        routeA.LANGUAGE = language

        # ── 2. 加载对齐模型 ──
        print(f"{prefix} 加载对齐模型... ({audio_path.name})")
        lang_cfg = routeA.LANG_CONFIG[language]
        load_kwargs = {"language_code": language, "device": device}
        if lang_cfg["align_model_name"]:
            load_kwargs["model_name"] = lang_cfg["align_model_name"]

        import whisperx
        align_model, metadata = whisperx.load_align_model(**load_kwargs)

        # ── 3. 加载 VAD 模型 ──
        vad_model = routeA._load_vad_model_safe()

        # ── 4. 处理文件（复用原脚本的核心逻辑）──
        print(f"{prefix} 处理: {audio_path.name}")

        srt_lines = routeA.parse_srt(srt_path)
        print(f"{prefix}   原始字幕: {len(srt_lines)} 条")

        audio = whisperx.load_audio(str(audio_path))
        audio_duration = len(audio) / 16000
        print(f"{prefix}   音频时长: {audio_duration:.1f}s")

        # 构造 segments → CTC 对齐
        segments = routeA.build_segments_from_srt(srt_lines, audio_duration)
        print(f"{prefix}   共 {len(segments)} 个对齐块")

        aligned = whisperx.align(
            segments, align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        words = routeA.extract_words(aligned)
        print(f"{prefix}   获得词级时间戳: {len(words)} 个词")

        if not words:
            print(f"{prefix}   警告：未获得任何词级时间戳！")
            output_segments = [{"start": 0, "end": 0.5, "text": line} for line in srt_lines]
        else:
            srt_total_words = sum(len(routeA.normalize_for_dp(l)) for l in srt_lines)
            print(f"{prefix}   SRT总词数: {srt_total_words}  ASR词数: {len(words)}")

            output_segments = routeA.match_srt_to_words_dp(srt_lines, words)
            print(f"{prefix}   映射完成: {len(output_segments)} 条")

            output_segments = routeA.snap_outlier_starts(
                output_segments, audio, vad_model=vad_model
            )

        # ── 5. 写出结果 ──
        routeA.write_srt(output_segments, output_path)
        elapsed = time.time() - t0
        print(f"{prefix}   已保存: {output_path.name}  ({elapsed:.1f}s)")
        return (audio_path.name, "OK", elapsed, f"{len(srt_lines)} 条字幕")

    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        print(f"{prefix}   失败: {e}")
        traceback.print_exc()
        return (audio_path.name, "FAIL", elapsed, str(e))


# ──────────────────────── 主入口 ────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="路线A 多进程并行版：多个音频文件同时 CTC 对齐"
    )
    parser.add_argument("--lang", default="es", choices=routeA.LANG_CONFIG.keys(),
                        help="语言代码 (默认: es)")
    parser.add_argument("--audio-dir", required=True,
                        help="音频+SRT 所在目录")
    parser.add_argument("--output-dir", required=True,
                        help="对齐结果输出目录")
    parser.add_argument("--workers", type=int, default=5,
                        help="并行进程数 (默认: 5)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 扫描音频文件 ──
    audio_files = sorted(
        f for ext in ("*.m4a", "*.mp3", "*.wav", "*.flac")
        for f in audio_dir.glob(ext)
    )
    if not audio_files:
        print(f"未找到音频文件！目录: {audio_dir}")
        sys.exit(1)

    # ── 匹配 SRT ──
    tasks = []
    for i, audio_path in enumerate(audio_files):
        srt_path = find_srt_for_audio(audio_path)
        if srt_path is None:
            print(f"  跳过（找不到对应 SRT）: {audio_path.name}")
            continue
        tasks.append({
            "audio_path": str(audio_path),
            "srt_path": str(srt_path),
            "output_path": str(output_dir / f"{audio_path.stem}.aligned.srt"),
            "language": args.lang,
            "device": "cuda",
            "worker_id": i + 1,
        })

    if not tasks:
        print("没有可处理的音频-SRT 对！")
        sys.exit(1)

    # ── 打印任务概览 ──
    lang_cfg = routeA.LANG_CONFIG[args.lang]
    print(f"语言: {args.lang} ({lang_cfg['name']})")
    print(f"对齐模型: {lang_cfg['align_model_name'] or 'WhisperX default'}")
    print(f"并行进程: {min(args.workers, len(tasks))}")
    print(f"待处理: {len(tasks)} 个文件")
    print()
    for t in tasks:
        print(f"  {Path(t['audio_path']).name}  ←  {Path(t['srt_path']).name}")
    print()

    # ── 多进程并行 ──
    t_start = time.time()
    n_workers = min(args.workers, len(tasks))

    # CUDA 要求 spawn 模式
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, tasks)

    t_total = time.time() - t_start

    # ── 汇总报告 ──
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    ok_count = 0
    for name, status, elapsed, msg in results:
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  [{icon}] {name:40s}  {elapsed:6.1f}s  {msg}")
        if status == "OK":
            ok_count += 1

    print(f"\n成功: {ok_count}/{len(results)}  总耗时: {t_total:.1f}s")
    if ok_count > 1:
        serial_est = sum(r[2] for r in results if r[1] == "OK")
        print(f"串行估算: {serial_est:.1f}s  加速比: {serial_est / t_total:.1f}x")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
