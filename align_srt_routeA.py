"""
路线A：绕过 Whisper ASR，直接把 SRT 文本喂给 whisperx.align()

核心思路：
- Whisper ASR 转录会把专名（老包、努登等）拼错 → LCS/DP 匹配困难
- 路线A：直接用 SRT 原文构造 segments → whisperx.align() → 词级时间戳
- 因为输入文本就是 SRT 本身（正确拼写），DP 匹配几乎全是精确命中
- 省去 Whisper 模型加载，只需 wav2vec2 对齐模型

对比实验目录：
  8号  → /root/aligned/
  路线A → /root/aligned_routeA/
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch"

import shutil
import imageio_ffmpeg
ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
ffmpeg_copy = ffmpeg_exe.parent / "ffmpeg.exe"
if not ffmpeg_copy.exists():
    shutil.copy(ffmpeg_exe, ffmpeg_copy)
os.environ["PATH"] = str(ffmpeg_exe.parent) + os.pathsep + os.environ.get("PATH", "")

import whisperx

AUDIO_DIR  = Path("/root")
OUTPUT_DIR = Path("/root/aligned_routeA")
DEVICE     = "cuda"
LANGUAGE   = "es"

# wav2vec2 单段最大处理时长（秒）。超出则分块对齐。
# 路线A 把全段文本放进一个 segment，若音频过长会 OOM，按此分块。
MAX_CHUNK_DURATION = 1800.0  # 30 分钟，确保单段音频整体处理

OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────── 公共工具 ────────────────────────────

def parse_srt(srt_path):
    """解析 SRT，只取文字，忽略原始时间戳"""
    text = Path(srt_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", text.strip())
    segments = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        text_content = " ".join(lines[2:]).strip()
        if text_content:
            segments.append(text_content)
    return segments


def seconds_to_srt_time(s):
    ms = int(round((s % 1) * 1000))
    total_s = int(s)
    hh = total_s // 3600
    mm = (total_s % 3600) // 60
    ss = total_s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(segments, out_path):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def normalize_for_dp(text):
    """规范化文本为词序列，连字符拆为空格，数字展开为西语读法"""
    try:
        from num2words import num2words as _n2w
        text = re.sub(r'\b\d+\b', lambda m: _n2w(int(m.group()), lang='es'), text)
    except Exception:
        pass
    text = text.lower()
    text = text.replace("-", " ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def levenshtein(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[lb]


def fuzzy_match(a, b, threshold=0.40):
    if a == b:
        return True
    max_len = max(len(a), len(b))
    if max_len == 0:
        return True
    return levenshtein(a, b) / max_len <= threshold


def extract_words(aligned_result):
    """从 whisperx align 结果中提取所有词及其时间戳"""
    words = []
    for seg in aligned_result["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word":  w["word"].strip(),
                    "start": w["start"],
                    "end":   w["end"],
                })
    return words


# ──────────────────── 后处理：词速异常值守门员 v2 ─────────────────

def _load_vad_model_safe():
    """尝试加载 Silero-VAD，失败则返回 None（触发 RMS fallback）"""
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        print("  Silero-VAD 加载成功")
        return model
    except Exception as e:
        print(f"  Silero-VAD 加载失败 ({e})，将使用 RMS 兜底")
        return None


def snap_outlier_starts(segments, audio, vad_model=None, sr=16000,
                        min_move_s=1.5,
                        min_words_per_sec=1.3,
                        min_suspect_duration=6.0):
    """
    词速异常值守门员：只修正词速异常低的字幕行（场景转场早开始）。

    触发条件（两者同时满足）：
      1. 词数 / 时长 < min_words_per_sec（正常说话 2-4 词/秒，低于此为可疑）
      2. 时长 > min_suspect_duration（短行豁免，避免误判慢语速行）

    检测到可疑行后，在该行时间窗口 [orig_start + min_move_s, orig_end] 内
    搜索第一个 VAD / RMS 语音起点，吸附过去。
    """
    # 预计算 VAD 时间段（一次性，全音频）
    speech_ts = None
    if vad_model is not None:
        try:
            import torch
            from silero_vad import get_speech_timestamps
            audio_tensor = torch.from_numpy(audio).float()
            speech_ts = get_speech_timestamps(
                audio_tensor, vad_model,
                sampling_rate=sr,
                return_seconds=True,
            )
        except Exception as e:
            print(f"  VAD 时间段提取失败 ({e})，使用 RMS")

    # 预计算 RMS（VAD 失败时的 fallback）
    rms_data = None
    if speech_ts is None:
        import numpy as np
        frame_ms  = 20
        frame_len = int(sr * frame_ms / 1000)
        n_frames  = len(audio) // frame_len
        if n_frames > 0:
            rms = np.array([
                np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
                for i in range(n_frames)
            ])
            ref_level = np.percentile(rms, 70)
            rms_data  = (rms, ref_level * 0.15, frame_ms / 1000.0)

    snapped = 0
    result  = []
    for i, seg in enumerate(segments):
        orig_start = seg["start"]
        orig_end   = seg["end"]
        duration   = orig_end - orig_start
        word_count = len(normalize_for_dp(seg["text"]))
        prev_end   = segments[i - 1]["end"] if i > 0 else None

        # 判断是否可疑（三个条件同时满足）：
        #   1. 词速过低（被拉伸到了静音段里）
        #   2. 时长足够长
        #   3. 与上一行之间只有极小间隔（clamping留下的50ms缝），
        #      说明对齐器是被迫紧接上一行，而非自然找到的起点。
        #      如果前面有自然间隔（>0.3s），则起点是可信的，不动。
        gap_to_prev = (orig_start - prev_end) if prev_end is not None else 999.0
        is_suspect = (
            word_count >= 3
            and duration >= min_suspect_duration
            and word_count / duration < min_words_per_sec
            and gap_to_prev < 0.3
        )

        if not is_suspect:
            result.append(seg)
            continue

        # 在 [orig_start + min_move_s, orig_end] 内搜索语音起点
        search_from = orig_start + min_move_s
        new_start   = orig_start

        if speech_ts is not None:
            for sp in speech_ts:
                if sp["start"] >= search_from and sp["start"] <= orig_end:
                    new_start = sp["start"]
                    snapped  += 1
                    break
        elif rms_data is not None:
            rms, threshold, frame_dur = rms_data
            f_from = int(search_from / frame_dur)
            f_end  = int(orig_end   / frame_dur)
            for f in range(f_from, min(f_end, len(rms))):
                if rms[f] >= threshold:
                    new_start = f * frame_dur
                    snapped  += 1
                    break

        result.append({**seg, "start": new_start})

    print(f"  词速异常修正: {snapped} 行起点被吸附"
          f"（触发阈值: <{min_words_per_sec} 词/秒 且时长 >{min_suspect_duration}s）")
    return result


# ──────────────────────────── DP 对齐 ────────────────────────────

def dp_align(srt_seq, asr_seq):
    """LCS 全局对齐（与 8 号相同）"""
    N, M = len(srt_seq), len(asr_seq)
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if srt_seq[i-1][0] == asr_seq[j-1][0]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    pairs = []
    i, j = N, M
    while i > 0 and j > 0:
        if srt_seq[i-1][0] == asr_seq[j-1][0] and dp[i][j] == dp[i-1][j-1] + 1:
            pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def match_srt_to_words_dp(srt_lines, words):
    """
    全局 DP 对齐：把 SRT 词序列映射到词级时间戳。
    路线A 中，words[] 由 SRT 文本生成，精确匹配率极高。
    """
    if not words:
        return [{"start": 0, "end": 0, "text": line} for line in srt_lines]

    srt_seq = []
    for i, line in enumerate(srt_lines):
        for w in normalize_for_dp(line):
            srt_seq.append((w, i))

    asr_seq = []
    for j, w in enumerate(words):
        for part in normalize_for_dp(w["word"]):
            asr_seq.append((part, j))

    pairs = dp_align(srt_seq, asr_seq)

    line_first = {}
    line_last  = {}
    for srt_i, asr_i in pairs:
        line_idx     = srt_seq[srt_i][1]
        asr_word_idx = asr_seq[asr_i][1]
        if line_idx not in line_first:
            line_first[line_idx] = asr_word_idx
        line_last[line_idx] = asr_word_idx

    result = []
    for i, line in enumerate(srt_lines):
        if i in line_first:
            seg_start = words[line_first[i]]["start"]
            seg_end   = words[line_last[i]]["end"]
            result.append({"start": seg_start, "end": seg_end, "text": line})
        else:
            prev_end = result[-1]["end"] if result else 0
            result.append({"start": prev_end, "end": prev_end + 0.5, "text": line})

    # 事后钳制：强制每行在下一行开始前 50ms 结束
    for i in range(len(result) - 1):
        next_start = result[i + 1]["start"]
        if result[i]["end"] > next_start - 0.05:
            result[i]["end"] = max(next_start - 0.05, result[i]["start"] + 0.1)

    return result


# ──────────────────────── 路线A 核心逻辑 ─────────────────────────

def build_segments_from_srt(srt_lines, audio_duration):
    """
    用 SRT 文本构造 whisperx.align() 所需的 segments 列表。

    策略：把全部 SRT 行合并为一个大 segment（start=0, end=duration）。
    wav2vec2 CTC 对齐器拿到正确文本 + 完整音频，自行找到每个词的边界。

    若音频超过 MAX_CHUNK_DURATION，按等份分块，每块分配对应行的文本。
    """
    if audio_duration <= MAX_CHUNK_DURATION:
        full_text = " ".join(srt_lines)
        return [{"text": full_text, "start": 0.0, "end": audio_duration}]

    # 超长音频：按字数比例分块
    print(f"  音频时长 {audio_duration:.1f}s > {MAX_CHUNK_DURATION}s，分块对齐...")
    n_chunks = int(audio_duration / MAX_CHUNK_DURATION) + 1
    chunk_size = len(srt_lines) // n_chunks
    segs = []
    for k in range(n_chunks):
        start_line = k * chunk_size
        end_line   = start_line + chunk_size if k < n_chunks - 1 else len(srt_lines)
        chunk_text = " ".join(srt_lines[start_line:end_line])
        t_start    = k * MAX_CHUNK_DURATION
        t_end      = min((k + 1) * MAX_CHUNK_DURATION, audio_duration)
        segs.append({"text": chunk_text, "start": t_start, "end": t_end})
    return segs


def process_file(audio_path, srt_path, align_model, metadata, vad_model=None):
    print(f"\n处理: {audio_path.name}")

    srt_lines = parse_srt(srt_path)
    print(f"  原始字幕: {len(srt_lines)} 条")

    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000   # whisperx 固定 16kHz 采样
    print(f"  音频时长: {audio_duration:.1f}s")

    # ── 路线A 核心：用 SRT 文本构造 segments，直接对齐 ──
    print("  构造 SRT segments → whisperx.align()...")
    segments = build_segments_from_srt(srt_lines, audio_duration)
    print(f"  共 {len(segments)} 个对齐块")

    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words = extract_words(aligned)
    print(f"  获得词级时间戳: {len(words)} 个词")

    if not words:
        print("  警告：未获得任何词级时间戳！检查音频/模型。")
        return [{"start": 0, "end": 0.5, "text": line} for line in srt_lines]

    # 打印 LCS 匹配率诊断
    srt_total_words = sum(len(normalize_for_dp(l)) for l in srt_lines)
    print(f"  SRT总词数: {srt_total_words}  ASR词数: {len(words)}")

    # ── DP 对齐：词时间戳 → SRT 行时间戳 ──
    output_segments = match_srt_to_words_dp(srt_lines, words)
    print(f"  映射完成: {len(output_segments)} 条")

    # 后处理：异常值守门员（只修正起点比实际语音早超过1.5秒的行）
    output_segments = snap_outlier_starts(output_segments, audio, vad_model=vad_model)

    return output_segments


# ──────────────────────────── 主入口 ─────────────────────────────

def main():
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
    if not audio_files:
        print("未找到音频文件！")
        sys.exit(1)

    print(f"找到 {len(audio_files)} 个音频文件")
    print("加载对齐模型（西语 wav2vec2）...")
    align_model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )
    print("模型加载完成！")
    print("加载 VAD 模型（静音检测）...")
    vad_model = _load_vad_model_safe()
    print()

    for audio_path in audio_files:
        stem     = audio_path.stem
        srt_path = AUDIO_DIR / f"{stem}.asr.qc.srt"
        if not srt_path.exists():
            print(f"  跳过（找不到对应 SRT）: {audio_path.name}")
            continue

        try:
            segments = process_file(audio_path, srt_path, align_model, metadata, vad_model=vad_model)
            out_path = OUTPUT_DIR / f"{stem}.aligned.srt"
            write_srt(segments, out_path)
            print(f"  已保存: {out_path.name}")
        except Exception as e:
            import traceback
            print(f"  失败: {e}")
            traceback.print_exc()

    print("\n全部完成！路线A结果在:", OUTPUT_DIR)
    print("对比命令（在服务器上执行）：")
    print("  diff /root/aligned/seg001.aligned.srt /root/aligned_routeA/seg001.aligned.srt")


if __name__ == "__main__":
    main()
