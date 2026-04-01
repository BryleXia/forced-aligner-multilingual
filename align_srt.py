"""
用 WhisperX 对现有 SRT 字幕重新对齐时间戳

正确逻辑：
- 原始 SRT 文字可信，时间戳完全不可信
- 用 large-v3 转录音频 → 得到准确词级时间戳
- 将原始 SRT 每行文字匹配到转录词级时间戳
- 输出：原始文字 + 音频导出的准确时间戳
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ["HF_HOME"] = "E:/ai知识库/cache/huggingface"
os.environ["TORCH_HOME"] = "E:/ai知识库/cache/torch"

# 注入 imageio-ffmpeg 携带的 ffmpeg 二进制到 PATH
import shutil
import imageio_ffmpeg
ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
ffmpeg_copy = ffmpeg_exe.parent / "ffmpeg.exe"
if not ffmpeg_copy.exists():
    shutil.copy(ffmpeg_exe, ffmpeg_copy)
os.environ["PATH"] = str(ffmpeg_exe.parent) + os.pathsep + os.environ.get("PATH", "")

import whisperx

AUDIO_DIR = Path("E:/my-ai-studio/西语SRT对齐初见/文字时间轴对齐案例【raw】")
OUTPUT_DIR = Path("E:/my-ai-studio/西语SRT对齐初见/aligned")
MODEL_PATH = "E:/ai知识库/cache/huggingface/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
DEVICE = "cuda"
LANGUAGE = "es"

OUTPUT_DIR.mkdir(exist_ok=True)


def parse_srt(srt_path):
    """解析 SRT，只取文字，忽略时间戳"""
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


def normalize(text):
    """统一化文字用于比较：去掉标点、小写、去重音"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def extract_words(aligned_result):
    """从 whisperx align 结果中提取所有词及其时间戳"""
    words = []
    for seg in aligned_result["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                })
    return words


def match_srt_to_words(srt_lines, words):
    """
    把原始 SRT 的每行文字匹配到词级时间戳序列。
    策略：把 SRT 所有行拼成一个词序列，与 words 做顺序匹配，
    按 SRT 行的词数分割时间戳。
    """
    # 把 SRT 每行拆成词（归一化用于匹配，原始用于输出）
    srt_word_counts = [len(normalize(line)) for line in srt_lines]
    total_srt_words = sum(srt_word_counts)
    total_audio_words = len(words)

    if total_audio_words == 0:
        return [{"start": 0, "end": 0, "text": line} for line in srt_lines]

    # 按词数比例分配：SRT第i行占比 → 对应音频词索引范围
    result = []
    audio_idx = 0
    for i, (line, wc) in enumerate(zip(srt_lines, srt_word_counts)):
        if wc == 0:
            # 空行，用前一条的结束时间
            prev_end = result[-1]["end"] if result else 0
            result.append({"start": prev_end, "end": prev_end + 0.5, "text": line})
            continue

        # 计算这行对应的音频词范围
        start_idx = audio_idx
        # 按比例分配剩余词
        remaining_srt = sum(srt_word_counts[i:])
        remaining_audio = total_audio_words - audio_idx
        take = max(1, round(wc / remaining_srt * remaining_audio)) if remaining_srt > 0 else 1
        end_idx = min(audio_idx + take, total_audio_words)

        seg_words = words[start_idx:end_idx]
        if seg_words:
            seg_start = seg_words[0]["start"]
            seg_end = seg_words[-1]["end"]
        else:
            prev_end = result[-1]["end"] if result else 0
            seg_start = prev_end
            seg_end = prev_end + 0.5

        result.append({"start": seg_start, "end": seg_end, "text": line})
        audio_idx = end_idx

    return result


def process_file(audio_path, srt_path, whisper_model, align_model, metadata):
    print(f"\n处理: {audio_path.name}")

    srt_lines = parse_srt(srt_path)
    print(f"  原始字幕: {len(srt_lines)} 条")

    audio = whisperx.load_audio(str(audio_path))

    # 第一步：用 large-v3 转录，获取准确时间戳
    print("  转录中（large-v3）...")
    transcribe_result = whisper_model.transcribe(audio, language=LANGUAGE)
    print(f"  转录完成: {len(transcribe_result['segments'])} 段")

    # 第二步：词级对齐
    print("  词级对齐中...")
    aligned = whisperx.align(
        transcribe_result["segments"],
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words = extract_words(aligned)
    print(f"  获得词级时间戳: {len(words)} 个词")

    # 第三步：将原始 SRT 文字映射到词级时间戳
    output_segments = match_srt_to_words(srt_lines, words)
    print(f"  映射完成: {len(output_segments)} 条")

    return output_segments


def main():
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
    if not audio_files:
        print("未找到音频文件！")
        sys.exit(1)

    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"加载转录模型（large-v3）...")
    whisper_model = whisperx.load_model(
        MODEL_PATH,
        device=DEVICE,
        compute_type="float16",
        language=LANGUAGE,
    )
    print("加载对齐模型（西语 wav2vec2）...")
    align_model, metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )
    print("模型全部加载完成！\n")

    for audio_path in audio_files:
        stem = audio_path.stem
        srt_path = AUDIO_DIR / f"{stem}.asr.qc.srt"
        if not srt_path.exists():
            print(f"  跳过（找不到对应 SRT）: {audio_path.name}")
            continue

        try:
            segments = process_file(audio_path, srt_path, whisper_model, align_model, metadata)
            out_path = OUTPUT_DIR / f"{stem}.aligned.srt"
            write_srt(segments, out_path)
            print(f"  已保存: {out_path.name}")
        except Exception as e:
            import traceback
            print(f"  失败: {e}")
            traceback.print_exc()

    print("\n全部完成！结果在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
