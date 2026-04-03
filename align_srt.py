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

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch"

# 注入 imageio-ffmpeg 携带的 ffmpeg 二进制到 PATH
import shutil
import imageio_ffmpeg
ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
ffmpeg_copy = ffmpeg_exe.parent / "ffmpeg.exe"
if not ffmpeg_copy.exists():
    shutil.copy(ffmpeg_exe, ffmpeg_copy)
os.environ["PATH"] = str(ffmpeg_exe.parent) + os.pathsep + os.environ.get("PATH", "")

import whisperx

AUDIO_DIR = Path("/root/文字时间轴对齐案例【raw】")
OUTPUT_DIR = Path("/root/aligned")
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


def normalize_word(w):
    """归一化单个词，用于与ASR输出匹配"""
    w = w.lower()
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = re.sub(r"[^\w]", "", w)
    return w


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
    """字符级编辑距离"""
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
    """归一化编辑距离 ≤ threshold 时认为近似匹配"""
    if a == b:
        return True
    max_len = max(len(a), len(b))
    if max_len == 0:
        return True
    return levenshtein(a, b) / max_len <= threshold


def boundary_extend(result, srt_lines, srt_seq, words, line_first, line_last,
                    look_ahead=6, fuzzy_thr=0.45, max_extend_s=2.5):
    """
    LCS对齐后，检查每行首尾是否有未匹配的SRT词。
    若存在，在words[]的邻近位置做模糊匹配，找回真实边界时间戳。

    look_ahead: 向前/向后最多搜索几个ASR词
    fuzzy_thr:  模糊匹配阈值（归一化编辑距离）
    """
    # 为每行建立：哪些srt_token位置属于该行
    line_srt_positions = {}   # line_idx → [srt_token_idx, ...]
    for si, (word, li) in enumerate(srt_seq):
        line_srt_positions.setdefault(li, []).append(si)

    # 收集LCS匹配的srt_token_idx集合
    # 需要重新从result反推——直接用line_first/line_last即可
    n_words = len(words)

    # 需要追踪的高风险词
    WATCH = {"shangri", "lao", "bao", "nuodeng", "nuodong", "liang", "zhuoma", "yuanes", "yenes"}

    for i, seg in enumerate(result):
        if i not in line_first:
            continue  # 插值行，跳过

        positions = line_srt_positions.get(i, [])
        if not positions:
            continue

        first_asr = line_first[i]
        srt_words_in_line = [srt_seq[p][0] for p in positions]

        # 判断是否是需要追踪的行
        watch_line = any(w in WATCH for w in srt_words_in_line)

        first_asr_norm = []
        for part in normalize_for_dp(words[first_asr]["word"]):
            first_asr_norm.append(part)
        matched_pos_in_line = 0
        for k, sw in enumerate(srt_words_in_line):
            if sw == first_asr_norm[0] or fuzzy_match(sw, first_asr_norm[0], fuzzy_thr):
                matched_pos_in_line = k
                break

        prefix_words = srt_words_in_line[:matched_pos_in_line]

        if watch_line:
            print(f"  [TRACE行{i}] SRT词={srt_words_in_line[:6]}...")
            print(f"    首个LCS锚点: words[{first_asr}]={words[first_asr]['word']!r}@{words[first_asr]['start']:.2f}s → norm={first_asr_norm}")
            print(f"    matched_pos={matched_pos_in_line}, prefix_words={prefix_words}")

        if prefix_words:
            search_start = max(0, first_asr - look_ahead)
            asr_window = [(j, normalize_for_dp(words[j]["word"]))
                          for j in range(search_start, first_asr)]
            new_start_asr = None
            # 策略1：只用第一个前缀词匹配（避免'la'/'un'等常见词误触发）
            first_pw = prefix_words[0]
            for (j, parts) in asr_window:
                for part in parts:
                    if fuzzy_match(first_pw, part, fuzzy_thr):
                        if new_start_asr is None or j < new_start_asr:
                            new_start_asr = j
                        break
            # 策略2：若未找到，尝试前两词拼接（处理ASR合并词，如Laopao=lao+bao）
            if new_start_asr is None and len(prefix_words) >= 2:
                concat_pw = prefix_words[0] + prefix_words[1]
                for (j, parts) in asr_window:
                    for part in parts:
                        if fuzzy_match(concat_pw, part, fuzzy_thr):
                            if new_start_asr is None or j < new_start_asr:
                                new_start_asr = j
                            break
            if watch_line:
                win_str = [(j, words[j]["word"], words[j]["start"]) for j, _ in asr_window]
                print(f"    ASR窗口(idx,词,时间): {win_str}")
                print(f"    找到new_start_asr={new_start_asr}" + (f" @{words[new_start_asr]['start']:.2f}s" if new_start_asr is not None else ""))
            if new_start_asr is not None:
                new_t = words[new_start_asr]["start"]
                anchor_t = words[first_asr]["start"]
                prev_end = result[i-1]["end"] if i > 0 else 0.0
                ok = new_t >= anchor_t - max_extend_s and new_t >= prev_end + 0.1
                if watch_line:
                    print(f"    护栏检查: new_t={new_t:.2f} anchor={anchor_t:.2f} diff={anchor_t-new_t:.2f}s max={max_extend_s} prev_end={prev_end:.2f} → {'通过✓' if ok else '拦截✗'}")
                if ok:
                    seg["start"] = new_t
        elif watch_line:
            print(f"    无前缀漏词，跳过起点回收")

        # ── 终点：找未匹配的后缀词 ──────────────────────────────
        last_asr = line_last[i]
        last_asr_norm = []
        for part in normalize_for_dp(words[last_asr]["word"]):
            last_asr_norm.append(part)
        matched_pos_in_line_end = len(srt_words_in_line) - 1
        for k in range(len(srt_words_in_line) - 1, -1, -1):
            sw = srt_words_in_line[k]
            if sw == last_asr_norm[-1] or fuzzy_match(sw, last_asr_norm[-1], fuzzy_thr):
                matched_pos_in_line_end = k
                break

        suffix_words = srt_words_in_line[matched_pos_in_line_end + 1:]

        if watch_line:
            print(f"    末个LCS锚点: words[{last_asr}]={words[last_asr]['word']!r}@{words[last_asr]['end']:.2f}s → norm={last_asr_norm}")
            print(f"    matched_end_pos={matched_pos_in_line_end}, suffix_words={suffix_words}")

        if suffix_words:
            search_end = min(n_words, last_asr + look_ahead + 1)
            asr_window = [(j, normalize_for_dp(words[j]["word"]))
                          for j in range(last_asr + 1, search_end)]
            new_end_asr = None
            for sw in reversed(suffix_words):
                for (j, parts) in reversed(asr_window):
                    for part in parts:
                        if fuzzy_match(sw, part, fuzzy_thr):
                            if new_end_asr is None or j > new_end_asr:
                                new_end_asr = j
                            break
            if watch_line:
                win_str = [(j, words[j]["word"], words[j]["end"]) for j, _ in asr_window]
                print(f"    后缀ASR窗口: {win_str}")
                print(f"    找到new_end_asr={new_end_asr}" + (f" @{words[new_end_asr]['end']:.2f}s" if new_end_asr is not None else ""))
            if new_end_asr is not None:
                new_t = words[new_end_asr]["end"]
                anchor_t = words[last_asr]["end"]
                # 护栏：不能比LCS锚点晚超过max_extend_s（防止误触发到下一行范围）
                if new_t <= anchor_t + max_extend_s:
                    seg["end"] = new_t

    return result


def dp_align(srt_seq, asr_seq):
    """
    LCS全局对齐（gap penalty=0，只统计完全匹配）。
    srt_seq: [(norm_word, line_idx), ...]
    asr_seq: [(norm_word, word_idx), ...]  ← word_idx 指向 words[] 中的词
    返回: [(srt_token_idx, asr_token_idx), ...] 单调匹配对列表
    """
    N, M = len(srt_seq), len(asr_seq)
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if srt_seq[i-1][0] == asr_seq[j-1][0]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    # 回溯
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
    全局DP对齐：把SRT词序列与ASR词序列做LCS，
    行时间由真实匹配到的首尾token决定，不再靠词数估算。
    """
    if not words:
        return [{"start": 0, "end": 0, "text": line} for line in srt_lines]

    # 展开SRT：每个词带行号
    srt_seq = []
    for i, line in enumerate(srt_lines):
        for w in normalize_for_dp(line):
            srt_seq.append((w, i))

    # 展开ASR：每个词带原始words[]序号（连字符同样拆分）
    asr_seq = []
    for j, w in enumerate(words):
        for part in normalize_for_dp(w["word"]):
            asr_seq.append((part, j))

    # 全局DP对齐
    pairs = dp_align(srt_seq, asr_seq)

    # 从匹配对中提取每行的首尾ASR词
    line_first = {}   # line_idx → asr word_idx（最早匹配）
    line_last  = {}   # line_idx → asr word_idx（最晚匹配）
    for srt_i, asr_i in pairs:
        line_idx     = srt_seq[srt_i][1]
        asr_word_idx = asr_seq[asr_i][1]
        if line_idx not in line_first:
            line_first[line_idx] = asr_word_idx
        line_last[line_idx] = asr_word_idx

    # 构建结果，未匹配行从邻近行插值
    result = []
    for i, line in enumerate(srt_lines):
        if i in line_first:
            seg_start = words[line_first[i]]["start"]
            seg_end   = words[line_last[i]]["end"]
            result.append({"start": seg_start, "end": seg_end, "text": line})
        else:
            prev_end = result[-1]["end"] if result else 0
            result.append({"start": prev_end, "end": prev_end + 0.5, "text": line})

    # 边界词回收：用模糊匹配找回首尾未匹配的词的真实时间戳
    result = boundary_extend(result, srt_lines, srt_seq, words, line_first, line_last)

    # 事后钳制：强制每行在下一行开始前50ms结束
    for i in range(len(result) - 1):
        next_start = result[i + 1]["start"]
        if result[i]["end"] > next_start - 0.05:
            result[i]["end"] = max(next_start - 0.05, result[i]["start"] + 0.1)

    return result


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


def find_line_anchor(srt_words_norm, audio_norm, start_pos, window=40):
    """
    在 audio_norm[start_pos : start_pos+window] 中寻找与 srt_words_norm
    前几个词最匹配的起始位置。返回 audio_norm 的索引，未找到则返回 start_pos。
    """
    best_pos = start_pos
    best_score = -1
    search_end = min(start_pos + window, len(audio_norm))

    for i in range(start_pos, search_end):
        score = 0
        for j, sw in enumerate(srt_words_norm[:3]):  # 只比对前3个词
            if i + j < len(audio_norm) and audio_norm[i + j] == sw:
                score += 1
            else:
                break
        if score > best_score:
            best_score = score
            best_pos = i
        if best_score == 3:  # 前3词全中，不必再搜
            break

    return best_pos


def match_srt_to_words(srt_lines, words):
    """
    把原始 SRT 的每行文字匹配到词级时间戳序列。
    策略：单遍顺序词匹配锚定每行首词，词数估算结束词，
    最后事后钳制：强制每行在下一行开始前结束，消除停顿期滞留。
    """
    if not words:
        return [{"start": 0, "end": 0, "text": line} for line in srt_lines]

    audio_norm = [normalize_word(w["word"]) for w in words]
    audio_pos = 0
    result = []
    n = len(srt_lines)

    for i, line in enumerate(srt_lines):
        srt_wn = normalize(line)
        if not srt_wn:
            prev_end = result[-1]["end"] if result else 0
            result.append({"start": prev_end, "end": prev_end + 0.5, "text": line})
            continue

        # 安全阀：为剩余每行至少保留1个词的位置，防止末尾崩塌
        remaining = n - i - 1
        audio_pos = max(0, min(audio_pos, len(words) - remaining - 1))

        # 关键修复①：允许往回看15个词，防止end_idx多走1步时跳过正确锚点
        search_start = max(0, audio_pos - 15)
        anchor = find_line_anchor(srt_wn, audio_norm, search_start, window=165)
        anchor = min(anchor, len(words) - 1)
        end_idx = min(anchor + len(srt_wn) - 1, len(words) - 1)

        seg_start = words[anchor]["start"]
        seg_end = words[end_idx]["end"]

        result.append({"start": seg_start, "end": seg_end, "text": line})
        # 关键修复②：保守推进，只推进一半词数，防止长句overshoot
        audio_pos = anchor + max(1, len(srt_wn) // 2)

    # 事后钳制：强制每行在下一行开始前50ms结束，消除停顿期滞留
    for i in range(len(result) - 1):
        next_start = result[i + 1]["start"]
        if result[i]["end"] > next_start - 0.05:
            result[i]["end"] = max(next_start - 0.05, result[i]["start"] + 0.1)

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

    # ── 调试：打印高风险词附近的ASR输出 ──────────────────────────
    DEBUG_TARGETS = ["lao", "bao", "nuodeng", "nuodong", "shangri", "aliang", "liang", "zhuoma", "yuanes"]
    print(f"\n  【ASR调试】关键词附近输出：")
    for j, w in enumerate(words):
        wn = normalize_for_dp(w["word"])
        for part in wn:
            if any(fuzzy_match(part, t, 0.45) for t in DEBUG_TARGETS):
                context = words[max(0,j-2):j+3]
                ctx_str = "  ".join(f"{c['word']}({c['start']:.2f})" for c in context)
                print(f"    [{j}] {w['start']:.2f}s  原文:{w['word']!r}  上下文: {ctx_str}")
                break
    print()
    # ────────────────────────────────────────────────────────────

    # 第三步：将原始 SRT 文字映射到词级时间戳（全局DP对齐）
    output_segments = match_srt_to_words_dp(srt_lines, words)
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
