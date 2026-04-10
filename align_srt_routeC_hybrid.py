"""
方案 C：混合对齐（B 打锚点 → A 精对齐）

核心思路：
  方案 B（ASR + SM + LLM）不需要完美——只要对齐成功的句子能当"锚点"，
  把失败句子（不管几句连续）夹在两锚点之间，就形成了有效的时间窗口。
  方案 A 的 CTC 对齐器在这个窗口内精确对齐，输出词级时间戳。

流程：
  1. faster-whisper 转录 + SequenceMatcher 初筛 + LLM 兜底
       → anchors = {句子编号: (start, end)}
  2. 按锚点将所有句子切成若干"块"，每块分配时间窗口
       - 锚点块（单句）：[b_start - margin, b_end + margin]
       - 夹住的 gap 块：[前锚点窗口右端, 后锚点窗口左端]
       - 头部/尾部 gap：单侧用音频边界
       - B 全失败兜底：整块覆盖全音频，退化为纯方案 A
  3. 对每块：whisperx.align() 在窗口内精对齐 → 词级时间戳
  4. 合并所有块 → 守门员后处理 → 输出 SRT

输入（与方案 A/B 相同）：
  /root/*.m4a              音频文件
  /root/*.asr.qc.srt       已质检的字幕文本

输出：
  /root/aligned_routeC/*.aligned.srt
"""

import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    from num2words import num2words as _n2w
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch"

import whisperx
from faster_whisper import WhisperModel
from openai import OpenAI

# ── 公共配置 ──────────────────────────────────────────────────────
AUDIO_DIR  = Path("/root")
OUTPUT_DIR = Path("/root/aligned_routeC")
DEVICE     = "cuda"
LANGUAGE   = "ru"
LANG_CONFIG = {
    "es": {"name": "Spanish", "align_model_name": None},
    "fr": {"name": "French",  "align_model_name": None},
    "ru": {"name": "Russian", "align_model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-russian"},
}

# ── 方案 A 配置 ───────────────────────────────────────────────────
MAX_CHUNK_DURATION = 1800.0   # 兜底分块上限（秒），正常不触发

# ── 方案 B 配置：faster-whisper ──────────────────────────────────
WHISPER_MODEL_PATH   = ""     # 留空则自动下载
WHISPER_MODEL_SIZE   = "large-v3"
WHISPER_COMPUTE_TYPE = "float16"

# ── 方案 B 配置：LLM ─────────────────────────────────────────────
LLM_API_KEY     = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL    = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen3.6-plus"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 2000
LLM_MAX_RETRIES = 3
LLM_TIMEOUT     = 120.0

# ── 方案 B 配置：对齐参数 ────────────────────────────────────────
SM_HIGH_CONF    = 0.40
SM_WINDOW       = 80
SM_MERGE_MAX    = 20
SM_LENGTH_RATIO = 3.0
LLM_BATCH       = 8
LLM_DELAY       = 1.0
HALL_SIM        = 0.70
HALL_LOOKBACK   = 5

# ── 方案 C 新增配置 ──────────────────────────────────────────────
ANCHOR_MARGIN = 0.5   # 锚点块左右各扩展的秒数（自适应上限）

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


# ──────────── 方案 B 文本规范化（SM 用） ─────────────────────────

def normalize(text):
    """文本规范化：小写 + Unicode NFD + 去标点 + 合并空格"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def sim(a, b):
    """SequenceMatcher 字符相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ──────────── 方案 A 文本规范化（DP 对齐用） ──────────────────────

def normalize_for_dp(text):
    """规范化文本为词序列，连字符拆为空格，数字展开为读法"""
    if _HAS_NUM2WORDS:
        text = re.sub(r'\b\d+\b', lambda m: _n2w(int(m.group()), lang=LANGUAGE), text)
    text = text.lower()
    text = text.replace("-", " ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


# ──────────────────── 方案 A：DP 词对齐 ──────────────────────────

def dp_align(srt_seq, asr_seq):
    """LCS 全局对齐"""
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


def extract_words(aligned_result):
    """从 whisperx align 结果中提取词级时间戳（绝对时间）"""
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


def match_srt_to_words_dp(srt_lines, words):
    """
    全局 DP 对齐：把 srt_lines 的词序列映射到 words 的时间戳。
    返回 [{"start", "end", "text"}, ...]，未匹配行用前邻时间戳插值。
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

    # 钳制：强制每行在下一行开始前 50ms 结束
    for i in range(len(result) - 1):
        next_start = result[i + 1]["start"]
        if result[i]["end"] > next_start - 0.05:
            result[i]["end"] = max(next_start - 0.05, result[i]["start"] + 0.1)

    return result


# ──────────── 方案 A：VAD 守门员 ──────────────────────────────────

def _load_vad_model_safe():
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
    """词速异常值守门员：修正起点过早的字幕行"""
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

    rms_data = None
    if speech_ts is None:
        import numpy as np
        frame_ms  = 20
        frame_len = int(sr * frame_ms / 1000)
        n_frames  = len(audio) // frame_len
        if n_frames > 0:
            frames    = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
            rms       = np.sqrt((frames ** 2).mean(axis=1))
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


# ──────────── 方案 B：Whisper 转录 ───────────────────────────────

def load_whisper():
    if WHISPER_MODEL_PATH:
        model = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE,
                             compute_type=WHISPER_COMPUTE_TYPE)
    else:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE,
                             compute_type=WHISPER_COMPUTE_TYPE)
    print(f"  faster-whisper 加载完成（{WHISPER_MODEL_SIZE}）")
    return model


def transcribe(model, audio_path):
    segs_raw, _ = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, threshold=0.3),
        temperature=0.0,
        beam_size=5,
    )
    segs = [{"text": s.text.strip(), "start": s.start, "end": s.end}
            for s in segs_raw if s.text.strip()]
    print(f"    转录: {len(segs)} 段")
    return segs


def filter_hallucinations(segs):
    norms = [normalize(s["text"]) for s in segs]
    clean = []
    for i, seg in enumerate(segs):
        is_hall = any(
            sim(norms[i], norms[j]) > HALL_SIM
            for j in range(max(0, i - HALL_LOOKBACK), i)
        )
        if not is_hall:
            clean.append(seg)
    removed = len(segs) - len(clean)
    if removed:
        print(f"    幻觉过滤: 移除 {removed} 段")
    return clean


# ──────────── 方案 B：SM + LLM 对齐 ─────────────────────────────

def sm_align(segs, sentences):
    """
    SequenceMatcher 初筛。
    返回 {idx: {"start", "end", "score", "method"}}
    method = "sm"（成功）或 "pending_llm"（需 LLM）
    """
    results = {}
    cursor  = 0
    n       = len(segs)

    for global_idx, text in enumerate(sentences):
        ref_norm = normalize(text)
        best_s, best_e, best_score = cursor, min(cursor + 1, n), 0.0

        search_end = min(cursor + SM_WINDOW, n)
        for s in range(cursor, search_end):
            acc = ""
            for e in range(s + 1, min(s + SM_MERGE_MAX, search_end + 1)):
                w = segs[e - 1]["text"].strip()
                acc = (acc + " " + w).strip()
                acc_norm = normalize(acc)
                score = sim(ref_norm, acc_norm)
                if score > best_score:
                    best_score = score
                    best_s, best_e = s, e
                if len(acc_norm) > len(ref_norm) * SM_LENGTH_RATIO:
                    break
            if best_score >= 0.85:
                break

        if best_score >= SM_HIGH_CONF and best_e > best_s:
            results[global_idx] = {
                "start": segs[best_s]["start"],
                "end":   segs[best_e - 1]["end"],
                "score": best_score,
                "method": "sm",
            }
            cursor = best_e
        else:
            results[global_idx] = {
                "start": None, "end": None,
                "score": best_score,
                "method": "pending_llm",
            }

    return results


_llm_client = None

def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        if not LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY 未设置。请 export LLM_API_KEY=your-key 后重试。"
            )
        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                             timeout=LLM_TIMEOUT)
    return _llm_client


def call_llm(prompt):
    client = _get_llm_client()
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < LLM_MAX_RETRIES - 1:
                print(f"    [重试 {attempt + 1}] {e}")
                time.sleep(2)
            else:
                raise


def llm_align_batch(segs, sentence_batch):
    lang_cfg   = LANG_CONFIG[LANGUAGE]
    segs_text  = "\n".join(f"[S{i}] {s['text'].strip()}"
                           for i, s in enumerate(segs))
    sent_text  = "\n".join(f"[{idx}] {text}" for idx, text in sentence_batch)

    prompt = f"""你是专业的语音对齐助手。你的任务是找出参考句子与音频转录段落之间的对应关系。

本批文本语言为：{lang_cfg['name']}。

【音频转录段落】（由语音识别生成，可能包含朗读口误、重复、漏读等）：
{segs_text}

【参考句子】（正确文稿，朗读者应尽量朗读这些内容）：
{sent_text}

【任务】找出下列每个参考句子对应哪个或哪几个连续的转录段落。

注意：
- 朗读者可能出现口误（读错词、多读、漏读、重复），转录内容与参考文本可能不完全一致
- 请根据语义内容判断对应关系，而非逐字匹配
- 一句参考文本可能对应多个连续的转录段落
- 找不到匹配时返回 null

只返回 JSON，格式：{{"句子编号": [起始S编号, 结束S编号], ...}}
编号为整数。找不到则返回 null。

示例：{{"1": [0, 0], "2": [1, 3], "3": null}}"""

    response = call_llm(prompt)
    try:
        m = re.search(r'\{[\s\S]*\}', response)
        if not m:
            print(f"    [LLM 响应无 JSON] {response[:200]}")
            return {}
        result = json.loads(m.group())
        if not isinstance(result, dict):
            return {}
        return result
    except json.JSONDecodeError as e:
        print(f"    [LLM JSON 解析失败] {e}")
        return {}


def llm_align(segs, pending_sentences):
    """批量 LLM 对齐，返回 {global_idx: (start, end) or (None, None)}"""
    max_idx = len(segs) - 1
    results = {}
    n_batches = (len(pending_sentences) + LLM_BATCH - 1) // LLM_BATCH
    print(f"    LLM 对齐: {len(pending_sentences)} 句，分 {n_batches} 批")

    for i in range(0, len(pending_sentences), LLM_BATCH):
        batch = pending_sentences[i:i + LLM_BATCH]
        raw   = llm_align_batch(segs, [(idx, text) for idx, text in batch])
        ok    = 0
        for global_idx, text in batch:
            val = raw.get(str(global_idx))
            if val and isinstance(val, list) and len(val) == 2:
                s_i = max(0, min(int(val[0]), max_idx))
                e_i = max(s_i, min(int(val[1]), max_idx))
                results[global_idx] = (segs[s_i]["start"], segs[e_i]["end"])
                ok += 1
            else:
                results[global_idx] = (None, None)
        print(f"    批次 {i // LLM_BATCH + 1}/{n_batches}: {ok}/{len(batch)} 成功")
        time.sleep(LLM_DELAY)

    return results


# ──────────────────── 方案 C：核心新函数 ─────────────────────────

def get_anchors(segs, srt_lines):
    """
    运行完整的方案 B 流程，返回锚点字典。
    anchors = {句子编号: (start, end)}，仅包含对齐成功的句子。
    """
    print("  [锚点] SequenceMatcher 初筛...")
    sm_results = sm_align(segs, srt_lines)
    sm_ok      = sum(1 for r in sm_results.values() if r["method"] == "sm")
    pending    = [(idx, text) for idx, text in enumerate(srt_lines)
                  if sm_results[idx]["method"] == "pending_llm"]
    print(f"  [锚点] SM 成功: {sm_ok}/{len(srt_lines)} 句，"
          f"{len(pending)} 句待 LLM")

    llm_results = {}
    if pending:
        print("  [锚点] LLM 语义裁判...")
        llm_results = llm_align(segs, pending)

    anchors = {}
    for idx in range(len(srt_lines)):
        r = sm_results[idx]
        if r["method"] == "sm":
            anchors[idx] = (r["start"], r["end"])
        elif idx in llm_results:
            ts, te = llm_results[idx]
            if ts is not None:
                anchors[idx] = (ts, te)

    print(f"  [锚点] 共获得 {len(anchors)}/{len(srt_lines)} 个锚点")
    return anchors


def build_blocks(n_sentences, anchors, audio_duration, margin=ANCHOR_MARGIN):
    """
    按锚点将全部句子切成若干块，每块分配时间窗口。

    规则：
    - 锚点块（单句）：窗口 = [b_start - margin, b_end + margin]
    - gap 块（两锚点之间）：窗口 = [前锚点窗口右端, 后锚点窗口左端]
    - 头部 gap（第一锚点之前）：窗口右端 = 第一锚点窗口左端
    - 尾部 gap（最后锚点之后）：窗口左端 = 最后锚点窗口右端
    - 无锚点兜底：单块 [0, audio_duration]（退化为纯方案 A）

    返回：[{"lines": [idx, ...], "window": (ws, we)}, ...]
    """
    if not anchors:
        return [{"lines": list(range(n_sentences)),
                 "window": (0.0, audio_duration)}]

    anchor_set = set(anchors.keys())

    # ── 第一轮：确定锚点块的窗口（含 margin） ──────────────────────
    # 先按顺序收集所有锚点块和 gap 块（gap 块的窗口后续填充）
    raw_blocks = []
    i = 0
    while i < n_sentences:
        if i in anchor_set:
            bs, be = anchors[i]
            raw_blocks.append({
                "lines":  [i],
                "window": (bs - margin, be + margin),
                "is_anchor": True,
            })
            i += 1
        else:
            gap_start = i
            while i < n_sentences and i not in anchor_set:
                i += 1
            raw_blocks.append({
                "lines":  list(range(gap_start, i)),
                "window": None,
                "is_anchor": False,
            })

    # ── 第二轮：为 gap 块填充窗口 ──────────────────────────────────
    for b_idx, block in enumerate(raw_blocks):
        if block["window"] is not None:
            continue

        # 左边界：前一个锚点块的右端，或音频起点
        left = 0.0
        for prev in reversed(raw_blocks[:b_idx]):
            if prev["window"] is not None:
                left = prev["window"][1]
                break

        # 右边界：后一个锚点块的左端，或音频终点
        right = audio_duration
        for nxt in raw_blocks[b_idx + 1:]:
            if nxt["window"] is not None:
                right = nxt["window"][0]
                break

        block["window"] = (left, right)

    # ── 第三轮：修正 margin 溢出 + 确保不重叠 ─────────────────────
    prev_end = 0.0
    for block in raw_blocks:
        ws, we = block["window"]
        ws = max(ws, prev_end)                   # 不早于上一块结束
        ws = max(ws, 0.0)
        we = min(we, audio_duration)
        if we <= ws:
            we = min(ws + 0.5, audio_duration)   # 保底窗口宽度
        block["window"] = (ws, we)
        prev_end = we

    return [{"lines": b["lines"], "window": b["window"]} for b in raw_blocks]


def align_block(block, srt_lines, audio, align_model, metadata):
    """
    对单个块执行方案 A 的 CTC 对齐，返回该块的对齐结果列表。
    """
    lines  = [srt_lines[i] for i in block["lines"]]
    ws, we = block["window"]

    segment = {"text": " ".join(lines), "start": ws, "end": we}
    aligned = whisperx.align(
        [segment],
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )
    words = extract_words(aligned)

    if not words:
        # 兜底：在窗口内均匀分配
        n      = len(lines)
        step   = (we - ws) / max(n, 1)
        return [{"start": ws + k * step,
                 "end":   ws + (k + 1) * step,
                 "text":  line}
                for k, line in enumerate(lines)]

    return match_srt_to_words_dp(lines, words)


# ──────────────────────── 主流程 ─────────────────────────────────

def process_file(whisper_model, audio_path, srt_path, align_model, metadata,
                 vad_model=None):
    print(f"\n处理: {audio_path.name}")

    srt_lines = parse_srt(srt_path)
    print(f"  参考句子: {len(srt_lines)} 条")

    audio          = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000
    print(f"  音频时长: {audio_duration:.1f}s")

    # ── 步骤 1：faster-whisper 转录 ──────────────────────────────
    print("  [步骤 1] faster-whisper 转录...")
    segs_raw = transcribe(whisper_model, audio_path)
    segs     = filter_hallucinations(segs_raw)
    print(f"  有效转录段: {len(segs)} 段")

    if not segs:
        print("  [警告] 转录为空，退化为纯方案 A（全音频单块）")
        segs = []

    # ── 步骤 2：获取锚点 ─────────────────────────────────────────
    print("  [步骤 2] 打锚点（SM + LLM）...")
    if segs:
        anchors = get_anchors(segs, srt_lines)
    else:
        anchors = {}

    # ── 步骤 3：切块 ─────────────────────────────────────────────
    blocks = build_blocks(len(srt_lines), anchors, audio_duration)
    n_anchor_blocks = sum(1 for b in blocks if len(b["lines"]) == 1
                          and b["lines"][0] in anchors)
    n_gap_blocks    = len(blocks) - n_anchor_blocks
    print(f"  [步骤 3] 分块完成: {len(blocks)} 块"
          f"（锚点块 {n_anchor_blocks} 个，gap 块 {n_gap_blocks} 个）")
    for b in blocks:
        ws, we = b["window"]
        tag = "锚" if (len(b["lines"]) == 1 and b["lines"][0] in anchors) else "gap"
        print(f"    [{tag}] 句 {b['lines'][0]}~{b['lines'][-1]} "
              f"→ [{seconds_to_srt_time(ws)}, {seconds_to_srt_time(we)}]")

    # ── 步骤 4：逐块精对齐 ────────────────────────────────────────
    # 锚点块直接用 Route B 时间戳；gap 块走 CTC
    print("  [步骤 4] 逐块精对齐（锚点块跳过 CTC）...")
    all_segments = []
    for b_idx, block in enumerate(blocks):
        ws, we = block["window"]
        lines_idx = block["lines"]

        if len(lines_idx) == 1 and lines_idx[0] in anchors:
            # 锚点块：直接用 Route B 的时间戳
            idx = lines_idx[0]
            bs, be = anchors[idx]
            result = [{"start": bs, "end": be, "text": srt_lines[idx]}]
            print(f"    块 {b_idx + 1}/{len(blocks)}: [锚] 句 {idx} "
                  f"[{seconds_to_srt_time(bs)}→{seconds_to_srt_time(be)}]")
        else:
            # gap 块：CTC 精对齐
            result = align_block(block, srt_lines, audio, align_model, metadata)
            print(f"    块 {b_idx + 1}/{len(blocks)}: [CTC] "
                  f"{len(result)} 句 [{seconds_to_srt_time(ws)}→{seconds_to_srt_time(we)}]")

        all_segments.extend(result)

    # ── 步骤 5：守门员后处理 ─────────────────────────────────────
    print("  [步骤 5] 守门员后处理...")
    all_segments = snap_outlier_starts(all_segments, audio, vad_model=vad_model)

    print(f"  最终输出: {len(all_segments)} 条")
    return all_segments


def main():
    audio_files = sorted(
        f for ext in ("m4a", "mp3", "wav", "flac")
        for f in AUDIO_DIR.glob(f"{LANGUAGE}_*.{ext}")
    )
    if not audio_files:
        print("未找到音频文件！")
        return

    lang_cfg = LANG_CONFIG[LANGUAGE]
    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"语言: {LANGUAGE} ({lang_cfg['name']})")
    print(f"LLM: {LLM_MODEL}（{LLM_BASE_URL}）")
    print(f"对齐模型: {lang_cfg['align_model_name'] or 'WhisperX default'}")
    print()

    print("加载 faster-whisper 模型...")
    whisper_model = load_whisper()

    print("加载 wav2vec2 对齐模型...")
    load_kwargs = {"language_code": LANGUAGE, "device": DEVICE}
    if lang_cfg["align_model_name"]:
        load_kwargs["model_name"] = lang_cfg["align_model_name"]
    align_model, metadata = whisperx.load_align_model(**load_kwargs)
    print("wav2vec2 加载完成！")

    print("加载 VAD 模型...")
    vad_model = _load_vad_model_safe()
    print()

    total_aligned   = 0
    total_sentences = 0

    for audio_path in audio_files:
        stem     = audio_path.stem
        srt_path = AUDIO_DIR / f"{stem}.asr.qc.srt"
        if not srt_path.exists():
            print(f"  跳过（找不到对应 SRT）: {audio_path.name}")
            continue

        try:
            n_ref    = len(parse_srt(srt_path))
            segments = process_file(
                whisper_model, audio_path, srt_path,
                align_model, metadata, vad_model=vad_model,
            )
            if segments:
                out_path = OUTPUT_DIR / f"{stem}.aligned.srt"
                write_srt(segments, out_path)
                print(f"  已保存: {out_path.name}")
                total_aligned   += len(segments)
                total_sentences += n_ref
        except Exception as e:
            import traceback
            print(f"  失败: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"全部完成！方案 C 结果在: {OUTPUT_DIR}")
    print(f"总计: {total_aligned}/{total_sentences} 句对齐")


if __name__ == "__main__":
    main()
