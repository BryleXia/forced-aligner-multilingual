"""
方案 B：ASR + LLM 语义对齐兜底

适用场景：朗读者出现口误（多读、漏读、读错词），导致参考文稿与音频内容不完全一致，
          方案 A（参考文本强制对齐）无法处理此类偏差。

核心流程：
  音频 → faster-whisper 转录（获取"实际说了什么"）
       → SequenceMatcher 初筛（字符相似度匹配）
       → LLM 语义裁判（处理初筛失败的句子，理解口误/近义替换/漏读等语义关系）
       → 时间戳

迁移自：博物馆语料库项目 ja_align_v2.py（日语汉字↔假名场景）
         在日语 212 句上实现 97.6% 对齐率（207/212），其中 LLM 兜底解决了
         SequenceMatcher 无法处理的跨书写系统匹配问题。

输入（与方案 A 相同）：
  /root/*.m4a              音频文件
  /root/*.asr.qc.srt       已质检的字幕文本

输出：
  /root/aligned_routeB/*.aligned.srt
"""

import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from faster_whisper import WhisperModel
from openai import OpenAI

# ── 路径配置 ─────────────────────────────────────────────────────
AUDIO_DIR  = Path("/root")
OUTPUT_DIR = Path("/root/aligned_routeB")
DEVICE     = "cuda"
LANGUAGE   = "ru"
LANG_CONFIG = {
    "es": {"name": "Spanish"},
    "fr": {"name": "French"},
    "ru": {"name": "Russian"},
}

# ── faster-whisper 配置 ─────────────────────────────────────────
# 本地模型路径（AutoDL 环境下预缓存），留空则自动从 HuggingFace 下载
WHISPER_MODEL_PATH = ""  # 例如 "/root/autodl-tmp/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/..."
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_COMPUTE_TYPE = "float16"

# ── LLM 配置（DashScope + Qwen）─────────────────────────────────
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL    = "qwen3.6-plus"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 2000
LLM_MAX_RETRIES = 3
LLM_TIMEOUT     = 120.0  # 秒

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)

# ── 对齐参数 ────────────────────────────────────────────────────
SM_HIGH_CONF = 0.40      # SequenceMatcher 置信度阈值，高于此直接采用
SM_WINDOW    = 80         # 向前搜索的 Whisper segment 窗口大小
SM_MERGE_MAX = 20         # 最多合并多少个连续 segment 来匹配
SM_LENGTH_RATIO = 3.0     # 累积文本超过参考文本此倍数则停止合并

LLM_BATCH   = 8           # LLM 每批处理句子数
LLM_DELAY   = 1.0         # 批次间隔（秒）

HALL_SIM    = 0.70        # 幻觉检测相似度阈值
HALL_LOOKBACK = 5         # 幻觉检测回看窗口

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


def normalize(text):
    """文本规范化：小写 + Unicode 归一化 + 去标点 + 去多余空格"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def sim(a, b):
    """SequenceMatcher 相似度"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ──────────────────── faster-whisper 转录 ────────────────────────

def load_whisper():
    if WHISPER_MODEL_PATH:
        model = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE,
                             compute_type=WHISPER_COMPUTE_TYPE)
    else:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE,
                             compute_type=WHISPER_COMPUTE_TYPE)
    print(f"  Whisper 模型加载完成（{WHISPER_MODEL_SIZE}）")
    return model


def transcribe(model, audio_path):
    """faster-whisper 转录，返回 segment 列表"""
    segs_raw, _ = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        word_timestamps=True,
        condition_on_previous_text=False,   # 防止循环幻觉（保加利亚语方案经验）
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
    """过滤重复幻觉段（与近期 segment 高度相似的视为幻觉）"""
    norms = [normalize(s["text"]) for s in segs]  # 预计算，避免 lookback 重复归一化
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


# ──────────────── SequenceMatcher 初步对齐 ────────────────────────

def sm_align(segs, sentences):
    """
    用 SequenceMatcher 对每句话找最佳 Whisper segment 窗口。
    返回 {index: {"start", "end", "score", "method"}}
    method 为 "sm"（匹配成功）或 "pending_llm"（需 LLM 兜底）
    """
    results = {}
    cursor = 0
    n = len(segs)

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
            t_s = segs[best_s]["start"]
            t_e = segs[best_e - 1]["end"]
            results[global_idx] = {"start": t_s, "end": t_e,
                                   "score": best_score, "method": "sm"}
            cursor = best_e
        else:
            results[global_idx] = {"start": None, "end": None,
                                   "score": best_score, "method": "pending_llm"}

    return results


# ──────────────── LLM 语义对齐裁判 ──────────────────────────────

def call_llm(prompt):
    """调用 LLM，带重试机制"""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
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
    """
    将一批参考句子和所有 Whisper 转录段落发送给 LLM，
    让 LLM 做语义层面的匹配判断。

    返回 {句子序号: [起始S编号, 结束S编号]} 或 {}（解析失败）
    """
    lang_cfg = LANG_CONFIG[LANGUAGE]
    segs_text = "\n".join(f"[S{i}] {s['text'].strip()}"
                          for i, s in enumerate(segs))
    sent_text = "\n".join(f"[{idx}] {text}" for idx, text in sentence_batch)

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
- 一句参考文本可能对应多个连续的转录段落（如朗读者读完后又重复了一遍）
- 找不到匹配时返回 null

只返回 JSON，格式：{{"句子编号": [起始S编号, 结束S编号], ...}}
编号为整数（对应上面的 S0, S1, S2...）。找不到则返回 null。

示例：{{"1": [0, 0], "2": [1, 3], "3": null}}"""

    response = call_llm(prompt)
    try:
        # 匹配最外层 JSON 对象（贪婪匹配最后一个 }）
        m = re.search(r'\{[\s\S]*\}', response)
        if not m:
            print(f"    [LLM 响应无 JSON] {response[:200]}")
            return {}
        result = json.loads(m.group())
        if not isinstance(result, dict):
            print(f"    [LLM 响应非 dict] 类型={type(result).__name__}")
            return {}
        return result
    except json.JSONDecodeError as e:
        print(f"    [LLM JSON 解析失败] {e}")
        return {}


def llm_align(segs, pending_sentences):
    """
    对所有 pending 句子批量用 LLM 对齐。
    返回 {global_idx: (start_time, end_time)}
    """
    max_idx = len(segs) - 1
    results = {}
    n_batches = (len(pending_sentences) + LLM_BATCH - 1) // LLM_BATCH
    print(f"    LLM 对齐: {len(pending_sentences)} 句，分 {n_batches} 批")

    for i in range(0, len(pending_sentences), LLM_BATCH):
        batch = pending_sentences[i:i + LLM_BATCH]
        raw = llm_align_batch(segs, [(idx, text) for idx, text in batch])
        ok = 0
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


# ──────────────────── 主流程 ─────────────────────────────────────

def process_file(whisper_model, audio_path, srt_path):
    """处理单个音频文件：转录 → 匹配 → LLM 兜底 → 输出 SRT"""
    print(f"\n处理: {audio_path.name}")

    srt_lines = parse_srt(srt_path)
    print(f"  参考句子: {len(srt_lines)} 条")

    # 1. faster-whisper 转录
    print("  [步骤 1] faster-whisper 转录...")
    segs_raw = transcribe(whisper_model, audio_path)
    segs = filter_hallucinations(segs_raw)
    print(f"  有效转录段: {len(segs)} 段")

    if not segs:
        print("  [警告] 转录结果为空，跳过此文件")
        return []

    # 2. SequenceMatcher 初筛
    print("  [步骤 2] SequenceMatcher 初筛...")
    sm_results = sm_align(segs, srt_lines)
    sm_ok = sum(1 for r in sm_results.values() if r["method"] == "sm")
    pending_count = len(srt_lines) - sm_ok
    print(f"  初筛结果: {sm_ok}/{len(srt_lines)} 句匹配成功，"
          f"{pending_count} 句待 LLM 兜底")

    # 3. LLM 语义对齐（仅处理初筛失败的句子）
    llm_results = {}
    if pending_count > 0:
        print("  [步骤 3] LLM 语义对齐...")
        pending = [(idx, text) for idx, text in enumerate(srt_lines)
                   if sm_results[idx]["method"] == "pending_llm"]
        llm_results = llm_align(segs, pending)
    else:
        print("  [步骤 3] 跳过（无需 LLM 兜底）")

    # 4. 合并结果
    output_segments = []
    for global_idx, text in enumerate(srt_lines):
        sm_r = sm_results[global_idx]
        if sm_r["method"] == "sm":
            output_segments.append({
                "start": sm_r["start"],
                "end": sm_r["end"],
                "text": text,
            })
        elif global_idx in llm_results:
            ts, te = llm_results[global_idx]
            if ts is not None:
                output_segments.append({
                    "start": ts,
                    "end": te,
                    "text": text,
                })
            else:
                print(f"  [未对齐] 第 {global_idx + 1} 句")
        else:
            print(f"  [未对齐] 第 {global_idx + 1} 句")

    aligned = len(output_segments)
    print(f"  最终对齐: {aligned}/{len(srt_lines)} 句")
    return output_segments


def main():
    # 查找音频文件
    audio_files = sorted(
        f for ext in ("*.m4a", "*.mp3", "*.wav", "*.flac")
        for f in AUDIO_DIR.glob(ext)
    )
    if not audio_files:
        print("未找到音频文件！")
        return

    print(f"找到 {len(audio_files)} 个音频文件")
    lang_cfg = LANG_CONFIG[LANGUAGE]
    print(f"语言: {LANGUAGE} ({lang_cfg['name']})")
    print(f"LLM: {LLM_MODEL}（{LLM_BASE_URL}）")
    print()

    # 加载 Whisper
    print("加载 faster-whisper 模型...")
    whisper_model = load_whisper()
    print()

    # 逐文件处理
    total_aligned = 0
    total_sentences = 0

    for audio_path in audio_files:
        stem = audio_path.stem
        srt_path = AUDIO_DIR / f"{stem}.asr.qc.srt"
        if not srt_path.exists():
            print(f"  跳过（找不到对应 SRT）: {audio_path.name}")
            continue

        try:
            n_ref = len(parse_srt(srt_path))
            segments = process_file(whisper_model, audio_path, srt_path)
            if segments:
                out_path = OUTPUT_DIR / f"{stem}.aligned.srt"
                write_srt(segments, out_path)
                print(f"  已保存: {out_path.name}")
                total_aligned += len(segments)
                total_sentences += n_ref
        except Exception as e:
            import traceback
            print(f"  失败: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"全部完成！方案 B 结果在: {OUTPUT_DIR}")
    print(f"总计: {total_aligned}/{total_sentences} 句对齐")
    print(f"\n提示: 方案 B 适合口误较多的录音。")
    print(f"      如录音与参考文稿高度一致，建议优先使用方案 A（align_srt_routeA.py）。")


if __name__ == "__main__":
    main()
