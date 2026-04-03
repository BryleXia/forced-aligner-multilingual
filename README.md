# 文本优先强制对齐：纪录片语音的零ASR字幕时间戳生成

> **Text-First Forced Alignment for Documentary Speech: Zero-ASR Subtitle Timestamping**

**作者：** BryleXia &nbsp;·&nbsp; 北京第二外国语学院欧洲学院  
**测试集规模：** 390句 / 5段 / ~56分钟  
**最终结果：** 人工感知审听错误率 **0.0%**  
**核心脚本：** `align_srt_routeA.py`  
**里程碑标签：** `v13-milestone`

---

## 摘要

本项目针对"SRT文本可信、时间戳需要重建"这一翻译字幕场景，提出了一种**文本优先的CTC强制对齐（Text-First CTC Forced Alignment）**方法。核心思路是：直接将已经过人工质检的SRT字幕文本构造为对齐输入，完全绕过Whisper ASR转录步骤，交由wav2vec2 CTC对齐器生成词级时间戳，再通过全局LCS动态规划映射为行级时间戳。

该方法从根本上消除了标准流程中ASR转录误差（尤其是中文音译专名的系统性误转写）对DP匹配的干扰。在此基础上，我们设计了一种**三条件词速守门员（Word-Rate Outlier Guardian）**，精准检测并修正CTC对齐器在长静音段（场景转场）产生的早开始伪影。

在5段西班牙语纪录片旁白音频（共390条字幕行，约56分钟）构成的测试集上，本方法经人工逐句感知审听，取得了**100%对齐准确率（0条错误）**，优于基线方法（Whisper ASR + LCS + 模糊边界扩展，错误率3.8%），且无需安装MFA、无需编写任何发音词典。

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [基线方案与残余问题](#2-基线方案与残余问题)
3. [核心方法：文本优先CTC对齐](#3-核心方法文本优先ctc对齐)
4. [CTC早开始伪影与词速守门员](#4-ctc早开始伪影与词速守门员)
5. [实验设置](#5-实验设置)
6. [实验结果](#6-实验结果)
7. [与现有方法对比](#7-与现有方法对比)
8. [快速开始](#8-快速开始)
9. [关键参数说明](#9-关键参数说明)
10. [局限性与未来工作](#10-局限性与未来工作)
11. [参考文献](#11-参考文献)

---

## 1. 背景与问题定义

### 任务

本项目的输入是：

- **音频文件**（m4a/wav格式，西班牙语纪录片旁白，含同期声采访）
- **已质检的SRT字幕文本**（人工翻译并校对，文本内容完全可信，但原始时间戳不可用）

目标是为每一条字幕行生成精确的起止时间戳，输出标准SRT文件，供后续语料库建设和AI同声传译评测使用。

这一任务本质上是**有参考文本的强制对齐（Forced Alignment with Reference Transcript）**。与自动语音识别（ASR）不同，我们不需要从音频中"猜测"文本内容——文本已知且可信，问题只是：每句话在音频的哪个时间区间被说出？

### 核心挑战：中文音译专名的系统性误转写

本项目语料来自中国纪录片的西班牙语配音版。字幕中包含大量源于中文的音译专名，例如：

| 中文原名 | SRT正确拼写 | Whisper ASR 典型误转写 |
|---------|------------|----------------------|
| 老包 | Lao Bao | "Alien"、"Lab"、"Labao" |
| 诺邓 | Nuodeng | "Nudon"、"Nuodong"、"Nodeng" |
| 生五 | Shengwu | "Chengwu"、"Shengoo" |
| 卓玛 | Zhuoma | "Chroma"、"Juoma" |
| 阿亮 | A Liang | "Alien"（编辑距离极大） |

这些专名不存在于任何西班牙语词汇表，是典型的**词汇表外词（Out-of-Vocabulary, OOV）**。Whisper大模型在处理这类词时，倾向于将其替换为发音相近的西班牙语词或随机输出，导致ASR转录文本与SRT真实文本之间出现无法通过模糊匹配弥合的鸿沟。

---

## 2. 基线方案与残余问题

### 基线方案（8号实验）

基线方案采用标准的"ASR → 对齐 → 匹配"三阶段流程：

```
音频
 │
 ▼
[Whisper large-v3 ASR 转录]
 │  → 含噪声的词级转录文本（含专名误转写）
 ▼
[whisperx.align() — wav2vec2 CTC对齐]
 │  → 词级时间戳
 ▼
[全局LCS动态规划对齐]
 │  → SRT词序列 ↔ ASR词序列 的最长公共子序列匹配
 ▼
[boundary_extend() — 模糊边界扩展]
 │  → 对LCS未命中的边界词，在ASR窗口内进行模糊搜索（阈值0.45）
 ▼
输出对齐SRT
```

针对专名问题，基线方案进行了三项专项修复：
1. 只取`prefix_words[0]`进行边界匹配（避免ASR合并词干扰）
2. 对"Laopao"类ASR合并词，将SRT前两词拼接后再匹配
3. 模糊匹配阈值从0.40提升至0.45

### 基线残余问题

尽管经过针对性修复，基线方案在人工感知审听中仍存在以下无法自修复的问题：

- **"A Liang"** → ASR输出"Alien"，编辑距离/最大长度 = 5/5 = 1.0，远超0.45阈值，完全无法匹配
- **部分"Lao Bao"** 出现位置，ASR输出形态随机，无规律可循

**基线审听结果（237句详细审听）：9处可感知错误，错误率3.8%**

| 片段 | 句数 | 错误数 | 错误描述 |
|------|------|--------|---------|
| seg001 | 52 | 2 | 03:06 yuanes早结束1-2秒；06:59 句首"Un"晚进半秒 |
| seg002 | 79 | 5 | 06:31早结束近2秒；07:55早结束1秒；08:11晚进半秒；09:54/12:12 nuodong早结束 |
| seg003 | 106 | 2 | 01:46 Shengwu晚进半秒；12:18 hielo早结束半秒 |
| seg004 | 81 | 0 | — |

---

## 3. 核心方法：文本优先CTC对齐

### 关键洞见

CTC强制对齐器（如wav2vec2）的工作原理，是将**已知文本**中的每个token分配到音频的某个时间帧。它从设计上就接受"用户提供的参考文本"——这正是"强制"对齐得名的原因。

标准的whisperx流程先运行ASR生成转录文本，再将转录文本交给CTC对齐器。但既然我们已经拥有经过质检的SRT文本，**ASR转录步骤完全多余**——它不仅耗时，还会引入专名误转写噪声。

### 流程对比

```
标准流程：  音频 → [Whisper ASR] → 含噪声文本 → [CTC对齐] → 时间戳
                                         ↑
                                    OOV专名误转写发生于此

本方法：    音频 ────────────────────────────→ [CTC对齐] ← SRT文本 → 时间戳
```

### 实现

核心代码极为简洁——将SRT所有行拼接为一个大segment，起止时间覆盖整段音频：

```python
def build_segments_from_srt(srt_lines, audio_duration):
    """将SRT文本构造为whisperx.align()所需的segments格式"""
    full_text = " ".join(srt_lines)
    return [{"text": full_text, "start": 0.0, "end": audio_duration}]
```

随后直接调用：

```python
aligned = whisperx.align(
    segments,        # 由SRT文本构造，非ASR输出
    align_model,     # wav2vec2西语模型
    metadata,
    audio,
    device="cuda",
    return_char_alignments=False,
)
```

wav2vec2 CTC对齐器获得**正确拼写的参考文本**后，直接在音频帧上寻找每个词的边界，无需任何专名词典，无需模糊匹配。

### 词级→行级映射：全局LCS动态规划

对齐器输出词级时间戳后，通过与基线相同的全局LCS动态规划，将词级时间戳映射回SRT行级时间戳：

```python
# 归一化处理：小写、去声调、数字转西语读法、连字符拆分
srt_seq = [(normalize(word), line_idx) for line_idx, line in enumerate(srt_lines)
           for word in normalize_for_dp(line)]
asr_seq = [(normalize(word["word"]), word_idx) for word_idx, word in enumerate(words)]

# LCS全局对齐（无间隔惩罚，只计精确匹配）
pairs = dp_align(srt_seq, asr_seq)

# 每行取首尾匹配词的时间戳
for line_idx, line in enumerate(srt_lines):
    start = words[line_first[line_idx]]["start"]
    end   = words[line_last[line_idx]]["end"]
```

由于输入文本就是SRT本身（拼写完全正确），LCS的精确命中率接近100%，大幅优于基线方案的60-80%命中率（基线受ASR误转写影响）。

---

## 4. CTC早开始伪影与词速守门员

### 问题描述

文本优先对齐在实践中引入了一种新型伪影：**CTC早开始（Early-Start Artifact）**。

CTC对齐器在处理整段音频时，必须将输入文本中的每一个token都分配到某个时间帧，不允许"跳过"。当音频中存在场景转场带来的长静音段时，对齐器无法自动识别静音并跳过——它只能将下一句的首词紧贴在上一句末词之后50ms处（由后处理钳制代码保证的最小间隔），再将该句的词均匀拉伸到行末。

结果：某些句子的计算起点比实际语音起点早**数秒至十余秒**。

```
实际音频：  [句14音频]  ←16秒静音（场景转场）→  [句15音频]
CTC输出：  [句14时间戳]─50ms─[句15时间戳: 从14结束后50ms开始，词被拉伸16秒]
                               ↑
                       早开始约16秒
```

### 诊断特征

这种伪影具有三个可量化的诊断特征，**同时满足**时可高置信度判定为异常：

| 特征 | 阈值 | 说明 |
|------|------|------|
| 词速（词数/时长） | < 1.3 词/秒 | 正常旁白语速2-4词/秒；被拉伸的句子降至0.5词/秒以下 |
| 行时长 | ≥ 6.0 秒 | 短行豁免（短行也可能说得慢） |
| 与前行间隔 | < 0.3 秒 | 钳制代码签名：自然停顿>300ms；50ms间隔说明对齐器被迫紧贴前行 |

第三个条件是关键创新：如果与前行之间存在自然间隔（>0.3秒），则对齐器找到了真实的语音起点，无需修正；只有间隔极小（钳制代码留下的50ms），才说明起点是"被迫"放置的。

### 三条件守门员（Word-Rate Outlier Guardian）

```python
def snap_outlier_starts(segments, audio, vad_model=None, sr=16000,
                        min_move_s=1.5,
                        min_words_per_sec=1.3,
                        min_suspect_duration=6.0):
    for i, seg in enumerate(segments):
        duration   = seg["end"] - seg["start"]
        word_count = len(normalize_for_dp(seg["text"]))
        gap_to_prev = seg["start"] - segments[i-1]["end"] if i > 0 else 999.0

        is_suspect = (
            word_count >= 3
            and duration >= min_suspect_duration          # 时长≥6秒
            and word_count / duration < min_words_per_sec # 词速<1.3词/秒
            and gap_to_prev < 0.3                         # 与前行间隔<300ms
        )

        if is_suspect:
            # 在 [orig_start + 1.5s, orig_end] 内搜索第一个VAD语音起点
            search_from = seg["start"] + min_move_s
            new_start = find_speech_onset(audio, search_from, seg["end"], vad_model)
            seg["start"] = new_start
```

**搜索策略（双层保险）：**
- 主路径：Silero-VAD \[1\] 全音频一次性预计算语音时间段，精确识别背景音/环境音中的语音起点
- 备用路径：20ms帧级RMS能量阈值（70百分位参考电平的15%），无额外依赖

### 修正案例

| 位置 | 伪影起点（CTC） | 修正后起点 | 修正量 |
|------|--------------|-----------|--------|
| seg003 第16行 "¡Mira, aquí hay otro!" | 02:00.803 | **02:17.100** | +16.3秒 |
| seg003 第37行 "Se sumerge la raíz de loto…" | 04:44.486 | **04:51.800** | +7.3秒 |
| seg005 第6行 "En invierno, la embarcación…" | 00:39.140 | **00:45.400** | +6.3秒 |

全部5段共触发**3次**修正，其余387行未触发（守门员精度：0误触发）。

---

## 5. 实验设置

### 数据集

| 属性 | 详情 |
|------|------|
| 来源 | 中国自然人文纪录片（西班牙语配音版） |
| 语言 | 西班牙语（es） |
| 音频格式 | M4A，单声道，16kHz |
| 片段数 | 5段（seg001–seg005） |
| 字幕行数 | 390条（52 + 79 + 106 + 81 + 72） |
| 总时长 | 约56分钟 |
| 语体 | 正式旁白叙述 + 口语化采访对话（混合） |
| 核心难点 | 中文音译专名（诺邓、老包、生五、卓玛、阿亮等），在西班牙语中均为OOV词 |

### 模型与硬件

| 组件 | 版本/规格 |
|------|---------|
| CTC对齐模型 | wav2vec2-large-xlsr（西语fine-tune版，via whisperx） |
| VAD模型 | Silero-VAD |
| 计算设备 | NVIDIA GPU，CUDA |
| Python | ≥ 3.10 |

### 评测方法

采用**人工感知审听（Human Perceptual Evaluation）**。评测人员对每条字幕行进行二元判断：时间戳是否与实际语音在感知上对齐（"正确"/"有可感知偏差"）。

此方法的优势在于：它直接对应字幕的实际使用场景（视频字幕显示），而非实验室条件下的毫秒级帧偏差统计。评测标准约为：±200ms以内的偏差通常不可感知；超过500ms则明显可感知。

---

## 6. 实验结果

### 主结果

| 方案 | 审听片段 | 审听句数 | 感知错误率 |
|------|---------|---------|-----------|
| 基线（8号：Whisper ASR + LCS + 模糊边界扩展） | 3/5 | 237 | **3.8%**（9条错误） |
| 路线A v1（10号：文本优先，无守门员） | 5/5 | 390 | **~0.8%**（3条早开始伪影） |
| **路线A v13（文本优先 + 三条件守门员）** | **5/5** | **390** | **0.0%（0条错误）** |

### 基线无法修复的问题在路线A中的自然消解

基线方案存在两类结构性问题，无论如何调整参数都无法修复：

1. **"A Liang" → ASR输出"Alien"**：编辑距离5，最大长度5，归一化距离1.0，完全超出任何合理的模糊匹配阈值
2. **部分"Lao Bao"**：ASR输出形态高度不稳定，无规律可预测

路线A由于从不运行ASR，这两类问题**从根本上不再存在**——wav2vec2 CTC对齐器直接面对正确拼写的文本，不需要任何形式的文本匹配。

---

## 7. 与现有方法对比

### 方法对比矩阵

| 方法 | 需要自定义发音词典 | 需要ASR | OOV专名处理 | 单段时长限制 | 适用条件 |
|------|:---:|:---:|------|:---:|------|
| MFA \[2\] | **是**（~30分钟/语种） | 否 | 通过手写词典条目 | 无 | 学术研究，音素精度要求高 |
| WhisperX 标准流程 \[3\] | 否 | **是** | 差（ASR误转写） | 无 | 通用场景，无特殊专名 |
| NeMo Forced Aligner \[4\] | 否 | 可选 | 中（依赖模型词汇表） | 无 | 英语场景较优 |
| Qwen3-ForcedAligner-0.6B \[5\] | 否 | 否 | 好（LLM） | **300秒** | 短音频，LLM推理可用 |
| **本方法（路线A v13）** | **否** | **否** | **透明（直接使用SRT文本）** | **无** | SRT文本已质检，有GPU |

### 精度参考数据

Rousso et al.（Interspeech 2024）\[6\] 在TIMIT语料库上对主流强制对齐方法进行了系统比较：

| 方法 | 词级精度（10ms容差） | 词级精度（20ms容差） |
|------|:---:|:---:|
| MFA | 41.6% | 72.8% |
| WhisperX | 22.4% | 52.7% |

> **注**：上述数字来自标准语音语料库（TIMIT，美式英语朗读语音），与本项目场景（西班牙语纪录片旁白，含OOV专名）不可直接比较。本项目采用人工感知审听而非帧级标注，因此不报告毫秒级精度数字。

---

## 8. 快速开始

### 环境依赖

```bash
pip install whisperx silero-vad num2words imageio-ffmpeg
```

> **注**：`whisperx` 将自动安装 `torch`、`faster-whisper`、`transformers` 等依赖。

### 准备输入文件

将以下文件放置于同一目录（默认 `/root/`）：

```
/root/
├── seg001.m4a              # 音频文件（支持m4a/wav/mp3）
├── seg001.asr.qc.srt       # 对应的质检后SRT（文件名stem需与音频一致）
├── seg002.m4a
├── seg002.asr.qc.srt
└── ...
```

### 运行

```bash
python align_srt_routeA.py
```

输出文件保存至 `/root/aligned_routeA/`，格式为 `*.aligned.srt`。

### 修改路径配置

脚本顶部修改以下常量：

```python
AUDIO_DIR  = Path("/root")              # 音频和SRT输入目录
OUTPUT_DIR = Path("/root/aligned_routeA")  # 输出目录
DEVICE     = "cuda"                     # "cuda" 或 "cpu"
LANGUAGE   = "es"                       # ISO 639-1 语言代码
```

---

## 9. 关键参数说明

### 词速守门员参数（`snap_outlier_starts`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `min_words_per_sec` | 1.3 | 词速低于此值触发嫌疑（正常旁白语速2-4词/秒） |
| `min_suspect_duration` | 6.0秒 | 行时长不足此值不触发（短行豁免） |
| `min_move_s` | 1.5秒 | 搜索语音起点的最小偏移量（跳过轻微停顿） |
| `gap_to_prev` 阈值 | 0.3秒 | 与前行间隔超过此值则认为起点可信，不修正 |
| RMS `threshold_ratio` | 0.15 | VAD不可用时，RMS能量参考电平的15%作为有声门槛 |

### 分块参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `MAX_CHUNK_DURATION` | 1800.0秒 | 单段音频超过此时长时自动按字数比例分块对齐 |

> 对于绝大多数纪录片片段（≤30分钟），使用整段对齐模式，精度最优。

---

## 10. 局限性与未来工作

### 当前局限

1. **依赖SRT文本质量**：方法的核心假设是SRT文本完全正确。若存在错字、漏句或额外插入句，CTC对齐器无法自我纠正，将产生漂移。
2. **长音频分块边界伪影**：超过30分钟的音频需要分块处理，分块边界处可能引入轻微误差。
3. **评测为感知测试**：本项目未使用帧级标注的ground truth，因此无法报告毫秒级精度数字，亦无法与Rousso et al.等基准直接比较。
4. **单语言验证**：目前仅在西班牙语上完整验证。方法理论上适用于任何whisperx支持的语言，但需在各语种上独立验证。

### 未来工作

- **路线D（Qwen3-ForcedAligner-0.6B）** \[5\]：阿里QwenLM团队2026年1月发布的LLM强制对齐器，基于Qwen3-Omni后训练，支持11种语言。其LLM架构可能对OOV专名具有更强的内在鲁棒性。限制：单段最长300秒，需切片处理。建议以seg001为基准做对比实验。
- **帧级标注评测**：引入字幕对齐标注工具（如 `aeneas`、ELAN），建立毫秒级精度的评测基准。
- **多语种迁移**：将本方法迁移至俄语和法语，验证对相应OOV专名场景的泛化能力。

---

## 11. 参考文献

```bibtex
@misc{silero-vad,
  author       = {{Silero Team}},
  title        = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector},
  year         = {2021},
  howpublished = {\url{https://github.com/snakers4/silero-vad}}
}

@inproceedings{mcauliffe17_interspeech,
  author    = {McAuliffe, Michael and Socolof, Michaela and Mihuc, Sarah
               and Wagner, Michael and Sonderegger, Morgan},
  title     = {{Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi}},
  booktitle = {Proc. Interspeech 2017},
  pages     = {498--502},
  year      = {2017},
  doi       = {10.21437/Interspeech.2017-1386}
}

@inproceedings{bain23_interspeech,
  author    = {Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
  title     = {{WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}},
  booktitle = {Proc. Interspeech 2023},
  pages     = {548--554},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-920}
}

@inproceedings{rastorgueva23_interspeech,
  author    = {Rastorgueva, Elena and Lavrukhin, Vitaly and Ginsburg, Boris},
  title     = {{NeMo Forced Aligner and its application to word alignment
                for subtitle generation}},
  booktitle = {Proc. Interspeech 2023},
  pages     = {5257--5258},
  year      = {2023},
  url       = {https://www.isca-archive.org/interspeech_2023/rastorgueva23_interspeech.html}
}

@misc{qwen3-forcedaligner,
  author       = {{Qwen Team, Alibaba Cloud}},
  title        = {Qwen3-ForcedAligner-0.6B},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B}},
  note         = {arXiv:2601.21337}
}

@inproceedings{rousso24_interspeech,
  author    = {Rousso, Rotem and Cohen, Eyal and Keshet, Joseph and Chodroff, Eleanor},
  title     = {{Tradition or Innovation: A Comparison of Modern ASR Methods
                for Forced Alignment}},
  booktitle = {Proc. Interspeech 2024},
  pages     = {1525--1529},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.19363}
}

@inproceedings{baevski2020wav2vec,
  author    = {Alexei Baevski and Yuhao Zhou and Abdelrahman Mohamed and Michael Auli},
  title     = {wav2vec 2.0: A Framework for Self-Supervised Learning
               of Speech Representations},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.11477}
}

@inproceedings{graves2006ctc,
  author    = {Graves, Alex and Fern\'{a}ndez, Santiago and Gomez, Faustino
               and Schmidhuber, J\"{u}rgen},
  title     = {Connectionist Temporal Classification: Labelling Unsegmented
               Sequence Data with Recurrent Neural Networks},
  booktitle = {Proc. ICML 2006},
  pages     = {369--376},
  year      = {2006},
  doi       = {10.1145/1143844.1143891}
}

@software{faster-whisper,
  author       = {{SYSTRAN}},
  title        = {faster-whisper},
  year         = {2023},
  url          = {https://github.com/SYSTRAN/faster-whisper}
}
```

---

*本项目由 BryleXia 开发，北京第二外国语学院欧洲学院多语种平行语料库建设项目。*
