# 字幕强制对齐工具（多语言版）

> 把字幕文件的时间戳准确对齐到音频上。提供三套互补方案，覆盖不同录音质量场景。
> 支持 **西班牙语、法语、俄语**，切换语言只需一个参数。
> 方案 A 在 5 段纪录片音频（390 条字幕，约 56 分钟）上经过人工逐句审听，**0 条错误**；在 5 段俄语纪录片（1028 词）上自动匹配率达 **98.8%**。

**作者：** BryleXia · 北京第二外国语学院欧洲学院
**核心脚本：** `align_srt_routeA.py`（方案 A）、`align_srt_routeA_multi.py`（方案 A 并行版）、`align_srt_routeB_llm.py`（方案 B）、`align_srt_routeC_hybrid.py`（方案 C）
**里程碑标签：** `v13-milestone`

---

## 三方案决策树

```
                    ┌──────────────────────────────────┐
                    │      输入：音频 + 字幕文字        │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │  录音与参考文稿是否高度一致？      │
                    └────┬─────────────────────┬───────┘
                         │ 是                   │ 否（有口误/多读/漏读）
                         ▼                      ▼
              ┌─────────────────┐    ┌──────────────────────┐
              │  方案 A（主力）  │    │  方案 B（兜底）       │
              │  参考文本强制对齐 │    │  ASR + LLM 语义对齐  │
              │  align_srt_      │    │  align_srt_           │
              │  routeA.py       │    │  routeB_llm.py        │
              │                  │    │                       │
              │  wav2vec2 CTC    │    │  faster-whisper 转录  │
              │  完全跳过 ASR    │    │  + SequenceMatcher   │
              │  专名问题=不存在  │    │  + LLM 语义裁判      │
              └────────┬────────┘    └───────────┬───────────┘
                       │                         │
                       │  ┌──────────────────────┘
                       │  │ 两者都有用但不完美？
                       │  ▼
              ┌────────────────────────────────┐
              │  方案 C（混合）                 │
              │  align_srt_routeC_hybrid.py     │
              │                                 │
              │  B 打锚点 → A 填 gap → 合并    │
              └──────────┬─────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────┐
              │     输出：带时间戳的 SRT 字幕     │
              └──────────────────────────────────┘
```

**方案 A**（主力）：直接把字幕文字送进 CTC 声学对齐器（wav2vec2），跳过语音识别。适用于录音与参考文稿高度一致的场景。

**方案 B**（兜底）：先用语音识别（faster-whisper）获取"录音里实际说了什么"，再用 LLM（Qwen3.6-plus）做语义匹配。适用于口误、多读、漏读等音频与文稿不一致的场景。

**方案 C**（混合）：B 打锚点 + A 填 gap。先用 B 尽可能多地获取句子时间戳作为"锚点"，再用 A 的 CTC 对齐器在锚点之间的空白窗口内精确对齐。适用于 B 无法全覆盖、但 A 也因文稿偏差无法独立工作的中间场景。

---

## 方案 A：参考文本强制对齐

### 背景

常规做法是：先用 AI 把音频转录成文字，再把转录结果和字幕文字做匹配。问题出在第一步——当字幕里含有 AI 从没见过的词（比如中文人名地名的西语音译），转录就会出错，后面全乱：

| 正确写法 | AI 语音识别（Whisper）的输出 |
|---------|--------------------------|
| Lao Bao（老包） | "Alien"、"Lab"、"Labao" |
| Nuodeng（诺邓） | "Nudon"、"Nodeng"、"Nuodong" |
| A Liang（阿亮） | "Alien"（完全无法识别） |

### 核心思路

既然字幕文字本来就是对的，为什么还要先转录再匹配？直接跳过转录，把字幕文字送进对齐器：

```
常规流程：  音频 → [Whisper 转录] → 含错误的文字 → [对齐] → 时间戳
                                          ↑ 专名在此出问题

方案 A：    音频 ─────────────────────────────→ [对齐] ← 字幕文字 → 时间戳
```

核心代码非常简洁：

```python
full_text = " ".join(srt_lines)
segments  = [{"text": full_text, "start": 0.0, "end": audio_duration}]
aligned   = whisperx.align(segments, align_model, metadata, audio, device)
```

wav2vec2 返回词级时间戳，再用 LCS 动态规划映射回每一行字幕。

### 词速异常守门员：修复时间戳提前跑的问题

在场景切换（长时间静音）的地方，CTC 对齐器有时会把下一句字幕的起点放得过早。同时满足以下三个条件时触发修正：

1. **词速低于 1.5 词/秒**（正常语速 2~4 词/秒）
2. **时长超过 6 秒**
3. **与前一句间隔小于 0.3 秒**（被强行塞进去的标志）

修正方式：用 Silero-VAD 找到真正的语音起点。

| 位置 | 修正前 | 修正后 | 提前了 |
|------|--------|--------|--------|
| 第3段第16句 | 02:00 | **02:17** | 17 秒 |
| 第3段第37句 | 04:44 | **04:51** | 7 秒 |
| 第5段第6句 | 00:39 | **00:45** | 6 秒 |

触发 3 次，修正 3 次，0 次误判。

### 未对齐行智能回退

对于极少数无法对齐的句子（通常 1% 以内），方案 A 不会给固定的 0.5 秒占位符，而是查找前后已对齐行的时间范围，按文字长度比例分配时间段。

### 优化项（v13 起）

- **BOM 兼容**：SRT 文件读取使用 `utf-8-sig`，自动去除 Windows BOM 头
- **命令行参数**：支持 `--lang`、`--audio-dir`、`--output-dir`，不用改代码
- **DP 内存优化**：LCS 动态规划用 `bytearray` 存方向矩阵，内存减少约 96%
- **VAD 搜索加速**：用二分查找（`bisect`）替代线性遍历

### 多进程并行版（align_srt_routeA_multi.py）

生产环境中每次任务通常包含 5+ 对音频-SRT 文件。原脚本串行处理，GPU 利用率极低（wav2vec2 仅 ~1.2GB，RTX 5090 的 32GB 显存大量闲置）。

并行版每个子进程独立加载模型，多个文件同时跑：

```
进程1: [加载 wav2vec2] → 处理 seg001
进程2: [加载 wav2vec2] → 处理 seg002
进程3: [加载 wav2vec2] → 处理 seg003
...                               （5 × 1.2GB ≈ 6GB，显存绰绰有余）
```

- 5 个文件并行 → 总耗时接近单个最慢文件的耗时，加速比 ~4-5x
- 自动匹配两种 SRT 命名格式：`*_tgt.asr.qc.srt`（生产格式）和 `*.asr.qc.srt`（原格式）
- 核心逻辑全部复用 `align_srt_routeA.py`，零重复代码

---

## 方案 B：ASR + LLM 语义对齐兜底

### 适用场景

当朗读者出现以下情况时，方案 A 无法正确工作——因为对齐器拿着"错误的地图"（与实际音频不一致的参考文稿）去"找路"，必然偏移：

- **口误**：读错了某个词
- **多读**：把某句话重复读了一遍
- **漏读**：跳过了某句话或某个短语
- **改词**：临时换了一种说法

方案 B 的思路：**先"听"清楚录音里实际说了什么，再用 LLM 做语义判断，把实际内容与参考文稿对应起来。**

### 流程

```
音频
  │
  ▼
[第一步] faster-whisper 转录
  │ 输出：录音实际内容的逐段时间戳
  │ 防幻觉措施：condition_on_previous_text=False + 重复段过滤
  ▼
[第二步] SequenceMatcher 字符相似度初筛
  │ 相似度 ≥ 0.40 的句子直接匹配成功
  │ 剩余句子交给 LLM
  ▼
[第三步] LLM 语义裁判（Qwen3.6-plus）
  │ 将转录段落 + 参考句子一起发给 LLM
  │ LLM 理解口误/重复/近义替换等语义关系
  │ 输出：每句参考文稿对应哪个转录段落
  ▼
最终 SRT 字幕
```

### LLM 语义裁判

LLM 的核心能力在于**语义理解**——它知道"朗读者把 X 读成了 Y"不意味着两句不相关，而是同一内容的不同表达。这是字符匹配算法（SequenceMatcher、Levenshtein）做不到的。

**技术来源：** 此方案在北京博物馆语料库项目中经过验证（日语 212 句，97.6% 对齐率），那里 LLM 解决了汉字↔假名跨书写系统的匹配问题。迁移到西语后，它解决的是口误导致的文本不一致问题。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SM_HIGH_CONF` | 0.40 | SequenceMatcher 置信度阈值 |
| `LLM_BATCH` | 8 | LLM 每批处理句子数 |
| `HALL_SIM` | 0.70 | 幻觉段过滤相似度阈值 |

---

## 方案 C：B 打锚点 + A 填空白（混合）

### 思路

方案 B 不需要 100% 覆盖——只要一部分句子能成功对齐，就可以作为"锚点"。锚点之间的空白窗口交给方案 A 的 CTC 对齐器精对齐，结合了两者的优点：

```
音频 → B 转录+SM初筛+LLM裁判 → 锚点（已对齐句子）
                                   │
         锚点句子 ──────────────────┘ 直接用 B 的时间戳
         空白窗口 ─────────────────→ CTC 在窄窗口内精确对齐
                                   │
                              合并 → 守门员后处理 → SRT
```

### 工作流程

1. **faster-whisper 转录** + **SequenceMatcher 初筛** + **LLM 兜底** → 获取尽可能多的锚点
2. 按锚点将句子切块：
   - **锚点块**：单句锚点，直接用 B 的时间戳，**跳过 CTC**
   - **gap 块**：两锚点之间的空白，用 CTC 在窗口内精对齐
   - **B 全失败**：整块覆盖全音频，退化为纯方案 A
3. 合并所有块 → 词速异常守门员 → 输出 SRT

### 适用场景

- B 覆盖率中等（50%~80%），有足够锚点但不够全覆盖
- A 无法独立工作（文稿与音频偏差较大），但 B 也无法处理所有句子
- 需要词级精度（方案 A 的 CTC 输出）

---

## 多语言支持

三个方案均已支持多语言。方案 A 支持命令行参数一键切换；方案 B/C 修改脚本顶部的 `LANGUAGE` 变量即可：

| 语言代码 | 语言 | 方案 A/C 对齐模型 | 方案 B ASR 模型 |
|---------|------|-----------------|---------------|
| `es` | 西班牙语 | WhisperX 默认 wav2vec2 | faster-whisper large-v3 |
| `fr` | 法语 | WhisperX 默认 wav2vec2 | faster-whisper large-v3 |
| `ru` | 俄语 | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | faster-whisper large-v3 |

**俄语验证：** 在 5 段俄语纪录片音频（约 56 分钟，1028 词）上测试，方案 A 自动匹配率 **98.8%**（1016/1028）。CTC 模型 `wav2vec2-large-xlsr-53-russian` 对俄语有效，非模型限制。

**输入格式要求：** 字幕需为 UTF-8 编码的 SRT 格式，法语和俄语的 Unicode 变音符号会被自动归一化处理。

---

## 实验结果

### 方案 A（参考文本强制对齐）

| 语言 | 测试句数/词数 | 出错句数/词数 | 错误率 |
|------|-------------|-------------|--------|
| 西班牙语（旧：Whisper 转录 + 模糊匹配） | 237 句 | 9 | **3.8%** |
| 西班牙语（方案 A 无守门员） | 390 句 | 3 | 0.8% |
| **西班牙语（方案 A v13 含守门员）** | **390 句** | **0** | **0%** |
| **俄语（方案 A v13）** | **1028 词** | **12** | **1.2%** |

### 方案 B（ASR + LLM 语义对齐）

在北京博物馆语料库项目（日语 212 句）中验证：

| 方法 | 对齐句数 | 成功率 |
|------|---------|--------|
| 仅 SequenceMatcher | ~148 / 212 | ~70% |
| **SequenceMatcher + LLM 兜底** | **207 / 212** | **97.6%** |

失败的 5 句是 Whisper 将相邻句子合并转录、边界本身无法确定的极端情况。

---

## 与其他工具的对比

| 工具 | 需要发音词典 | 需要 ASR | 口误处理 | 单段时长限制 |
|------|:----------:|:-------:|:-------:|:----------:|
| MFA（Montreal Forced Aligner） | 是 | 否 | 需手动 | 无 |
| WhisperX 标准流程 | 否 | **是** | 差 | 无 |
| Qwen3-ForcedAligner-0.6B（2026） | 否 | 否 | 未知 | **300 秒** |
| **本方案 A** | **否** | **否** | 不适用 | **无** |
| **本方案 B** | **否** | **是** | **LLM 语义裁判** | 无 |
| **本方案 C** | **否** | **是+否** | **锚点+CTC 混合** | 无 |

---

## 使用指南

### 1. 安装依赖

```bash
# 方案 A 依赖
pip install whisperx silero-vad num2words imageio-ffmpeg

# 方案 B/C 额外依赖
pip install faster-whisper openai
```

### 2. 准备文件

三套方案共享相同的输入格式，支持 `.m4a`、`.mp3`、`.wav`、`.flac` 音频格式。支持两种 SRT 命名：

```
格式一（原格式）:
/root/
├── seg001.m4a
├── seg001.asr.qc.srt
├── seg002.mp3
├── seg002.asr.qc.srt
└── ...

格式二（生产格式）:
/root/
├── es_tour_serv_0004_seg001.m4a
├── es_tour_serv_0004_seg001_tgt.asr.qc.srt
├── es_tour_serv_0004_seg002.m4a
├── es_tour_serv_0004_seg002_tgt.asr.qc.srt
└── ...
```

并行版（`routeA_multi`）自动识别两种格式；原脚本（`routeA`）仅支持格式一。

### 3. 运行

**方案 A（推荐首选）：**

```bash
# 默认俄语
python align_srt_routeA.py

# 切换语言
python align_srt_routeA.py --lang es
python align_srt_routeA.py --lang fr

# 自定义目录
python align_srt_routeA.py --lang ru --audio-dir /root/audio --output-dir /root/output
```

输出目录：`/root/aligned_routeA/`

**方案 A 并行版（5+ 文件时推荐）：**

```bash
# 5 个进程同时跑，充分利用 GPU
python align_srt_routeA_multi.py --lang es --audio-dir /root/input --output-dir /root/aligned_routeA --workers 5

# 调整并行数
python align_srt_routeA_multi.py --lang ru --audio-dir /root/audio --output-dir /root/output --workers 3
```

输出目录与原脚本一致。运行结束自动打印加速比。

**方案 B（口误较多时使用）：**

```bash
# 需先设置 LLM API Key
export LLM_API_KEY="your-key-here"

python align_srt_routeB_llm.py
```

输出目录：`/root/aligned_routeB/`

**方案 C（混合场景）：**

```bash
# 需先设置 LLM API Key
export LLM_API_KEY="your-key-here"

python align_srt_routeC_hybrid.py
```

输出目录：`/root/aligned_routeC/`

### 4. 工作流建议

1. **优先用方案 A**——速度快、不依赖外部 API、对齐质量已验证
2. **文件多时用并行版**——`align_srt_routeA_multi.py`，5+ 文件加速 ~4-5x
3. **发现口误/漏读/多读时切换方案 B**——LLM 能理解语义偏差
3. **B 覆盖不全时用方案 C**——锚点+CTC 混合，兼顾 B 的容错和 A 的精度
4. **对比验证**：`diff /root/aligned/xxx.aligned.srt /root/aligned_routeA/xxx.aligned.srt`

---

## 服务器环境（AutoDL）

```bash
# HF 镜像源（每次新开终端需重新设置）
export HF_ENDPOINT=https://hf-mirror.com

# 模型缓存路径
HF_HOME=/root/autodl-tmp/huggingface
TORCH_HOME=/root/autodl-tmp/torch
```

已缓存模型：
- `wav2vec2-large-xlsr-53-russian`（~1.26GB，safetensors 格式）
- `faster-whisper large-v3`
- `Silero-VAD`

---

## 局限性

- **方案 A** 的前提是参考文本与录音高度一致；有错字、漏句时无法自动纠正（此情况应切换方案 B 或 C）
- **方案 B** 依赖 LLM API 调用，需要网络连接和 API 费用；且 LLM 裁判并非 100% 可靠，需人工抽检验证
- **方案 C** 依赖方案 B 的锚点覆盖率；若 B 完全失败则退化为纯方案 A
- **超长音频**（单段 > 30 分钟）方案 A 自动分块处理，方案 B/C 无此限制
- **俄语对齐**使用社区维护的 wav2vec2 模型，精度可能略低于西/法语的 WhisperX 内置模型，建议对齐后人工抽检
- **音频截断**：上传不完整的 WAV 文件（文件头与实际数据不符）会导致 CTC 完全失败。请检查文件大小是否合理

---

## 参考文献

- Bain et al. (2023). *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.* Interspeech 2023. [arXiv:2303.00747](https://arxiv.org/abs/2303.00747)
- Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116)（XLSR 多语言预训练）
- Rousso et al. (2024). *Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment.* Interspeech 2024. [arXiv:2406.19363](https://arxiv.org/abs/2406.19363)
- McAuliffe et al. (2017). *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi.* Interspeech 2017.
- Silero Team (2021). *Silero VAD.* [GitHub](https://github.com/snakers4/silero-vad)
- SYSTRAN. *faster-whisper: High-performance Whisper inference.* [GitHub](https://github.com/SYSTRAN/faster-whisper)
- jonatasgrosman. *wav2vec2 Large XLSR-53 Russian.* [Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian)
