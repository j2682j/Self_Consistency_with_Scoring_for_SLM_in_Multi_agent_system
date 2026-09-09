# AgentConsis-Verify

AgentConsis-Verify是一套以本地小型語言模型為核心的多代理人研究系統，主要用於執行 GAIA 類型的複合任務。系統將網路檢索、附件解析、確定型工具、多次獨立推理、過程驗證與最終答案選擇整合成可追蹤的實驗流程。

正式流程分為三個主要模組：

- **Evidence Prepare（證據準備）**：依任務與附件資訊取得網頁、文件、圖片、音訊、影片及確定型工具結果，整理成可供推理與驗證使用的證據。
- **Agent Reasoning**：多個 SLM Agent 在共享問題與證據下重複推理，必要時可在受限回合內主動呼叫工具。
- **Reasoning Step Verify（推理驗證與答案選擇）**：使用 VersaPRM 對解析後的推理步驟計算 reward probability，再透過 Evidence Support Check 與 Ordered Gates 選出最終答案。

## 支援的任務能力

| 類型 | 主要處理方式 | 說明 |
|---|---|---|
| 開放式事實查找 | SearXNG、全文擷取、FAISS、證據轉換 | 支援多輪檢索與 next-hop query。 |
| 網頁與 PDF | HTTP/full-page fetch、Playwright、PDF parser | 保留來源、頁面與 passage provenance。 |
| Office 與表格附件 | DOCX、PPTX、Excel reader | 保留段落、儲存格與結構化記錄。 |
| 圖片與影片 | `qwen3-vl:4b`、frame sampling、transcript | 視覺模型採需要時載入、使用後釋放。 |
| 音訊 | transcript 或 faster-whisper | 產生可引用的逐字內容。 |
| 數值與確定型問題 | calculator、deterministic handlers | 支援單位、座標、圖、字串、表格與棋盤等處理。 |
| 多代理人推理 | 三個 SLM Agent、每個 Agent 多次執行 | 以 Agent 內與跨 Agent 一致性建立候選集合。 |
| 推理過程驗證 | VersaPRM | 對每個 reasoning step 保留原始 reward probability。 |

## 快速開始

### 系統需求

- Windows 10/11 與 Python 3.12
- NVIDIA CUDA GPU（建議 16 GB VRAM 以上）
- 32 GB RAM（建議）
- [Ollama](https://ollama.com/)：Stage1、Query Generator 與視覺模型
- Docker Desktop：本地 SearXNG
- FFmpeg / ffprobe：音訊與影片處理
- Stockfish：棋類 handler
- Hugging Face 帳號：GAIA、VersaPRM 與其 gated base model

### 必要安裝 

'''
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
'''

`requirements.txt` 目前使用 CUDA 12.8 的 PyTorch wheel。若主機使用其他 CUDA 版本或 CPU，請先改成相符的 PyTorch 套件來源。

### 安裝 Ollama 模型
'''
ollama pull nemotron-3-nano:4b
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull qwen3-vl:4b
ollama list
'''
### 啟動 SearXNG
'''
docker compose -f searxng/core-config/docker-compose.yml up -d
'''

確認 JSON 搜尋介面：

```bash
curl "http://localhost:8080/search?q=OpenAI&format=json"
```

### 驗證 Hugging Face 與 VersaPRM

```bash
hf auth login
python score/test_versa_load.py
```

第一次執行會下載或確認 VersaPRM 與 gated base model。模型進入 Hugging Face cache 後，正式實驗可使用 local-files-only，避免每題重新連線。

## 命令列使用方式

### 語法

```bash
python run_gaia.py [-h] [--split {validation,test}] [--level {1,2,3}]
                   [--max-samples N] [--task-ids IDS]
                   [--stage1-runs-per-agent N] [--max-stage1-workers N]
                   [--temperature T] [--models MODEL_LIST]
                   [--evidence-prepare | --no-evidence-prepare]
                   [--enable-stage1-tool-use] [--without-stage2-score]
                   [--log-name NAME]
```

未指定額外開關時，Evidence Prepare、evidence-driven search 與 VersaPRM Stage2 皆會啟用；Stage1 Tool Use 與 candidate verification search 預設關閉。

### 主要參數

| 參數 | 預設值 | 功能與限制 |
|---|---:|---|
| `--data-dir` | `data/gaia` | GAIA 本地資料與附件目錄。 |
| `--split` | `validation` | 選擇 `validation` 或 `test`。 |
| `--level` | 全部 | 限定 GAIA Level 1、2 或 3。 |
| `--max-samples` | `1` | 最多執行題數。 |
| `--task-ids` | 無 | 以逗號指定 task id 或 id prefix，優先於 `--max-samples`。 |
| `--stage1-seed` | `42` | Stage1 重現種子；使用 `off` 關閉固定種子。 |
| `--stage1-runs-per-agent` | `3` | 每個 Agent 的獨立推理次數。 |
| `--max-stage1-workers` | 自動 | Stage1 最大平行 worker 數；VRAM 不足時建議設為 `1`。 |
| `--temperature` | `0.5` | Stage1 生成溫度。 |
| `--models` | 三個預設 Agent | 以逗號指定 Ollama 模型名稱。 |
| `--evidence-prepare` | 開啟 | 執行附件、deterministic 與網路證據準備。 |
| `--no-evidence-prepare` | 關閉 | 跳過 Evidence Prepare。 |
| `--enable-evidence-driven-search` | 開啟 | 啟用關係目標與 next-hop retrieval。 |
| `--bypass-search-labeler` | 關閉 | 跳過 EfficientRAG Labeler，直接建立句子級 evidence units。 |
| `--compact-search-evidence` | 關閉 | 使用較精簡的搜尋證據內容。 |
| `--enable-stage1-tool-use` | 關閉 | 允許 Agent 在推理過程提出工具請求。 |
| `--max-stage1-tool-turns` | `4` | Stage1 單次 trajectory 的基礎工具回合上限。 |
| `--stage1-prepared-search-budget` | `2` | 已有 prepared evidence 時，Stage1 可補充搜尋的任務級預算。 |
| `--enable-stage1-early-stop` | 關閉 | 啟用 Stage1 early-stop 實驗模式。 |
| `--without-stage2-score` | 關閉 | 跳過 VersaPRM scoring。 |
| `--stage2-verifier` | `versa` | Stage2 verifier；目前僅支援 VersaPRM。 |
| `--versa-prm-local-files-only` | 開啟 | 只從本地 Hugging Face cache 載入 VersaPRM。 |
| `--versa-prm-allow-download` | 關閉 | 允許啟動時下載缺少的 VersaPRM 檔案。 |
| `--enable-candidate-verification-search` | 關閉 | 所有 factual candidates 均 unsupported 時，執行受限候選驗證搜尋。 |
| `--log-name` | `gaia_run` | 輸出目錄與 Markdown 報告名稱。 |
| `--output-dir` | 自動 | 覆寫逐題 JSON 輸出目錄。 |
| `--report-md` | 自動 | 覆寫 Markdown 報告路徑。 |

完整參數請執行：

```bash
python run_gaia.py --help
```

### 執行範例

單題 smoke test：

```bash
/c/SCP/venv312/Scripts/python.exe run_gaia.py \
  --split validation \
  --level 1 \
  --max-samples 1 \
  --log-name smoke_level1
```

完整 Level 1 實驗：

```bash
/c/SCP/venv312/Scripts/python.exe run_gaia.py \
  --split validation \
  --level 1 \
  --max-samples 53 \
  --stage1-runs-per-agent 3 \
  --max-stage1-workers 1 \
  --temperature 0.3 \
  --enable-stage1-tool-use \
  --max-stage1-tool-turns 4 \
  --stage1-prepared-search-budget 2 \
  --versa-prm-local-files-only \
  --log-name level1_full_system
```

關閉 Labeler 的檢索對照實驗：

```bash
/c/SCP/venv312/Scripts/python.exe run_gaia.py \
  --split validation \
  --level 1 \
  --max-samples 53 \
  --bypass-search-labeler \
  --log-name level1_without_labeler
```

只執行 Stage1 與 Stage2，不預先準備證據：

```bash
/c/SCP/venv312/Scripts/python.exe run_gaia.py \
  --split validation \
  --level 1 \
  --max-samples 10 \
  --no-evidence-prepare \
  --enable-stage1-tool-use \
  --log-name level1_stage1_stage2_only
```

## 環境設定

在專案根目錄建立 `.env`。此檔案已由 `.gitignore` 排除，不應提交 token 或 API key。

```env
# Ollama / SLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_NATIVE_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=ollama
OLLAMA_TIMEOUT=180

Nemotron_MODEL_ID=nemotron-3-nano:4b
Qwen_MODEL_ID=qwen3:4b
Gemma_MODEL_ID=gemma3:4b
QUERY_GENERATOR_MODEL=qwen3:4b
TOOL_PLANNER_MODEL=qwen3:4b
SPAN_ROLE_CLASSIFIER_MODEL=qwen3:4b
OLLAMA_VISION_MODEL=qwen3-vl:4b

# Search / Retrieval
SEARCH_BACKEND=hybrid
SEARXNG_URL=http://localhost:8080
SEARCH_SALIENCE_HF_MODEL=BAAI/bge-m3
SEARCH_EMBED_MODEL=bge-m3
SEARCH_LABELER_DEVICE=cpu

# VersaPRM
VERSA_PRM_MODEL=UW-Madison-Lee-Lab/VersaPRM-Base-3B
VERSA_PRM_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
VERSA_PRM_DEVICE=auto
VERSA_PRM_DTYPE=auto
VERSA_PRM_LOCAL_FILES_ONLY=true

# Reproducibility / Hugging Face
SCP_STAGE1_SEED=42
HF_TOKEN=
```

`SEARCH_BACKEND=hybrid` 會依可用狀態選擇搜尋後端。SearXNG 不需要 API key；Tavily、SerpAPI 與 Perplexity 只有在對應環境變數存在時才會使用。

## 預設模型

| 系統角色 | 預設模型 | 執行方式 |
|---|---|---|
| Stage1 Agent 1 | `nemotron-3-nano:4b` | Ollama native chat |
| Stage1 Agent 2 | `qwen3:4b` | Ollama native chat |
| Stage1 Agent 3 | `gemma3:4b` | Ollama native chat |
| Query Generator | `qwen3:4b` | Ollama native chat，完成後 `keep_alive: 0` |
| Attachment Strategy | `qwen3:4b` | Ollama native chat |
| Span Role / Semantic Fact | `qwen3:4b` | Ollama native chat，完成後釋放 |
| Vision / Video Frames | `qwen3-vl:4b` | Ollama native chat，使用後釋放 |
| Semantic Impact / Retrieval | `BAAI/bge-m3` | Transformers |
| EfficientRAG Labeler | `models/labeler_v2` | Transformers，預設 CPU |
| Stage2 Verifier | `VersaPRM-Base-3B` | Transformers / PEFT |

Stage1 的預設 Agent 定義位於 `benchmark/gaia/gaia_runner.py`；alias 與實際 Ollama model id 的對應位於 `core/model_registry.py`。

## 工具模組

系統啟動時會註冊下列工具：

| 工具 | 功能 |
|---|---|
| `search` | 呼叫 SearXNG 或其他已設定的搜尋後端。 |
| `attachment_reader` | 解析附件並回傳結構化 evidence payload。 |
| `python_calculator` | 執行受限的數值計算。 |
| `deterministic_solver` | 呼叫已登錄且 schema 相符的確定型 handler。 |
| `video_evidence` | 下載影片、抽取影格並產生視覺證據。 |
| `video_transcript` | 讀取字幕或使用 ASR 產生 transcript。 |

Evidence Prepare 與 Stage1 Tool Use 共用同一個 `ToolManager`、工具契約與結果驗證層，但兩者的呼叫時機不同：前者在推理前建立共享證據，後者回應個別 Agent 在推理過程提出的工具請求。

## 輸出格式

每次實驗輸出一個任務一個 JSON，並建立一份彙整後的 Markdown 報告：

```text
outputs/<log_name>/
├── tasks/
│   ├── 001_<task_id>.json
│   ├── 002_<task_id>.json
│   └── ...
└── <log_name>.md
```

逐題 JSON 主要保存：

- 原始問題、附件 metadata、正確答案與系統答案
- exact/partial 評估結果、response time 與 token usage
- Evidence Prepare 的 search、attachment、handler 與 Fact Store
- 每個 Agent 的多次 reasoning、final answer 與 tool trajectory
- reasoning parser 與 structured-output repair 結果
- 每個推理步驟的 VersaPRM reward probabilities
- Evidence Support Check、候選聚合與完整 Ordered Gates trace
- 最終 winner、selection reason 與 GPU memory snapshot

Markdown 報告包含實驗設定、逐題結果、整體準確率、平均 token、平均 response time，以及搜尋與工具摘要。

## 依賴套件

主要 Python 依賴依功能分為：

```text
模型與推論：torch、transformers、accelerate、peft、sentence-transformers
向量檢索：faiss-cpu、BAAI/bge-m3、spaCy
網頁取得：requests、BeautifulSoup、lxml、trafilatura、readability、Playwright
附件解析：PyMuPDF、pypdf、python-docx、python-pptx、openpyxl、Pillow
影音處理：yt-dlp、youtube-transcript-api、faster-whisper、FFmpeg
資料與評估：pandas、pyarrow、huggingface-hub
測試：pytest
```

完整固定版本請參考 [`requirements.txt`](requirements.txt)。

## 回歸測試

測試不會由 `run_gaia.py` 載入。只有開發、重構與驗收時需要執行：

```bash
/c/SCP/venv312/Scripts/python.exe -m pytest -q
```

檢查主要 Python package 是否可編譯：

```bash
/c/SCP/venv312/Scripts/python.exe -m compileall benchmark context core parsers score tools
```

離線 replay 與診斷工具位於 `scripts/`，可在不重新呼叫 Agent、VersaPRM 或網路服務的情況下分析既有 task JSON。

## 專案結構

```text
SCP/
├── benchmark/gaia/               # GAIA loader、evaluator、CLI 與報告輸出
├── context/                      # Stage1 context 組裝、壓縮與 token budget
├── core/                         # Network、Evidence、Stage1、Stage2 主控流程
│   ├── network.py                # 單題端到端協調
│   ├── evidence_runner.py        # Evidence Prepare
│   ├── stage1_runner.py          # 多 Agent 重複推理
│   ├── stage1_tool_use_runner.py # Stage1 工具 trajectory
│   ├── candidate_path_evaluator.py # 候選路徑的證據與 Versa 評估
│   └── stage2_runner.py          # VersaPRM 呼叫與結果整理
├── parsers/                      # JSON、Stage1 reply、reasoning 與 tool request parser
├── score/                        # 答案契約、證據支持、Versa 與 winner selector
├── tools/
│   ├── attachment_reader/        # 文件、表格、圖片、音訊與壓縮檔解析
│   ├── attachment_strategy/      # 附件題策略規劃與執行
│   ├── deterministic_handlers/   # schema-bound handlers 與 trust gate
│   ├── evidence/fact_extraction/ # Semantic facts、grounding 與 derivation
│   ├── search_result_builder/    # Query、頁面、corpus、FAISS 與 evidence
│   └── video/                    # 影片下載、抽幀與視覺分析
├── models/                       # 本地 labeler 與 retriever checkpoints
├── scripts/                      # Evidence/selector 離線重播與診斷
├── tests/                        # 單元與整合測試
├── outputs/                      # 實驗輸出，預設不納入 git
├── requirements.txt             # 執行與測試依賴
└── run_gaia.py                   # 正式 GAIA CLI 入口
```

## 架構設計

### 整體工作流程

```text
GAIA Task + Attachment Metadata
          │
          ▼
Evidence Prepare
  ├── Attachment Strategy / Reader
  ├── Deterministic Handler
  └── Web Retrieval / Fact Extraction
          │
          ▼
Stage1 Context Builder
          │
          ▼
3 SLM Agents × repeated runs
  └── Optional bounded tool use
          │
          ▼
Answer normalization + candidate aggregation
          │
          ├── Evidence Support Check
          └── VersaPRM step verification
          │
          ▼
Ordered Gates
          │
          ▼
Final Answer + JSON/Markdown Report
```

### Evidence Prepare

Evidence Prepare 在 Agent 推理前建立共享上下文。若 metadata 含有附件，系統會先產生附件 profile，再由 Attachment Strategy 決定需要的資訊與可用 handler；一般 factual task 則進入 web retrieval。工具成功與失敗都會轉成具 provenance 的記錄，避免工具輸出直接被當成可信 final answer。

Web retrieval 的主要資料流為：

```text
Question
→ Question Role + Token/Span Semantic Impact
→ qwen3:4b Query Generation
→ Search Backend（SearXNG 為主要本地後端）
→ Source Filter / Fetch Candidate Selection
→ HTTP Full-page Fetch / Playwright Fallback
→ Cleaning / Chunking / Structured Records
→ Task-scoped Corpus
→ BGE-M3 Passage Embedding
→ FAISS Top-k Retrieval
→ EfficientRAG Labeler（可 bypass）
→ Span Role Classification
→ Semantic Fact / Grounding / Evidence Contract
→ Goal Completion / Next-hop Retrieval
→ Strict Evidence + Unverified References
```

每一題使用獨立的 temporary corpus 與 FAISS index。嚴格證據可進入後續 Evidence Support Check；未通過完整 contract 的高相關 passage 只作為 unverified reference 提供給 Agent 閱讀，不會因此取得可信支持等級。

### Stage1 多代理人推理

Stage1 預設使用三個不同 SLM Agent，每個 Agent 在相同問題與共享證據下獨立推理三次。每次輸出會先經過 structured schema parser、reasoning parser 與 answer repair，再進行 Agent 內答案聚合。

假設 Agent `i` 共產生 `R` 次答案，而其中最多的正規化答案出現 `m_i` 次，其自我一致性為：

```text
confidence_i = m_i / R
```

當三次答案完全相同時 confidence 為 `1.0`；兩次相同為約 `0.67`；三次皆不同為約 `0.33`。此值只描述生成穩定性，不代表答案已被證據證實。

啟用 Stage1 Tool Use 後，Agent 可輸出工具請求。`Stage1ToolUseRunner` 會解析請求、檢查工具能力與重複呼叫、執行工具、驗證結果，再將新證據回傳同一條 trajectory。若 Evidence Prepare 已有可用 search evidence，額外搜尋會受 task-level budget 限制。

### Stage2 推理步驟驗證

Reasoning Parser 先將 Agent 輸出切分為獨立步驟，並從最後一個 reasoning step 移除 Final Answer 區塊。VersaPRM 接收問題與已解析步驟，為每一步保留 reward probability：

```text
p(i,j) = P(step(i,j) is valid | question, previous steps)
```

系統同時保存候選路徑的最低步驟機率與幾何平均：

```text
P_min(i) = min_j p(i,j)

P_geo(i) = exp((1 / n_i) * sum_j log p(i,j))
```

`P_min` 用於觀察最脆弱的推理步驟，`P_geo` 表示整條推理鏈的整體穩定程度。VersaPRM 不直接產生答案，也不單獨決定 winner。

### Evidence Support Check 與 Ordered Gates

Evidence Support Check 對候選答案、answer requirement、semantic facts、附件、handler 與工具結果建立關係，區分 contradicted、no support、intermediate evidence、direct evidence、derived evidence 與 trusted tool final。

候選答案不使用單一 weighted score 排序，而是依序通過：

```text
Validity
→ Answer Requirement
→ Contradiction
→ Evidence Support
→ Corpus Attestation
→ Cross-Agent Consensus
→ Self-Consistency
→ VersaPRM Verification
```

前段 gate 負責淘汰格式錯誤、答非所問、證據衝突或未被語料支持的候選；後段 gate 才比較跨 Agent 支持數、Agent 內穩定性與推理步驟機率。完整決策過程會寫入 `selection_trace`，供離線重播與回歸分析。

## 目前限制

- 搜尋引擎找到正確頁面，不代表正確 passage 一定能形成 grounded direct evidence。
- JavaScript-heavy、付費牆、反爬蟲與媒體型頁面仍可能造成全文取得失敗。
- EfficientRAG Labeler 對開放網頁資料的 recall 仍屬實驗性，因此保留 bypass 模式作對照。
- 聚合、集合差異、跨文件關係與負向缺席事實需要完整來源，不可只根據 snippet 判斷。
- Self-consistency 可能放大多個小模型共享的錯誤知識，因此不能視為正確性證明。
- VersaPRM 評估推理過程，不保證能分辨具有合理敘述的錯誤答案。
- Candidate verification search 可能產生候選回聲，目前預設關閉。
- 模型採依階段載入與釋放以控制 VRAM，會增加部分 response time。

## 專案狀態

SCP 是持續開發中的研究程式，主要用於 GAIA 實驗、證據漏斗分析、候選答案選擇與 ablation study。資料結構、模型 checkpoint 與實驗參數仍可能隨研究進度調整。

本專案目前未提供獨立授權檔案；加入正式 license 前，請勿假設可自由再散布。
