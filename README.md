# B5 Memory Module - Agent 记忆管理模块

> 说明：本模块负责 Agent 的记忆保存和加载功能，为 Agent 提供跨会话的知识积累和上下文复用能力。

---

## 1. 模块概述

### 1.1 模块名称

`B5 - Agent 记忆管理模块`

### 1.2 模块说明

```text
功能定位：
    负责 Agent 的记忆保存（Save）和加载（Load）两大核心功能，
    为 Agent 提供跨会话的知识积累和上下文复用能力。

设计目标：
    1. 支持多种检索方式（关键词检索、向量检索、混合检索）
    2. 自动管理记忆文档长度，压缩过长内容
    3. 支持记忆更新与合并，处理冲突信息
    4. 提供批量操作机制，提升效率
    5. 实现缓存机制，减少重复计算

适用场景：
    - Agent 跨会话记忆管理
    - RAG（检索增强生成）系统
    - 对话历史管理
    - 知识图谱构建
```

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | 全部完成（记忆保存、关键词检索、精确ID查找） |
| 进阶要求 | 全部完成（向量检索、混合检索、记忆压缩、更新合并、批量操作、缓存机制） |
| 可独立运行的演示 | `test_b5_features.sh` 测试脚本 |
| 与团队系统集成情况 | 通过 CLI 命令行和函数调用与 B1 模块集成 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | `>= 3.8` |
| 必要依赖 | `jieba`, `rank_bm25`, `scikit-learn`, `PyYAML` |
| 是否需要模型 | `不需要` |
| 是否需要 GPU | `不需要` |
| 是否需要外部数据集 | `不需要` |

### 2.2 模型依赖

本模块不需要额外模型。

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| 测试输入数据 | 项目自带 | `data/test_inputs/` | 测试各功能模块 |
| 配置文件 | 项目自带 | `configs/memory.yaml` | 模块配置参数 |

### 2.4 安装步骤

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装模块
pip install -e .
```

---

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
b5-memory-module/
├── src/
│   ├── b5_memory/
│   │   └── __init__.py        # B5核心实现（记忆保存、检索、更新、批量操作）
│   └── common/
│       ├── io_utils.py        # 文件读写工具
│       ├── logging_utils.py   # 日志工具
│       └── path_utils.py      # 路径工具
├── configs/
│   └── memory.yaml            # 配置文件（路径、容量限制、检索参数）
├── data/
│   └── test_inputs/           # 测试输入数据（保存、检索、更新、批量操作）
├── outputs/                   # 输出目录（检索结果、保存结果、更新结果）
├── docs/                      # 文档目录
├── setup.py                   # 安装配置
├── requirements.txt           # 依赖列表
├── test_b5_features.sh        # 功能测试脚本
└── README.md                  # 模块说明文档
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| 输入 | B1 模块 / CLI 命令行 | JSON / 文件路径 / 参数 | 记忆内容、检索词、配置参数 |
| 输出 | B1 模块 / 保存到文件 | JSON / Markdown | 检索结果、保存状态、更新结果 |

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

```text
基础版本实现了以下功能：
1. 记忆保存：将对话记录（messages、trace、final_answer）保存为记忆文档
2. 关键词检索：使用 BM25Okapi 算法进行关键词匹配，返回最相关的前 k 个记忆
3. 精确 ID 查找：根据记忆 ID 精确加载指定记忆
4. 全局/局部记忆区分：支持全局记忆（跨会话共享）和局部记忆（当前会话专属）
```

### 4.2 基础功能实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `src/b5_memory/__init__.py` | 核心实现文件 |
| `save_memory()` | 保存记忆到文件系统 |
| `load_memory()` | 加载记忆（精确ID + 关键词检索） |
| `_search_by_keyword()` | BM25 关键词检索实现 |

流程：

```text
保存记忆：
[输入JSON] -> [解析输入] -> [生成摘要] -> [压缩长内容] -> [保存Markdown] -> [更新索引]

检索记忆：
[检索词] -> [加载索引] -> [加载文档内容] -> [BM25计算] -> [排序] -> [返回结果]
```

### 4.3 基础功能输入格式与样例

| 字段 / 输入文件 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `conversation_id` | string | 是 | 会话ID |
| `save_type` | string (conversation/global) | 是 | 记忆类型 |
| `messages_path` | string (JSON文件路径) | 是 | 对话消息路径 |
| `trace_path` | string (JSON文件路径) | 否 | 执行轨迹路径 |
| `final_answer_path` | string (MD文件路径) | 是 | 最终答案路径 |

样例输入：

| 样例文件 | 用途 |
|---|---|
| `data/test_inputs/memory_save_keyword_test_01.json` | 测试基础记忆保存功能 |

### 4.4 基础功能演示命令

```bash
# 保存记忆
python -m src.b5_memory --config configs/memory.yaml \
    --save_type conversation \
    --save_input_path data/test_inputs/memory_save_keyword_test_01.json \
    --outdir outputs/save

# 关键词检索
python -m src.b5_memory --config configs/memory.yaml \
    --query "数据分析" \
    --top_k 3 \
    --search_mode keyword \
    --outdir outputs/keyword_search

# 精确ID查找
python -m src.b5_memory --config configs/memory.yaml \
    --select_memory_ids mem_conversation_conv_save_keyword_test_01 \
    --outdir outputs/exact_search
```

观察点：
- 保存命令执行后，`memory/conversations/` 目录下生成新的记忆文档
- 关键词检索命令执行后，`outputs/keyword_search/selected_memory.json` 包含检索结果
- 精确ID查找命令直接返回指定的记忆内容

### 4.5 基础功能输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `saved_memory.json` | JSON | 保存结果（memory_id、路径、状态） |
| `selected_memory.json` | JSON | 检索结果（memory_id、content、得分） |
| `listed_memory.json` | JSON | 记忆列表（所有memory_id和元数据） |
| `memory/conversations/*.md` | Markdown | 记忆文档（messages、trace、final_answer） |

### 4.6 基础功能结果截图

保存功能：
![保存功能](./屏幕截图%202026-07-13%20194731.png)

![保存功能](./屏幕截图%202026-07-13%20203053.png)
关键词查询：
![关键词查询](./屏幕截图%202026-07-13%20203256.png)

![关键词查询](./屏幕截图%202026-07-13%20203347.png)

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求 | 是否完成 | 对应文件 / 函数 | 简要说明 |
|---|---|---|---|
| 向量检索 | ✅ | `_search_by_vector()` | 使用 TF-IDF + 余弦相似度实现语义检索 |
| 混合检索 | ✅ | `_search_hybrid()` | 关键词检索和向量检索加权融合 |
| 记忆压缩 | ✅ | `_compress_content()` | 自动压缩过长的记忆内容 |
| 记忆更新合并 | ✅ | `update_memory()` | 支持四种合并策略 |
| 批量操作 | ✅ | `batch_save_memory()`, `batch_delete_memory()` | 批量保存、删除、列表查询 |
| 缓存机制 | ✅ | `MemoryCache` | 双层 LRU 缓存，支持 TTL 过期 |

### 5.2 进阶功能 1：向量检索

#### 功能说明

```text
向量检索使用 TF-IDF（词频-逆文档频率）算法将文本向量化，
然后使用余弦相似度计算文档与查询的相似度，返回相似度最高的前 k 个记忆。

相比关键词检索，向量检索能够发现语义上相关但关键词不完全匹配的文档，
例如查询"数据分析"可以匹配到包含"Python数据处理"、"机器学习"等内容的文档。
```

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `_search_by_vector()` | 向量检索核心函数 |
| `_build_tfidf_index()` | 构建 TF-IDF 索引 |
| `_calculate_cosine_similarity()` | 计算余弦相似度 |

流程：

```text
[检索词] -> [TF-IDF向量化] -> [计算余弦相似度] -> [排序] -> [返回结果]
```

#### 输入格式与样例

| 字段 / 输入文件 / 配置项 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `--query` | string | 是 | 检索词 |
| `--top_k` | integer | 否 | 返回数量，默认3 |
| `--search_mode` | string | 是 | 必须为 `vector` |
| `similarity_threshold` | float | 否 | 相似度阈值，默认0.0 |

#### 演示命令

```bash
python -m src.b5_memory --config configs/memory.yaml \
    --query "数据分析" \
    --top_k 3 \
    --search_mode vector \
    --outdir outputs/vector_search
```

#### 输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `selected_memory.json` | JSON | 检索结果（含 `_vector_score` 字段） |

#### 示例图片

![向量检索](./屏幕截图%202026-07-13%20204215.png)

![向量检索](./屏幕截图%202026-07-13%20204238.png)


### 5.3 进阶功能 2：混合检索

#### 功能说明

```text
混合检索将关键词检索（BM25）和向量检索（TF-IDF）的结果进行加权融合，
综合两者的优势，提供更全面的检索结果。

关键词检索的优势：精准匹配，对关键词敏感
向量检索的优势：语义理解，能发现相关但关键词不完全匹配的文档

混合检索通过可调权重参数（keyword_weight、vector_weight）
将两种检索方式的得分融合，得到综合得分。
```

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `_search_hybrid()` | 混合检索核心函数 |
| `_search_by_keyword()` | 关键词检索 |
| `_search_by_vector()` | 向量检索 |

流程：

```text
[检索词] -> [关键词检索(BM25)] -> [向量检索(TF-IDF)] -> [加权融合] -> [排序] -> [返回结果]
```

#### 输入格式与样例

| 字段 / 输入文件 / 配置项 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `--query` | string | 是 | 检索词 |
| `--top_k` | integer | 否 | 返回数量，默认3 |
| `--search_mode` | string | 是 | 必须为 `hybrid` |
| `keyword_weight` | float | 否 | 关键词权重，默认0.5 |
| `vector_weight` | float | 否 | 向量权重，默认0.5 |

#### 演示命令

```bash
python -m src.b5_memory --config configs/memory.yaml \
    --query "数据分析" \
    --top_k 3 \
    --search_mode hybrid \
    --outdir outputs/hybrid_search
```

#### 输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `selected_memory.json` | JSON | 检索结果（含 `_bm25_score`、`_vector_score`、`_hybrid_score`） |

#### 示例图片

![混合检索](./屏幕截图%202026-07-13%20204332.png)

![混合检索](./屏幕截图%202026-07-13%20204345.png)

### 5.4 进阶功能 3：记忆压缩

#### 功能说明

```text
当记忆内容（messages或final_answer）超过配置的最大长度时，
自动将内容压缩为摘要，保留核心信息，减少存储空间占用。

压缩策略：
1. final_answer 超过 max_answer_chars 时，截取前 summary_length 个字符
2. messages 超过 max_messages_chars 时，截取前半部分
3. 在压缩内容末尾添加提示，告知用户内容已被压缩
```

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `_compress_content()` | 内容压缩核心函数 |
| `_should_compress()` | 判断是否需要压缩 |

#### 演示命令

```bash
python -m src.b5_memory --config configs/memory.yaml \
    --save_type conversation \
    --save_input_path data/test_inputs/memory_save_compress_test.json \
    --outdir outputs/compress
```

### 5.5 进阶功能 4：记忆更新与合并

#### 功能说明

```text
支持对已有的记忆文档进行更新，对比新旧文档内容，
对重复、补充或冲突信息进行合并、修正或冲突管理。

合并策略：
1. prefer_new：优先使用新内容（默认）
2. prefer_old：优先使用旧内容
3. keep_both：保留新旧内容
4. manual：人工干预（预留）
```

#### 演示命令

```bash
# 先保存基础记忆
python -m src.b5_memory --config configs/memory.yaml \
    --save_type conversation \
    --save_input_path data/test_inputs/memory_save_update_test.json \
    --outdir outputs/update_save

# 更新记忆（替换为新内容）
python -m src.b5_memory --config configs/memory.yaml \
    --update_memory_id mem_conversation_conv_update_test \
    --save_input_path data/test_inputs/memory_update_input_new.json \
    --merge_strategy prefer_new \
    --outdir outputs/update
```

### 5.6 进阶功能 5：批量操作

#### 功能说明

```text
支持批量保存、删除和列表查询记忆，提升操作效率。
```

#### 演示命令

```bash
# 批量保存
python -m src.b5_memory --config configs/memory.yaml \
    --batch_save data/test_inputs/memory_batch_test_input.json \
    --outdir outputs/batch_save

# 列表查询
python -m src.b5_memory --config configs/memory.yaml \
    --list \
    --list_type conversation \
    --outdir outputs/list

# 批量删除
python -m src.b5_memory --config configs/memory.yaml \
    --batch_delete mem_conversation_conv_batch_test_01 \
    --outdir outputs/batch_delete
```

### 5.7 进阶功能 6：缓存机制

#### 功能说明

```text
实现双层 LRU 缓存：
1. 文档内容缓存：缓存已加载的记忆文档内容
2. 检索结果缓存：缓存检索结果

缓存支持 TTL（Time-To-Live）过期机制，
在配置中可以设置缓存大小和过期时间。
```

#### 演示命令

```bash
# 查看缓存统计
python -m src.b5_memory --config configs/memory.yaml \
    --cache_stats \
    --outdir outputs/cache

# 清除缓存
python -m src.b5_memory --config configs/memory.yaml \
    --clear_cache \
    --outdir outputs/cache
```

---

## 6. 与团队系统的集成说明

本模块通过以下方式与团队系统集成：

### 6.1 调用方式

**CLI 命令行调用（推荐）**：
```bash
python -m src.b5_memory --config configs/memory.yaml \
    --query "数据分析" \
    --top_k 3 \
    --search_mode hybrid \
    --outdir outputs/B5_memory
```

**函数调用**：
```python
from b5_memory import load_memory, save_memory

# 加载记忆
selected_memory = load_memory(
    config_path="configs/memory.yaml",
    memory_ids=["mem_conversation_conv_000"],
    use_global_memory=True,
    query="数据分析",
    top_k=3,
    outdir="outputs/B5_memory"
)

# 保存记忆
saved_memory = save_memory(
    config_path="configs/memory.yaml",
    conversation_id="conv_000",
    save_type="conversation",
    messages_path="data/messages.json",
    trace_path="data/trace.json",
    answer_path="data/final_answer.md",
    outdir="outputs/B5_memory"
)
```

### 6.2 接口格式

**输入**：
- CLI 参数：配置文件路径、操作类型、输入路径、输出路径
- 函数参数：同上

**输出**：
- JSON 文件：`selected_memory.json`（检索结果）、`saved_memory.json`（保存结果）、`updated_memory.json`（更新结果）
- Markdown 文件：`memory/conversations/*.md`（记忆文档）

### 6.3 联调问题与解决方案

| 问题 | 解决方案 |
|---|---|
| 路径配置不一致 | 使用相对路径，通过 `resolve_from_file` 统一解析 |
| 全局/局部记忆区分 | 在配置中区分 `global_memory_dir` 和 `conversation_memory_dir` |
| 索引文件更新冲突 | 使用文件锁机制，确保并发写入安全 |

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| 向量检索语义理解有限 | 使用 TF-IDF 算法，无法理解深层语义 | 引入预训练嵌入模型（如 BERT、Sentence-BERT） |
| 中文分词效果一般 | 使用 jieba 分词，对专业领域词汇支持有限 | 引入领域词典或使用更先进的分词工具 |
| 缓存机制简单 | 仅支持 LRU 缓存 | 引入多级缓存策略，支持缓存持久化 |
| 无分布式支持 | 仅支持单进程运行 | 引入 Redis 作为分布式缓存 |
| 无版本控制 | 记忆更新直接覆盖 | 引入版本控制，支持历史回滚 |