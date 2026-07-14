## 1. 模块概述

### 1.1 模块名称

`B5 - Agent 记忆管理模块`

### 1.2 模块说明

本模块负责 Agent 的记忆保存（Save）和加载（Load）两大核心功能，为 Agent 提供跨会话的知识积累和上下文复用能力。

````text
功能定位：
  - 记忆保存：将对话内容（messages、trace、final_answer）持久化存储
  - 记忆加载：支持精确ID查找、关键词检索、向量检索、混合检索四种方式
  - 记忆更新：支持对已有记忆进行更新和合并
  - 批量操作：支持批量保存、批量删除、列表查询

解决的问题：
  - Agent 跨会话记忆能力缺失问题
  - 记忆文档过长导致的存储和检索效率问题
  - 记忆检索的准确性和召回率问题

主要输入输出：
  - 输入：对话内容、检索词、配置参数
  - 输出：记忆文档、检索结果、操作状态

在系统中的作用：
  - 作为 Agent 的"记忆中枢"，为 B1 模块提供历史对话上下文
  - 使 Agent 能够记住之前的对话内容，实现连贯的多轮对话
  - 支持知识的积累和复用，提升 Agent 的回答质量


### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | 全部完成（记忆保存、记忆加载、关键词检索、精确ID查找） |
| 进阶要求 | 全部完成并额外添加（向量检索、混合检索、记忆压缩、记忆更新合并、批量操作、缓存机制） |
| 可独立运行的演示 | `test_b5_features.sh` 全功能测试脚本 |
| 与团队系统集成情况 | 通过 CLI 命令行和函数调用两种方式被 B1 模块调用 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | `>= 3.8` |
| 必要依赖 | `jieba`, `rank_bm25`, `scikit-learn`（可选，用于TF-IDF向量化） |
| 是否需要模型 | `不需要`（使用 TF-IDF 算法，无需外部嵌入模型） |
| 是否需要 GPU | `不需要` |
| 是否需要外部数据集 | `不需要` |

### 2.2 模型依赖

本模块不依赖外部模型，使用基于 TF-IDF 的轻量级向量检索方案。

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| `memory_search_test_data_01.json` | 自行构造 | `data/memory_inputs/` | 检索测试数据（数据分析主题） |
| `memory_search_test_data_02.json` | 自行构造 | `data/memory_inputs/` | 检索测试数据（汽车保养主题） |
| `memory_search_test_data_03.json` | 自行构造 | `data/memory_inputs/` | 检索测试数据（统计学主题） |
| `memory_save_compress_test.json` | 自行构造 | `data/memory_inputs/` | 压缩测试数据 |
| `memory_save_update_test.json` | 自行构造 | `data/memory_inputs/` | 更新测试数据（基础版本） |
| `memory_update_input_new.json` | 自行构造 | `data/memory_inputs/` | 更新测试数据（新版本） |
| `memory_batch_test_input.json` | 自行构造 | `data/memory_inputs/` | 批量操作测试数据 |
| `sample_final_answer_compress.md` | 自行构造 | `data/memory_inputs/` | 压缩测试文档（长文本） |
| `sample_messages_compress.json` | 自行构造 | `data/memory_inputs/` | 压缩测试消息数据 |
| `sample_trace_compress.json` | 自行构造 | `data/memory_inputs/` | 压缩测试追踪数据 |

### 2.4 安装步骤

```bash
# 进入项目目录
cd /root/siton-tmp/agent

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install jieba rank_bm25 scikit-learn

# 安装项目公共模块
pip install -e .
````

***

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
agent/
├── code/
│   └── b5_memory.py        # B5记忆模块核心实现（保存、检索、更新、批量操作）
├── configs/
│   └── memory.yaml         # 记忆模块配置文件（路径、容量、检索参数等）
├── data/
│   └── memory_inputs/      # 测试数据目录
│       ├── test_b5_features.sh         # 全功能测试脚本
│       ├── memory_search_test_data_*.json   # 检索测试数据
│       ├── memory_save_compress_test.json   # 压缩测试数据
│       ├── memory_save_update_test.json     # 更新测试数据
│       ├── memory_update_input_new.json     # 更新测试数据（新版本）
│       ├── memory_batch_test_input.json     # 批量操作测试数据
│       ├── sample_final_answer_compress.md  # 压缩测试文档
│       ├── sample_messages_compress.json    # 压缩测试消息
│       └── sample_trace_compress.json       # 压缩测试追踪数据
├── memory/                 # 记忆存储目录（运行时自动创建）
│   ├── memory_index.json           # 记忆索引文件
│   ├── global/                     # 全局记忆（跨会话共享）
│   └── conversations/              # 局部记忆（当前会话专属）
├── outputs/
│   └── B5_memory/                  # 输出目录（运行时自动创建）
└── README.md               # 项目说明文档
```

### 3.2 接口边界

| 类型 | 来源 / 去向     | 数据格式            | 说明                                                                          |
| -- | ----------- | --------------- | --------------------------------------------------------------------------- |
| 输入 | B1 模块 / 命令行 | JSON / CLI 参数   | 保存：conversation\_id、messages、trace、answer检索：query、top\_k、search\_mode       |
| 输出 | B1 模块 / 文件  | JSON / Markdown | 检索结果：selected\_memory\_docs保存结果：saved\_memory.json更新结果：updated\_memory.json |

***

## 4. 基础要求实现与演示

### 4.1 基础功能说明

```text
基础功能包括：
1. 记忆保存：将对话内容（messages、trace、final_answer）保存为 Markdown 文档
2. 记忆加载：支持精确 ID 查找和关键词检索两种方式
3. 索引管理：维护 memory_index.json 索引文件，记录记忆的元数据
4. 两种记忆类型：
   - 全局记忆（global）：跨会话共享，存储在 memory/global/ 目录
   - 局部记忆（conversation）：当前会话专属，存储在 memory/conversations/ 目录
```

### 4.2 基础功能实现路径

| 文件 / 函数 / 脚本                                         | 作用                    |
| ---------------------------------------------------- | --------------------- |
| `code/b5_memory.py:save_memory()`                    | 保存记忆核心函数，处理压缩和索引更新    |
| `code/b5_memory.py:load_memory()`                    | 加载记忆核心函数，支持多种检索方式     |
| `code/b5_memory.py:_search_by_keyword()`             | 关键词检索，使用 BM25Okapi 算法 |
| `code/b5_memory.py:_memory_paths()`                  | 解析配置文件，返回标准化路径        |
| `code/b5_memory.py:_read_index()` / `_write_index()` | 读取和写入索引文件             |

流程：

```text
保存流程：
[输入配置文件] -> [读取messages/trace/answer] -> [内容压缩（如有）] -> [生成Markdown文档] -> [更新索引] -> [输出结果]

检索流程：
[检索词/记忆ID] -> [加载索引] -> [读取记忆文档] -> [关键词/BM25检索] -> [排序过滤] -> [输出Top-K结果]
```

关键代码片段：

```python
def save_memory(config_path: str, conversation_id: str, save_type: str,
               messages_path: str | None = None, trace_path: str | None = None,
               answer_path: str | None = None, outdir: str | None = None) -> dict:
    """保存对话记忆为文档"""
    # 解析配置路径
    paths = _memory_paths(config_path)
    
    # 读取输入数据
    messages = _read_json_or_none(messages_path)
    trace = _read_json_or_none(trace_path)
    answer = _read_text_or_none(answer_path)
    
    # 内容压缩处理
    answer = _compress_content_if_needed(answer, paths)
    messages = _compress_messages_if_needed(messages, paths)
    
    # 生成记忆文档内容
    content = _format_memory_content(conversation_id, messages, trace, answer)
    
    # 保存文档和更新索引
    memory_id = _save_memory_doc(paths, conversation_id, save_type, content)
    
    return {"memory_id": memory_id, "status": "saved"}
```

### 4.3 基础功能输入格式与样例

**保存记忆输入格式**：

| 字段 / 输入文件         | 类型 / 格式                      | 是否必需 | 说明     |
| ----------------- | ---------------------------- | ---- | ------ |
| `conversation_id` | string                       | 是    | 会话唯一标识 |
| `save_type`       | string (conversation/global) | 是    | 记忆类型   |
| `messages_path`   | string (JSON文件路径)            | 是    | 对话消息列表 |
| `trace_path`      | string (JSON文件路径)            | 否    | 执行追踪信息 |
| `answer_path`     | string (文本文件路径)              | 是    | 最终回答   |

**检索记忆输入格式**：

| 字段 / 参数             | 类型 / 格式       | 是否必需 | 说明                          |
| ------------------- | ------------- | ---- | --------------------------- |
| `query`             | string        | 否    | 检索关键词（检索模式）                 |
| `select_memory_ids` | list\[string] | 否    | 精确记忆ID列表（精确查找模式）            |
| `top_k`             | int           | 否    | 返回前K个结果（默认3）                |
| `search_mode`       | string        | 否    | 检索模式（keyword/vector/hybrid） |
| `use_global_memory` | bool          | 否    | 是否使用全局记忆                    |

样例输入文件：

| 样例文件                                                  | 用途             |
| ----------------------------------------------------- | -------------- |
| `data/memory_inputs/memory_save_keyword_test_01.json` | 保存基础记忆测试       |
| `data/memory_inputs/memory_search_test_data_01.json`  | 检索测试数据（数据分析主题） |

### 4.4 基础功能演示命令

**保存记忆**：

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --save_type conversation \
    --save_input_path "../data/memory_inputs/memory_save_keyword_test_01.json" \
    --outdir "../outputs/B5_memory"
```

观察点：

- `memory/conversations/` 目录下生成新的 Markdown 文件
- `memory/memory_index.json` 更新，包含新记忆的元数据

**关键词检索**：

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode keyword \
    --outdir "../outputs/B5_memory/test_features/keyword"
```

观察点：

- `outputs/B5_memory/test_features/keyword/selected_memory.json` 包含检索结果
- 返回包含"数据"、"分析"关键词的记忆文档

**精确ID查找**：

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --select_memory_ids mem_conversation_conv_save_keyword_test_01 \
    --outdir "../outputs/B5_memory/test_features/exact"
```

观察点：

- 返回指定记忆ID的完整内容

### 4.5 基础功能输出格式

| 输出文件 / 返回字段                 | 格式       | 说明                                |
| --------------------------- | -------- | --------------------------------- |
| `selected_memory.json`      | JSON     | 检索结果，包含 selected\_memory\_docs 列表 |
| `saved_memory.json`         | JSON     | 保存结果，包含 memory\_id 和 status       |
| `updated_memory.json`       | JSON     | 更新结果，包含更新后的记忆内容                   |
| `memory/conversations/*.md` | Markdown | 局部记忆文档                            |
| `memory/global/*.md`        | Markdown | 全局记忆文档                            |

**检索结果格式**：

```json
{
    "selected_memory_docs": [
        {
            "memory_id": "mem_conversation_conv_xxx",
            "conversation_id": "conv_xxx",
            "content": "...",
            "_bm25_score": 0.85
        }
    ],
    "total_count": 3
}
```

### 4.6 基础功能结果截图


保存功能：
![保存功能](./屏幕截图%202026-07-13%20194731.png)

![保存功能](./屏幕截图%202026-07-13%20203053.png)
关键词查询：
![关键词查询](./屏幕截图%202026-07-13%20203256.png)

![关键词查询](./屏幕截图%202026-07-13%20203347.png)


## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求   | 是否完成 | 对应文件 / 函数                                                         | 简要说明                    |
| ------ | ---- | ----------------------------------------------------------------- | ----------------------- |
| 向量检索   | 是    | `code/b5_memory.py:_search_by_vector()`                           | 使用 TF-IDF + 余弦相似度实现语义检索 |
| 混合检索   | 是    | `code/b5_memory.py:_search_hybrid()`                              | 关键词检索和向量检索加权融合          |
| 记忆压缩   | 是    | `code/b5_memory.py:_compress_content_if_needed()`                 | 内容过长时自动压缩为摘要            |
| 记忆更新合并 | 是    | `code/b5_memory.py:update_memory()`                               | 支持四种合并策略                |
| 批量操作   | 是    | `code/b5_memory.py:batch_save_memory()` / `batch_delete_memory()` | 批量保存和删除记忆               |
| 缓存机制   | 是    | `code/b5_memory.py:MemoryCache`                                   | 双层 LRU 缓存，支持 TTL 过期     |

### 5.2 进阶功能 1：向量检索与混合检索

#### 功能说明

```text
问题背景：
- 基础版本的关键词检索只能匹配关键词，无法理解语义
- 例如：搜索"数据分析"无法找到"Python数据处理"相关文档

解决方案：
- 向量检索：使用 TF-IDF 算法将文本向量化，通过余弦相似度计算语义相似度
- 混合检索：将关键词检索（BM25得分）和向量检索（余弦相似度）进行加权融合
- 平衡精准匹配和语义理解能力，提升检索准确性和召回率

实现效果：
- 向量检索：能找到关键词不完全匹配但语义相关的文档
- 混合检索：综合关键词检索的精准性和向量检索的语义理解能力
```

#### 实现路径

| 文件 / 函数 / 脚本                            | 作用                         |
| --------------------------------------- | -------------------------- |
| `code/b5_memory.py:_search_by_vector()` | 向量检索核心函数，TF-IDF向量化 + 余弦相似度 |
| `code/b5_memory.py:_search_hybrid()`    | 混合检索核心函数，加权融合两种检索结果        |
| `code/b5_memory.py:_tfidf_vectorize()`  | TF-IDF向量化工具函数              |
| `configs/memory.yaml:vector_search`     | 向量检索配置（启用、相似度阈值）           |
| `configs/memory.yaml:hybrid_search`     | 混合检索配置（权重参数）               |

流程：

```text
向量检索：
[检索词] -> [TF-IDF向量化] -> [计算余弦相似度] -> [排序] -> [返回Top-K]

混合检索：
[检索词] -> [关键词检索] -> [向量检索] -> [加权融合] -> [排序] -> [返回Top-K]
```

#### 输入格式与样例

| 字段 / 参数             | 类型 / 格式 | 是否必需 | 说明                  |
| ------------------- | ------- | ---- | ------------------- |
| `query`             | string  | 是    | 检索词                 |
| `top_k`             | int     | 否    | 返回前K个结果（默认3）        |
| `search_mode`       | string  | 是    | `vector` 或 `hybrid` |
| `use_global_memory` | bool    | 否    | 是否使用全局记忆            |

#### 演示命令

**向量检索**：

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode vector \
    --outdir "../outputs/B5_memory/test_features/vector"
```

**混合检索**：

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode hybrid \
    --outdir "../outputs/B5_memory/test_features/hybrid"
```

#### 输出格式

```json
{
    "selected_memory_docs": [
        {
            "memory_id": "mem_conversation_conv_xxx",
            "_bm25_score": 0.75,
            "_vector_score": 0.65,
            "_hybrid_score": 0.70
        }
    ]
}
```

#### 示例图片
向量检索：
![向量检索](./屏幕截图%202026-07-13%20204215.png)

![向量检索](./屏幕截图%202026-07-13%20204238.png)
混合检索：
![混合检索](./屏幕截图%202026-07-13%20204332.png)

![混合检索](./屏幕截图%202026-07-13%20204345.png)

### 5.3 进阶功能 2：记忆压缩与长度管理

#### 功能说明

```text
问题背景：
- 长对话内容会导致记忆文档过大，占用存储空间
- 过长的内容会降低检索效率和准确性

解决方案：
- 内容压缩：当 Final Answer 超过 2000 字符或 Messages 超过 3000 字符时自动压缩
- 压缩方式：截取前 N 个字符，添加压缩提示
- 配置可调整：最大长度和压缩后摘要长度可通过配置文件调整

实现效果：
- 超长文档（6470字符）压缩后仅 201 字符，压缩率达 97%
- 保留核心信息，同时节省存储空间
```

#### 实现路径

| 文件 / 函数 / 脚本                                         | 作用                 |
| ---------------------------------------------------- | ------------------ |
| `code/b5_memory.py:_compress_content_if_needed()`    | 内容压缩核心函数           |
| `code/b5_memory.py:_compress_messages_if_needed()`   | 消息压缩核心函数           |
| `configs/memory.yaml:compression`                    | 压缩配置（启用、最大长度、摘要长度） |
| `data/memory_inputs/sample_final_answer_compress.md` | 压缩测试数据             |

#### 输入格式与样例

| 字段 / 参数                          | 类型 / 格式 | 是否必需 | 说明                        |
| -------------------------------- | ------- | ---- | ------------------------- |
| `compression.enable`             | bool    | 否    | 是否启用压缩（默认false）           |
| `compression.max_answer_chars`   | int     | 否    | Final Answer最大字符数（默认2000） |
| `compression.max_messages_chars` | int     | 否    | Messages最大字符数（默认3000）     |

#### 演示命令

```bash
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --save_type conversation \
    --save_input_path "../data/memory_inputs/memory_save_compress_test.json" \
    --outdir "../outputs/B5_memory/test_features/compress"
```

#### 输出格式

```markdown
# Conversation conv_compress_test

- memory_id: `mem_conversation_conv_compress_test`
...

## Final Answer

机器学习算法主要分为监督学习、无监督学习和强化学习三大类。

---
 内容已压缩（原长度: 6470 字符，压缩后: 201 字符）
如需查看完整内容，请参考原始记录。
```

### 5.4 进阶功能 3：记忆更新与合并

#### 功能说明

```text
问题背景：
- 同一会话可能需要更新记忆内容
- 新旧内容可能存在重复、补充或冲突

解决方案：
- 支持四种合并策略：
  - prefer_new：新内容优先，替换旧内容
  - prefer_old：旧内容优先，保留原有内容
  - keep_both：保留双方内容，标记差异
  - manual：仅生成差异报告，等待人工审核

实现效果：
- 正确处理冲突内容、补充内容和重复内容
- 提供灵活的合并策略，适应不同场景需求
```

#### 实现路径

| 文件 / 函数 / 脚本                            | 作用        |
| --------------------------------------- | --------- |
| `code/b5_memory.py:update_memory()`     | 更新记忆核心函数  |
| `code/b5_memory.py:_compare_sections()` | 对比新旧内容差异  |
| `code/b5_memory.py:_merge_prefer_new()` | 新内容优先合并策略 |
| `code/b5_memory.py:_merge_prefer_old()` | 旧内容优先合并策略 |
| `code/b5_memory.py:_merge_keep_both()`  | 保留双方合并策略  |

#### 演示命令

```bash
# 先保存基础记忆
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --save_type conversation \
    --save_input_path "../data/memory_inputs/memory_save_update_test.json" \
    --outdir "../outputs/B5_memory/test_features/update_save"

# 更新记忆（新内容优先）
python b5_memory.py --config "../configs/memory.yaml" \
    --update_memory_id mem_conversation_conv_update_test \
    --save_input_path "../data/memory_inputs/memory_update_input_new.json" \
    --merge_strategy prefer_new \
    --outdir "../outputs/B5_memory/test_features/update_result"
```

### 5.5 进阶功能 4：批量操作

#### 功能说明

```text
问题背景：
- 需要同时保存或删除多个记忆
- 逐个操作效率低下

解决方案：
- 批量保存：一次保存多个记忆文档
- 批量删除：一次删除多个记忆文档
- 列表查询：查看所有记忆文档

实现效果：
- 提高操作效率，减少重复调用
- 便于记忆管理和维护
```

#### 实现路径

| 文件 / 函数 / 脚本                              | 作用       |
| ----------------------------------------- | -------- |
| `code/b5_memory.py:batch_save_memory()`   | 批量保存核心函数 |
| `code/b5_memory.py:batch_delete_memory()` | 批量删除核心函数 |
| `code/b5_memory.py:list_memory()`         | 列表查询核心函数 |

#### 演示命令

```bash
# 批量保存
cd /root/siton-tmp/agent/code
python b5_memory.py --config "../configs/memory.yaml" \
    --batch_save "../data/memory_inputs/memory_batch_test_input.json" \
    --outdir "../outputs/B5_memory/test_features/batch"

# 列表查询
python b5_memory.py --config "../configs/memory.yaml" \
    --list \
    --list_type conversation \
    --outdir "../outputs/B5_memory/test_features/list"

# 批量删除
python b5_memory.py --config "../configs/memory.yaml" \
    --batch_delete mem_conversation_conv_batch_test_01 mem_conversation_conv_batch_test_02 \
    --outdir "../outputs/B5_memory/test_features/batch_delete"
```

***

## 6. 与团队系统的集成说明

### 6.1 调用方式

本模块支持两种调用方式：

**1. CLI 命令行调用**（推荐用于独立测试和调试）：

```bash
# B1 模块通过 subprocess 调用
python b5_memory.py --config <config> --query "数据分析" --top_k 3 --search_mode hybrid --outdir <outdir>
```

**2. 函数调用**（推荐用于系统集成）：

```python
from code.b5_memory import load_memory, save_memory, update_memory

# 检索记忆
results = load_memory(
    config_path="../configs/memory.yaml",
    query="数据分析",
    top_k=3,
    search_mode="hybrid",
    use_global_memory=True,
    outdir="../outputs/B5_memory"
)

# 保存记忆
saved = save_memory(
    config_path="../configs/memory.yaml",
    conversation_id="conv_001",
    save_type="conversation",
    messages_path="../data/memory_inputs/messages.json",
    answer_path="../data/memory_inputs/answer.md",
    outdir="../outputs/B5_memory"
)

# 更新记忆
updated = update_memory(
    config_path="../configs/memory.yaml",
    memory_id="mem_conversation_conv_001",
    save_input_path="../data/memory_inputs/update.json",
    merge_strategy="prefer_new",
    outdir="../outputs/B5_memory"
)
```

### 6.2 数据接口

**B1 → B5（检索请求）**：

```json
{
    "query": "数据分析",
    "top_k": 3,
    "search_mode": "hybrid",
    "use_global_memory": true,
    "select_memory_ids": ["mem_conversation_conv_001"]
}
```

**B5 → B1（检索结果）**：

```json
{
    "selected_memory_docs": [
        {
            "memory_id": "mem_conversation_conv_xxx",
            "content": "...",
            "_bm25_score": 0.75,
            "_vector_score": 0.65
        }
    ],
    "total_count": 3
}
```

**B1 → B5（保存请求）**：

```json
{
    "conversation_id": "conv_001",
    "save_type": "conversation",
    "messages": [{"role": "user", "content": "..."}, ...],
    "trace": {...},
    "answer": "..."
}
```

### 6.3 联调问题与解决方案

| 问题      | 原因             | 解决方案                                         |
| ------- | -------------- | -------------------------------------------- |
| 路径解析错误  | B1 和 B5 运行目录不同 | 使用绝对路径或 `resolve_from_file()` 函数             |
| 数据格式不一致 | 记忆内容字段定义不同     | 统一定义数据模型和字段名称                                |
| 检索结果为空  | 记忆索引未更新        | 确保保存后正确更新索引文件                                |
| 压缩功能未生效 | 配置未启用          | 在 memory.yaml 中设置 `compression.enable: true` |

***

## 7. 已知问题与后续改进

| 问题           | 当前原因                    | 后续改进                                |
| ------------ | ----------------------- | ----------------------------------- |
| 向量检索语义理解能力有限 | 使用 TF-IDF 算法，不如深度学习嵌入模型 | 引入 Sentence-BERT 或 BGE 等预训练模型进行向量嵌入 |
| 记忆压缩采用简单截取   | 没有使用 LLM 生成智能摘要         | 使用 LLM 生成智能摘要进行记忆压缩                 |
| 缓存机制较为简单     | 没有考虑缓存一致性和淘汰策略优化        | 实现缓存淘汰策略优化，考虑内存占用和访问频率              |
| 检索性能受文档数量影响  | 每次检索都重新计算 TF-IDF        | 引入增量索引更新和预计算机制                      |
| 缺少记忆过期自动清理   | 清理需要手动触发                | 实现定时任务自动清理过期记忆                      |

