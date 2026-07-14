"""
B5 记忆模块 - Agent 记忆管理组件

功能定位：
    负责 Agent 的记忆保存（Save）和加载（Load）两大核心功能，
    为 Agent 提供跨会话的知识积累和上下文复用能力。

设计架构：
    ┌─────────────────────────────────────────────────────────────┐
    │                    B5-memory 模块                           │
    ├─────────────────────────────────────────────────────────────┤
    │  配置层：memory.yaml（路径配置 + 容量限制）                  │
    ├─────────────────────────────────────────────────────────────┤
    │  存储层：索引 + 文档分离                                     │
    │    ├── memory_index.json（元数据索引，轻量级）               │
    │    ├── memory/global/*.md（全局记忆，跨会话共享）            │
    │    └── memory/conversations/*.md（对话记忆，当前会话专属）   │
    ├─────────────────────────────────────────────────────────────┤
    │  核心逻辑层：                                               │
    │    ├── save_memory() → 将对话保存为记忆文档                  │
    │    ├── load_memory() → 加载记忆（精确ID + BM25检索）         │
    │    └── BM25 检索 → jieba分词 + BM25Okapi关键词匹配          │
    ├─────────────────────────────────────────────────────────────┤
    │  接口层：CLI命令行 + 函数调用（供B1集成）                     │
    └─────────────────────────────────────────────────────────────┘

使用方式：
    1. CLI保存记忆：
        python b5_memory.py --config ../configs/memory.yaml \
            --save_type conversation --save_input_path ../data/memory_inputs/memory_save_input.json \
            --outdir ../outputs/B5_memory

    2. CLI加载记忆（精确ID）：
        python b5_memory.py --config ../configs/memory.yaml \
            --select_memory_ids mem_conversation_conv_000 --use_global_memory true \
            --outdir ../outputs/B5_memory

    3. CLI加载记忆（关键词检索）：
        python b5_memory.py --config ../configs/memory.yaml \
            --query "Agent系统如何工作" --top_k 3 \
            --outdir ../outputs/B5_memory

    4. 函数调用（B1集成）：
        from b5_memory import load_memory, save_memory
        selected_memory = load_memory(config_path, memory_ids, use_global, query, top_k, outdir)
        saved_memory = save_memory(config_path, conversation_id, save_type, messages_path, trace_path, answer_path, outdir)
"""

from __future__ import annotations
import sys
from pathlib import Path

# 设置项目根目录到 Python 路径，确保跨目录导入公共模块
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 标准库导入
import argparse
import json
import re
import sys
from pathlib import Path

# 第三方库导入：jieba 用于中文分词，rank_bm25 用于关键词检索
import jieba
from rank_bm25 import BM25Okapi

# 公共模块导入：IO工具、日志工具、路径工具
from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


def _memory_paths(config_path: str | Path) -> dict:
    """
    解析记忆配置文件，返回标准化的路径字典和容量限制参数。

    配置文件格式（memory.yaml）：
        memory:
          root_dir: ../memory              # 记忆根目录（相对于配置文件）
          global_memory_dir: global        # 全局记忆子目录
          conversation_memory_dir: conversations  # 对话记忆子目录
          index_path: memory_index.json    # 索引文件名
          max_memory_chars: 2000           # 最大加载字符数限制
          vector_search:                   # 向量检索配置（可选）
            enable: true
            similarity_threshold: 0.3
          compression:                     # 内容压缩配置（可选）
            enable: true                   # 是否启用压缩
            max_answer_chars: 2000        # 最终回答最大字符数
            max_messages_chars: 3000      # 消息历史最大字符数
            summary_length: 200           # 压缩后摘要长度

    参数：
        config_path: 配置文件路径（字符串或 Path 对象）

    返回：
        包含以下键的字典：
            - "root": 记忆根目录的绝对路径（Path）
            - "global": 全局记忆目录的绝对路径（Path）
            - "conversations": 对话记忆目录的绝对路径（Path）
            - "index": 索引文件的绝对路径（Path）
            - "max_chars": 最大加载字符数（int）
            - "vector_search": 向量检索配置字典（包含 enable, similarity_threshold）
            - "compression": 内容压缩配置字典

    异常：
        ValueError: 配置文件格式不正确或缺少必要字段
    """
    # 将配置路径解析为绝对路径
    path = Path(config_path).resolve()
    
    # 读取 YAML 配置文件
    config = read_yaml(path)
    
    # 验证配置文件结构：必须包含 memory 对象
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    
    # 验证必要字段是否存在
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    
    # 将 root_dir 解析为相对于配置文件的绝对路径
    root = resolve_from_file(memory["root_dir"], path)
    
    # 验证 max_memory_chars 必须是正整数
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    
    # 解析向量检索配置（使用 TF-IDF 实现，无需外部模型）
    vector_search = memory.get("vector_search", {})
    vector_config = {
        "enable": vector_search.get("enable", False),
        "similarity_threshold": vector_search.get("similarity_threshold", 0.3),
    }
    
    # 解析内容压缩配置
    compression = memory.get("compression", {})
    compression_config = {
        "enable": compression.get("enable", False),
        "max_answer_chars": compression.get("max_answer_chars", 2000),
        "max_messages_chars": compression.get("max_messages_chars", 3000),
        "summary_length": compression.get("summary_length", 200),
    }
    
    # 解析记忆过期配置
    expiration = memory.get("expiration", {})
    expiration_config = {
        "enable": expiration.get("enable", False),
        "conversation_ttl_days": expiration.get("conversation_ttl_days", 30),
        "global_ttl_days": expiration.get("global_ttl_days", 365),
        "auto_clean_on_load": expiration.get("auto_clean_on_load", True),
        "max_conversations": expiration.get("max_conversations", 1000),
    }
    
    # 解析混合检索配置
    hybrid_search = memory.get("hybrid_search", {})
    hybrid_config = {
        "enable": hybrid_search.get("enable", True),
        "keyword_weight": hybrid_search.get("keyword_weight", 0.5),
        "vector_weight": hybrid_search.get("vector_weight", 0.5),
    }
    
    # 解析缓存配置
    cache = memory.get("cache", {})
    cache_config = {
        "enable": cache.get("enable", True),
        "max_size": cache.get("max_size", 100),
        "ttl_minutes": cache.get("ttl_minutes", 30),
    }
    
    # 返回标准化的路径字典
    return {
        "root": root,
        "global": root / memory["global_memory_dir"],           # 全局记忆目录
        "conversations": root / memory["conversation_memory_dir"],  # 对话记忆目录
        "index": root / memory["index_path"],                   # 索引文件路径
        "max_chars": max_chars,                                 # 容量限制
        "vector_search": vector_config,                         # 向量检索配置
        "compression": compression_config,                       # 内容压缩配置
        "expiration": expiration_config,                         # 记忆过期配置
        "hybrid_search": hybrid_config,                         # 混合检索配置
        "cache": cache_config,                                  # 缓存配置
    }


def _read_index(index_path: Path) -> dict:
    """
    读取记忆索引文件，返回索引字典。

    索引文件格式（memory_index.json）：
        {
          "mem_conversation_conv_001": {
            "memory_id": "mem_conversation_conv_001",
            "memory_type": "conversation",
            "title": "Conversation conv_001",
            "summary": "对话摘要...",
            "path": "conversations/conv_001.md",
            "conversation_id": "conv_001",
            "created_at": "2026-06-25T14:46:36+08:00",
            "updated_at": "2026-06-25T14:46:36+08:00"
          }
        }

    参数：
        index_path: 索引文件的绝对路径（Path 对象）

    返回：
        索引字典，key 为 memory_id，value 为元数据字典

    异常：
        ValueError: 索引文件存在但格式不是 JSON 对象
    """
    # 如果索引文件不存在，返回空字典（首次使用时自动创建）
    if not index_path.exists():
        return {}
    
    # 读取并解析 JSON 文件
    index = read_json(index_path)
    
    # 验证索引必须是字典格式
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    
    return index


# ==================== 关键词检索模块 ====================
def _tokenize_chinese(text: str) -> list[str]:
    """
    中文分词函数，使用 jieba 对文本进行分词处理。

    分词策略：
        1. 使用 jieba.cut 进行中文分词
        2. 过滤掉长度为1的标点符号和空白字符
        3. 保留长度>1的中文词和纯字母数字词（防止英文被过滤）

    参数：
        text: 待分词的中文文本

    返回：
        分词后的词列表
    """
    # 使用 jieba 进行分词
    words = list(jieba.cut(text))
    
    # 过滤：保留长度>1的词，或者纯字母数字（防止英文单词被过滤）
    return [w.strip() for w in words if len(w.strip()) > 1 or (w.strip().isalnum() and len(w.strip()) > 0)]


def _search_by_keyword(query: str, all_docs: list[dict], top_k: int = 3) -> list[dict]:
    """
    使用 BM25Okapi 算法对记忆文档进行关键词检索和排序。

    BM25 算法优势：
        - 不需要预先训练，开箱即用
        - 考虑文档长度归一化，比 TF-IDF 更适合长文本检索
        - 配合 jieba 分词支持中文语义检索

    参数：
        query: 用户查询字符串（中文）
        all_docs: 待检索的文档列表，每个文档包含 'content'、'title'、'memory_id' 等字段
        top_k: 返回前 K 个最相关的文档（默认3）

    返回：
        按相关性排序的文档列表，每个文档会额外添加 '_bm25_score' 字段（调试用）
    """
    # 边界处理：如果没有文档或没有查询词，直接返回空或前K个文档
    if not all_docs or not query:
        return all_docs[:top_k] if all_docs else []
    
    # 步骤1：构建语料库（将标题和内容拼接作为检索文本）
    corpus = []
    for doc in all_docs:
        title = doc.get('title', '')
        content = doc.get('content', '')
        text = f"{title} {content}"  # 标题权重更高（靠前）
        corpus.append(text)
    
    # 步骤2：对语料库进行分词，构建 BM25 模型
    tokenized_corpus = [_tokenize_chinese(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 步骤3：对查询词进行分词
    tokenized_query = _tokenize_chinese(query)
    
    # 步骤4：计算每个文档与查询的相似度得分
    scores = bm25.get_scores(tokenized_query)
    
    # 步骤5：按得分降序排序，保留所有文档
    doc_score_pairs = [(idx, score) for idx, score in enumerate(scores)]
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # 步骤6：提取前 top_k 个文档的索引
    selected_indices = [idx for idx, _ in doc_score_pairs[:top_k]]
    
    # 步骤7：构造结果列表，添加 BM25 得分字段（便于调试）
    result_docs = []
    for idx in selected_indices:
        doc = all_docs[idx].copy()
        doc['_bm25_score'] = scores[idx]  # 保留分数供调试分析
        result_docs.append(doc)
    
    return result_docs
# ==================== 关键词检索模块结束 ====================


# ==================== 向量检索模块 ====================
def _build_vocab(corpus: list[str]) -> dict:
    """
    构建词汇表，为每个词分配唯一的索引。

    参数：
        corpus: 文本语料库列表

    返回：
        词汇表字典：{word: index}
    """
    vocab = {}
    index = 0
    for text in corpus:
        words = _tokenize_chinese(text)
        for word in words:
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab


def _tfidf_vectorize(corpus: list[str]) -> tuple:
    """
    使用 TF-IDF 算法将文本转换为向量。

    参数：
        corpus: 文本语料库列表

    返回：
        (vocab, tfidf_matrix) 元组
    """
    import numpy as np

    # 构建词汇表
    vocab = _build_vocab(corpus)
    vocab_size = len(vocab)
    doc_count = len(corpus)

    # 计算词频矩阵（TF）
    tf_matrix = np.zeros((doc_count, vocab_size))
    for doc_idx, text in enumerate(corpus):
        words = _tokenize_chinese(text)
        word_count = len(words)
        for word in words:
            if word in vocab:
                tf_matrix[doc_idx, vocab[word]] += 1
        # 归一化
        tf_matrix[doc_idx] /= word_count if word_count > 0 else 1

    # 计算逆文档频率（IDF）
    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log((doc_count + 1) / (df + 1)) + 1

    # 计算 TF-IDF 矩阵
    tfidf_matrix = tf_matrix * idf

    return vocab, tfidf_matrix


def _text_to_tfidf_vector(text: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    """
    将单个文本转换为 TF-IDF 向量。

    参数：
        text: 待转换的文本
        vocab: 词汇表字典
        idf: 逆文档频率数组

    返回：
        TF-IDF 向量
    """
    import numpy as np

    vocab_size = len(vocab)
    tf = np.zeros(vocab_size)
    words = _tokenize_chinese(text)
    word_count = len(words)

    for word in words:
        if word in vocab:
            tf[vocab[word]] += 1

    if word_count > 0:
        tf /= word_count

    return tf * idf


def _search_by_vector(
    query: str,
    all_docs: list[dict],
    top_k: int = 3,
    model_name: str = None,
    cache_dir: str | Path = None,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """
    使用向量检索（语义搜索）对记忆文档进行相似度检索和排序。

    向量检索流程：
        1. 使用 TF-IDF 算法将文本转换为向量
        2. 计算查询向量与每个文档向量的余弦相似度
        3. 按相似度降序排序，返回前 K 个最相似的文档

    参数：
        query: 用户查询字符串（支持中英文）
        all_docs: 待检索的文档列表，每个文档包含 'content'、'title'、'memory_id' 等字段
        top_k: 返回前 K 个最相关的文档（默认3）
        model_name: 嵌入模型名称（本实现不使用）
        cache_dir: 模型缓存目录（本实现不使用）
        similarity_threshold: 相似度阈值，低于此值的结果不返回（默认0.3）

    返回：
        按相似度排序的文档列表，每个文档会额外添加 '_vector_score' 字段（余弦相似度）
    """
    import numpy as np

    # 边界处理：如果没有文档或没有查询词，直接返回空或前K个文档
    if not all_docs or not query:
        return all_docs[:top_k] if all_docs else []

    # 步骤1：构建语料库（将标题和内容拼接作为检索文本）
    corpus = []
    for doc in all_docs:
        title = doc.get("title", "")
        content = doc.get("content", "")
        text = f"{title} {content}"
        corpus.append(text)

    # 步骤2：使用 TF-IDF 向量化
    vocab, tfidf_matrix = _tfidf_vectorize(corpus)

    # 计算 IDF 数组（用于查询向量）
    doc_count = len(corpus)
    df = np.sum(tfidf_matrix > 0, axis=0)
    idf = np.log((doc_count + 1) / (df + 1)) + 1

    # 步骤3：将查询词转换为向量
    query_vector = _text_to_tfidf_vector(query, vocab, idf)

    # 步骤4：计算余弦相似度
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return all_docs[:top_k]

    doc_norms = np.linalg.norm(tfidf_matrix, axis=1)
    doc_norms[doc_norms == 0] = 1e-9

    similarities = np.dot(tfidf_matrix, query_vector) / (doc_norms * query_norm)

    # 步骤5：按相似度降序排序，只保留相似度>阈值的文档
    doc_score_pairs = [
        (idx, score) for idx, score in enumerate(similarities) if score >= similarity_threshold
    ]
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 步骤6：提取前 top_k 个文档的索引
    selected_indices = [idx for idx, _ in doc_score_pairs[:top_k]]

    # 步骤7：构造结果列表，添加向量相似度得分字段（便于调试）
    result_docs = []
    for idx in selected_indices:
        doc = all_docs[idx].copy()
        doc["_vector_score"] = float(similarities[idx])
        result_docs.append(doc)

    return result_docs
# ==================== 向量检索模块结束 ====================


# ==================== 混合检索模块 ====================
def _search_hybrid(
    query: str,
    all_docs: list[dict],
    top_k: int = 3,
    similarity_threshold: float = 0.3,
    keyword_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[dict]:
    """
    使用混合检索机制（关键词检索 + 向量检索）对记忆文档进行综合排序。

    混合检索策略：
        1. 分别执行关键词检索（BM25）和向量检索（TF-IDF）
        2. 获取各自的 top_k * 2 结果（扩大召回范围）
        3. 合并去重，保留每个文档的两种得分
        4. 使用加权融合公式计算综合得分：score = keyword_weight * bm25_score + vector_weight * cosine_score
        5. 按综合得分降序排序，返回前 top_k 个文档

    参数：
        query: 用户查询字符串（支持中英文）
        all_docs: 待检索的文档列表，每个文档包含 'content'、'title'、'memory_id' 等字段
        top_k: 返回前 K 个最相关的文档（默认3）
        similarity_threshold: 向量检索相似度阈值（默认0.3）
        keyword_weight: 关键词检索得分权重（默认0.5）
        vector_weight: 向量检索得分权重（默认0.5）

    返回：
        按综合得分排序的文档列表，每个文档会额外添加 '_hybrid_score'、'_bm25_score'、'_vector_score' 字段
    """
    if not all_docs or not query:
        return all_docs[:top_k] if all_docs else []

    # 步骤1：分别执行两种检索（扩大召回范围，取 top_k * 2）
    expanded_top_k = top_k * 2
    keyword_results = _search_by_keyword(query, all_docs, top_k=expanded_top_k)
    vector_results = _search_by_vector(
        query, all_docs, top_k=expanded_top_k, similarity_threshold=similarity_threshold
    )

    # 步骤2：合并去重，构建得分映射
    doc_scores = {}  # {memory_id: {'bm25_score': float, 'vector_score': float, 'doc': dict}}
    
    # 收集关键词检索结果
    for doc in keyword_results:
        memory_id = doc['memory_id']
        if memory_id not in doc_scores:
            doc_scores[memory_id] = {'bm25_score': 0.0, 'vector_score': 0.0, 'doc': doc}
        doc_scores[memory_id]['bm25_score'] = doc.get('_bm25_score', 0.0)
    
    # 收集向量检索结果
    for doc in vector_results:
        memory_id = doc['memory_id']
        if memory_id not in doc_scores:
            doc_scores[memory_id] = {'bm25_score': 0.0, 'vector_score': 0.0, 'doc': doc}
        doc_scores[memory_id]['vector_score'] = doc.get('_vector_score', 0.0)

    # 步骤3：计算综合得分并排序
    scored_docs = []
    for memory_id, scores in doc_scores.items():
        doc = scores['doc'].copy()
        
        # 归一化得分（BM25 得分范围通常为 0-20，余弦相似度范围为 0-1）
        bm25_score = scores['bm25_score']
        vector_score = scores['vector_score']
        
        # 归一化 BM25 得分到 [0, 1] 范围
        normalized_bm25 = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0.0
        normalized_vector = vector_score if vector_score >= similarity_threshold else 0.0
        
        # 加权融合
        hybrid_score = (keyword_weight * normalized_bm25) + (vector_weight * normalized_vector)
        
        doc['_bm25_score'] = bm25_score
        doc['_vector_score'] = vector_score
        doc['_hybrid_score'] = hybrid_score
        
        scored_docs.append(doc)

    # 步骤4：按综合得分降序排序
    scored_docs.sort(key=lambda x: x['_hybrid_score'], reverse=True)

    # 步骤5：返回前 top_k 个结果
    return scored_docs[:top_k]
# ==================== 混合检索模块结束 ====================


# ==================== 缓存管理模块 ====================
class MemoryCache:
    """
    记忆检索缓存管理器，使用 LRU（最近最少使用）策略。

    缓存类型：
        1. 文档内容缓存：缓存已读取的记忆文档内容
        2. 检索结果缓存：缓存检索查询结果

    缓存失效机制：
        1. 保存/更新/删除记忆时自动失效相关缓存
        2. 定时清理过期缓存
        3. 缓存达到最大容量时淘汰最久未使用的条目
    """

    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        """
        初始化缓存管理器。

        参数：
            max_size: 缓存最大条目数（默认100）
            ttl_minutes: 缓存过期时间（分钟，默认30）
        """
        from collections import OrderedDict
        self._doc_cache = OrderedDict()       # 文档内容缓存：{memory_id: {'content': str, 'timestamp': float}}
        self._search_cache = OrderedDict()    # 检索结果缓存：{cache_key: {'result': list, 'timestamp': float}}
        self._max_size = max_size
        self._ttl_seconds = ttl_minutes * 60

    def _clean_expired(self, cache: OrderedDict):
        """
        清理过期的缓存条目。

        参数：
            cache: 缓存字典（OrderedDict）
        """
        import time
        now = time.time()
        keys_to_remove = []
        for key, value in cache.items():
            if now - value.get('timestamp', 0) > self._ttl_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del cache[key]

    def _evict_lru(self, cache: OrderedDict):
        """
        淘汰最久未使用的缓存条目（LRU策略）。

        参数：
            cache: 缓存字典（OrderedDict）
        """
        while len(cache) >= self._max_size:
            cache.popitem(last=False)

    def get_doc(self, memory_id: str) -> str | None:
        """
        获取文档内容缓存。

        参数：
            memory_id: 记忆 ID

        返回：
            文档内容（如果缓存有效），否则返回 None
        """
        import time
        self._clean_expired(self._doc_cache)
        
        if memory_id in self._doc_cache:
            # 更新访问时间（LRU）
            item = self._doc_cache.pop(memory_id)
            item['timestamp'] = time.time()
            self._doc_cache[memory_id] = item
            return item['content']
        
        return None

    def set_doc(self, memory_id: str, content: str):
        """
        设置文档内容缓存。

        参数：
            memory_id: 记忆 ID
            content: 文档内容
        """
        import time
        self._clean_expired(self._doc_cache)
        self._evict_lru(self._doc_cache)
        
        self._doc_cache[memory_id] = {
            'content': content,
            'timestamp': time.time(),
        }

    def invalidate_doc(self, memory_id: str):
        """
        失效指定文档的缓存。

        参数：
            memory_id: 记忆 ID
        """
        if memory_id in self._doc_cache:
            del self._doc_cache[memory_id]

    def invalidate_all_docs(self):
        """
        失效所有文档内容缓存。
        """
        self._doc_cache.clear()

    def get_search(self, query: str, search_mode: str, top_k: int, use_global_memory: bool) -> list | None:
        """
        获取检索结果缓存。

        参数：
            query: 查询字符串
            search_mode: 检索模式（keyword/vector/hybrid）
            top_k: 返回数量
            use_global_memory: 是否使用全局记忆

        返回：
            检索结果列表（如果缓存有效），否则返回 None
        """
        import time
        self._clean_expired(self._search_cache)
        
        cache_key = f"{search_mode}:{top_k}:{use_global_memory}:{query}"
        
        if cache_key in self._search_cache:
            # 更新访问时间（LRU）
            item = self._search_cache.pop(cache_key)
            item['timestamp'] = time.time()
            self._search_cache[cache_key] = item
            return item['result']
        
        return None

    def set_search(self, query: str, search_mode: str, top_k: int, use_global_memory: bool, result: list):
        """
        设置检索结果缓存。

        参数：
            query: 查询字符串
            search_mode: 检索模式（keyword/vector/hybrid）
            top_k: 返回数量
            use_global_memory: 是否使用全局记忆
            result: 检索结果列表
        """
        import time
        self._clean_expired(self._search_cache)
        self._evict_lru(self._search_cache)
        
        cache_key = f"{search_mode}:{top_k}:{use_global_memory}:{query}"
        
        self._search_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
        }

    def invalidate_search(self):
        """
        失效所有检索结果缓存。
        """
        self._search_cache.clear()

    def get_stats(self) -> dict:
        """
        获取缓存统计信息。

        返回：
            统计信息字典
        """
        import time
        now = time.time()
        
        doc_expired = sum(1 for v in self._doc_cache.values() if now - v.get('timestamp', 0) > self._ttl_seconds)
        search_expired = sum(1 for v in self._search_cache.values() if now - v.get('timestamp', 0) > self._ttl_seconds)
        
        return {
            'doc_cache_size': len(self._doc_cache),
            'doc_cache_expired': doc_expired,
            'search_cache_size': len(self._search_cache),
            'search_cache_expired': search_expired,
            'max_size': self._max_size,
            'ttl_minutes': self._ttl_seconds // 60,
        }


# 全局缓存实例
_global_cache = None


def _get_cache(config_path: str | None = None) -> MemoryCache:
    """
    获取全局缓存实例（单例模式）。

    参数：
        config_path: 配置文件路径（首次初始化时使用）

    返回：
        MemoryCache 实例
    """
    global _global_cache
    
    if _global_cache is None:
        if config_path:
            paths = _memory_paths(config_path)
            cache_config = paths.get("cache", {})
            max_size = cache_config.get("max_size", 100)
            ttl_minutes = cache_config.get("ttl_minutes", 30)
            _global_cache = MemoryCache(max_size=max_size, ttl_minutes=ttl_minutes)
        else:
            _global_cache = MemoryCache()
    
    return _global_cache


def clear_cache(config_path: str | None = None):
    """
    清除所有缓存。

    参数：
        config_path: 配置文件路径（可选）
    """
    global _global_cache
    _global_cache = None


def get_cache_stats(config_path: str | None = None) -> dict:
    """
    获取缓存统计信息。

    参数：
        config_path: 配置文件路径（可选）

    返回：
        缓存统计信息字典
    """
    cache = _get_cache(config_path)
    return cache.get_stats()
# ==================== 缓存管理模块结束 ====================


# ==================== 内容压缩模块 ====================
def _compress_answer(answer: str, max_chars: int, summary_length: int) -> tuple:
    """
    压缩最终回答内容，如果超过最大长度则生成摘要。

    参数：
        answer: 原始回答内容
        max_chars: 最大字符数限制
        summary_length: 压缩后摘要长度

    返回：
        (compressed_answer, was_compressed) 元组
    """
    if len(answer) <= max_chars:
        return answer, False

    # 使用文本摘要算法生成摘要
    summary = _text_summary(answer, summary_length)
    
    # 在摘要后添加提示，说明内容已被压缩
    compressed = (
        f"{summary}\n\n"
        f"---\n"
        f"⚠️ 内容已压缩（原长度: {len(answer)} 字符，压缩后: {len(summary)} 字符）\n"
        f"如需查看完整内容，请参考原始记录。"
    )
    
    return compressed, True


def _compress_messages(messages: list, max_chars: int, summary_length: int) -> tuple:
    """
    压缩消息历史，如果超过最大长度则保留关键消息并生成摘要。

    参数：
        messages: 原始消息列表
        max_chars: 最大字符数限制
        summary_length: 压缩后摘要长度

    返回：
        (compressed_messages, was_compressed) 元组
    """
    # 计算当前消息的总字符数
    messages_json = json.dumps(messages, ensure_ascii=False)
    if len(messages_json) <= max_chars:
        return messages, False

    # 策略：保留最后几条消息（通常是最重要的），对前面的消息生成摘要
    # 先尝试只保留最后3条消息
    if len(messages) > 3:
        recent_messages = messages[-3:]
        old_messages = messages[:-3]
        
        # 生成旧消息的摘要
        old_messages_text = "\n".join(
            f"{msg.get('role', '')}: {msg.get('content', '')[:100]}" 
            for msg in old_messages
        )
        summary = _text_summary(old_messages_text, summary_length)
        
        # 创建摘要消息
        summary_msg = {
            "role": "system",
            "content": f"[对话摘要]：{summary}\n（前 {len(old_messages)} 条消息已压缩）"
        }
        
        compressed = [summary_msg] + recent_messages
    else:
        # 如果消息数量不多，但每条消息很长，对每条消息压缩
        compressed = []
        for msg in messages:
            content = msg.get("content", "")
            if len(content) > max_chars // len(messages):
                compressed_content = _text_summary(content, summary_length)
                compressed.append({
                    "role": msg.get("role", ""),
                    "content": f"{compressed_content}\n（内容已压缩）",
                    "tool_calls": msg.get("tool_calls", []),
                })
            else:
                compressed.append(msg)
    
    return compressed, True


def _text_summary(text: str, max_length: int) -> str:
    """
    生成文本摘要，基于关键词提取和句子选择。

    参数：
        text: 原始文本
        max_length: 摘要最大长度

    返回：
        摘要文本
    """
    if len(text) <= max_length:
        return text

    # 使用 jieba 分词提取关键词
    words = _tokenize_chinese(text)
    
    # 统计词频（简单实现）
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # 按词频排序，提取关键词
    keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
    keyword_set = {word for word, _ in keywords}
    
    # 按句子分割
    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 选择包含关键词最多的句子
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for kw in keyword_set if kw in sentence)
        scored_sentences.append((sentence, score))
    
    # 按得分排序
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # 组合摘要，不超过最大长度
    summary = ""
    for sentence, _ in scored_sentences:
        if len(summary) + len(sentence) <= max_length:
            summary += sentence + "。"
        else:
            # 截断最后一个句子
            remaining = max_length - len(summary)
            if remaining > 0:
                summary += sentence[:remaining] + "。"
            break
    
    return summary.strip()
# ==================== 内容压缩模块结束 ====================


# ==================== 记忆过期管理模块 ====================
def _is_expired(memory_item: dict, ttl_days: int) -> bool:
    """
    判断记忆是否已过期。

    参数：
        memory_item: 记忆索引项
        ttl_days: 过期时间（天）

    返回：
        True 如果已过期，False 否则
    """
    updated_at = memory_item.get("updated_at")
    if not updated_at:
        return False

    import datetime
    try:
        # 解析 ISO 8601 格式的时间戳
        if "+" in updated_at:
            # 处理带时区的格式：2026-06-25T14:46:36+08:00
            updated_time = datetime.datetime.fromisoformat(updated_at)
        else:
            # 处理不带时区的格式：2026-06-25T14:46:36
            updated_time = datetime.datetime.fromisoformat(updated_at + "+00:00")
        
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - updated_time
        return delta.days >= ttl_days
    except Exception:
        return False


def _clean_expired_memories(config_path: str, outdir: str | None = None) -> dict:
    """
    清理过期的记忆文档。

    清理策略：
        1. 根据配置的 TTL 判断记忆是否过期
        2. 对话记忆和全局记忆使用不同的 TTL
        3. 删除过期的记忆文档文件
        4. 更新索引，移除过期条目
        5. 如果对话记忆数量超过上限，删除最早的记忆

    参数：
        config_path: 配置文件路径
        outdir: 输出目录（可选，用于保存清理日志）

    返回：
        清理结果字典，包含删除的记忆列表和数量
    """
    # 步骤1：解析配置
    paths = _memory_paths(config_path)
    expiration_config = paths.get("expiration", {})
    
    if not expiration_config.get("enable", False):
        return {"status": "disabled", "message": "expiration is not enabled"}
    
    conversation_ttl = expiration_config.get("conversation_ttl_days", 30)
    global_ttl = expiration_config.get("global_ttl_days", 365)
    max_conversations = expiration_config.get("max_conversations", 1000)
    
    # 步骤2：读取索引
    index = _read_index(paths["index"])
    
    # 步骤3：找出过期的记忆
    expired_ids = []
    for memory_id, item in index.items():
        memory_type = item.get("memory_type", "conversation")
        ttl = global_ttl if memory_type == "global" else conversation_ttl
        
        if _is_expired(item, ttl):
            expired_ids.append(memory_id)
    
    # 步骤4：检查对话记忆数量是否超过上限
    conversation_ids = [
        mid for mid, item in index.items() 
        if item.get("memory_type") == "conversation"
    ]
    
    if len(conversation_ids) > max_conversations:
        # 按更新时间排序，删除最早的
        sorted_conversations = sorted(
            conversation_ids,
            key=lambda mid: index[mid].get("updated_at", ""),
            reverse=False
        )
        overflow_count = len(conversation_ids) - max_conversations
        overflow_ids = sorted_conversations[:overflow_count]
        expired_ids.extend(overflow_ids)
    
    # 去重
    expired_ids = list(dict.fromkeys(expired_ids))
    
    # 步骤5：删除过期的记忆文档和索引条目
    deleted_count = 0
    deleted_ids = []
    errors = []
    
    for memory_id in expired_ids:
        item = index.get(memory_id)
        if not item:
            continue
        
        relative_path = item.get("path")
        if not relative_path:
            continue
        
        document_path = (paths["root"] / relative_path).resolve()
        
        try:
            # 删除文档文件
            if document_path.exists():
                document_path.unlink()
            
            # 从索引中移除
            del index[memory_id]
            
            deleted_count += 1
            deleted_ids.append(memory_id)
        except Exception as e:
            errors.append({"memory_id": memory_id, "error": str(e)})
    
    # 步骤6：更新索引文件
    if deleted_count > 0:
        write_json(index, paths["index"])
    
    # 步骤7：构建结果
    now = now_iso()
    result = {
        "status": "success",
        "deleted_count": deleted_count,
        "deleted_ids": deleted_ids,
        "errors": errors,
        "timestamp": now,
        "expiration_config": {
            "conversation_ttl_days": conversation_ttl,
            "global_ttl_days": global_ttl,
            "max_conversations": max_conversations,
        },
    }
    
    # 步骤8：保存输出文件
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "cleaned_memories.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "clean",
                "status": "success",
                "deleted_count": deleted_count,
                "deleted_ids": deleted_ids,
            },
            output_dir / "memory_log.jsonl",
        )
    
    # 失效相关缓存
    cache = _get_cache(config_path)
    for memory_id in deleted_ids:
        cache.invalidate_doc(memory_id)
    cache.invalidate_search()
    
    return result


def clean_memory(config_path: str, outdir: str | None = None) -> dict:
    """
    清理过期记忆的公共接口。

    参数：
        config_path: 配置文件路径
        outdir: 输出目录（可选）

    返回：
        清理结果字典
    """
    return _clean_expired_memories(config_path, outdir)
# ==================== 记忆过期管理模块结束 ====================


# ==================== 批量操作模块 ====================
def batch_save_memory(
    config_path: str,
    batch_input: list[dict],
    outdir: str | None = None,
) -> dict:
    """
    批量保存记忆文档。

    参数：
        config_path: 配置文件路径
        batch_input: 批量输入列表，每个元素包含 conversation_id、save_type、messages_path、trace_path、answer_path
        outdir: 输出目录（可选）

    返回：
        批量保存结果字典
    """
    results = []
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(batch_input):
        try:
            result = save_memory(
                config_path=config_path,
                conversation_id=item["conversation_id"],
                save_type=item["save_type"],
                messages_path=item["messages_path"],
                trace_path=item["trace_path"],
                answer_path=item["answer_path"],
                outdir=None,
            )
            results.append({
                "index": idx,
                "status": "success",
                "memory_id": result.get("memory_id"),
                "conversation_id": item["conversation_id"],
            })
            success_count += 1
        except Exception as e:
            results.append({
                "index": idx,
                "status": "failed",
                "conversation_id": item.get("conversation_id"),
                "error": str(e),
            })
            fail_count += 1
    
    now = now_iso()
    final_result = {
        "status": "success" if fail_count == 0 else "partial",
        "total_count": len(batch_input),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
        "timestamp": now,
    }
    
    if outdir:
        output_dir = Path(outdir)
        write_json(final_result, output_dir / "batch_saved_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "batch_save",
                "status": final_result["status"],
                "total_count": len(batch_input),
                "success_count": success_count,
                "fail_count": fail_count,
            },
            output_dir / "memory_log.jsonl",
        )
    
    return final_result


def batch_delete_memory(
    config_path: str,
    memory_ids: list[str],
    outdir: str | None = None,
) -> dict:
    """
    批量删除记忆文档。

    参数：
        config_path: 配置文件路径
        memory_ids: 要删除的记忆 ID 列表
        outdir: 输出目录（可选）

    返回：
        批量删除结果字典
    """
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    results = []
    success_count = 0
    fail_count = 0
    
    for memory_id in memory_ids:
        try:
            item = index.get(memory_id)
            if not item:
                results.append({
                    "memory_id": memory_id,
                    "status": "not_found",
                })
                fail_count += 1
                continue
            
            relative_path = item.get("path")
            if relative_path:
                document_path = (paths["root"] / relative_path).resolve()
                if document_path.exists():
                    document_path.unlink()
            
            del index[memory_id]
            
            results.append({
                "memory_id": memory_id,
                "status": "success",
            })
            success_count += 1
        except Exception as e:
            results.append({
                "memory_id": memory_id,
                "status": "failed",
                "error": str(e),
            })
            fail_count += 1
    
    if success_count > 0:
        write_json(index, paths["index"])
    
    now = now_iso()
    final_result = {
        "status": "success" if fail_count == 0 else "partial",
        "total_count": len(memory_ids),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
        "timestamp": now,
    }
    
    if outdir:
        output_dir = Path(outdir)
        write_json(final_result, output_dir / "batch_deleted_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "batch_delete",
                "status": final_result["status"],
                "total_count": len(memory_ids),
                "success_count": success_count,
                "fail_count": fail_count,
            },
            output_dir / "memory_log.jsonl",
        )
    
    # 失效相关缓存
    cache = _get_cache(config_path)
    for memory_id in memory_ids:
        cache.invalidate_doc(memory_id)
    cache.invalidate_search()
    
    return final_result


def list_memory(
    config_path: str,
    memory_type: str | None = None,
    outdir: str | None = None,
) -> dict:
    """
    列出所有记忆文档。

    参数：
        config_path: 配置文件路径
        memory_type: 记忆类型过滤（conversation/global，默认为 None 列出所有）
        outdir: 输出目录（可选）

    返回：
        记忆列表字典
    """
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    memories = []
    for memory_id, item in index.items():
        if memory_type and item.get("memory_type") != memory_type:
            continue
        
        memories.append({
            "memory_id": memory_id,
            "memory_type": item.get("memory_type"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "conversation_id": item.get("conversation_id"),
            "path": item.get("path"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        })
    
    # 按更新时间排序
    memories.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    now = now_iso()
    result = {
        "status": "success",
        "count": len(memories),
        "memories": memories,
        "timestamp": now,
    }
    
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "listed_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "list",
                "status": "success",
                "count": len(memories),
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result
# ==================== 批量操作模块结束 ====================


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    top_k: int = 3,
    outdir: str | None = None,
    search_mode: str = "keyword",
) -> dict:
    """
    加载记忆文档的核心函数，支持四种检索模式：
        1. 精确 ID 查找模式：根据指定的 memory_id 精确加载
        2. 关键词检索模式：根据用户查询使用 BM25 算法检索相关记忆
        3. 向量检索模式：根据用户查询使用语义向量检索相关记忆
        4. 混合检索模式：结合关键词检索和向量检索，加权融合得分进行综合排序

    参数：
        config_path: 配置文件路径（memory.yaml）
        selected_memory_ids: 指定的 memory_id 列表（精确查找模式使用）
        use_global_memory: 是否加载全局记忆（True/False）
        query: 用户查询字符串（检索模式使用，为 None 时使用精确查找模式）
        top_k: 检索返回的前 K 个结果（默认3）
        outdir: 输出目录路径（可选，用于保存结果文件）
        search_mode: 检索模式（keyword/vector/hybrid，默认 keyword）

    返回：
        包含以下字段的字典：
            - "status": 状态（success / partial / error / no_match）
            - "query": 用户查询字符串
            - "search_mode": 使用的检索模式
            - "selected_memory_docs": 选中的记忆文档列表
            - "max_memory_chars": 最大加载字符数限制
            - "total_chars": 实际加载的总字符数
            - "truncated": 是否有文档被截断
            - "errors": 错误信息列表（精确查找模式）

    输出文件（当 outdir 不为 None 时）：
        - selected_memory.json: 选中的记忆内容和元数据
        - memory_log.jsonl: 操作日志（追加写入）
    """
    # 参数校验：selected_memory_ids 必须是字符串列表
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    
    # 解析配置文件，获取路径和容量限制
    paths = _memory_paths(config_path)
    
    # 自动清理过期记忆（如果启用）
    expiration_config = paths.get("expiration", {})
    if expiration_config.get("enable", False) and expiration_config.get("auto_clean_on_load", True):
        _clean_expired_memories(config_path)
    
    # 读取记忆索引
    index = _read_index(paths["index"])
    
    # 获取缓存实例（如果启用）
    cache_config = paths.get("cache", {})
    cache_enabled = cache_config.get("enable", True)
    cache = _get_cache(config_path) if cache_enabled else None

    # ==================== 模式一：检索模式（关键词/向量） ====================
    # 触发条件：传入了 query 参数，且没有指定 selected_memory_ids
    # 适用场景：未知具体记忆 ID，需要根据语义召回相关知识
    if query and not selected_memory_ids:
        # 步骤1：尝试从缓存获取检索结果
        cached_ranked_docs = None
        if cache:
            cached_ranked_docs = cache.get_search(query, search_mode, top_k, use_global_memory)
        
        if cached_ranked_docs:
            # 缓存命中，直接使用缓存结果
            ranked_docs = cached_ranked_docs
        else:
            # 缓存未命中，执行完整检索流程
            # 步骤1.1：收集所有可用的记忆文档
            all_docs = []
            all_ids = list(index.keys())
            
            # 根据 use_global_memory 参数决定是否过滤全局记忆
            if not use_global_memory:
                all_ids = [mid for mid in all_ids if index.get(mid, {}).get('memory_type') != 'global']
            
            # 遍历所有记忆 ID，读取对应的文档内容
            for memory_id in all_ids:
                # 获取记忆元数据
                metadata = index.get(memory_id)
                if not metadata:
                    continue
                
                # 获取文档相对路径
                relative_path = metadata.get('path')
                if not relative_path:
                    continue
                
                # 构建文档绝对路径并解析
                document_path = (paths['root'] / relative_path).resolve()
                
                # 安全检查：防止路径穿越攻击（确保路径在记忆根目录内）
                try:
                    document_path.relative_to(paths['root'].resolve())
                except ValueError:
                    continue  # 跳过路径非法的文档
                
                # 检查文件是否存在
                if not document_path.is_file():
                    continue
                
                # 读取文档内容（优先使用缓存）
                original = None
                if cache:
                    original = cache.get_doc(memory_id)
                
                if original is None:
                    original = read_text(document_path)
                    # 将文档内容存入缓存
                    if cache:
                        cache.set_doc(memory_id, original)
                
                # 添加到文档列表
                all_docs.append({
                    'memory_id': memory_id,
                    'memory_type': metadata.get('memory_type'),
                    'title': metadata.get('title', memory_id),
                    'path': relative_path,
                    'content': original,
                    'original_chars': len(original),
                })
            
            # 步骤1.2：根据 search_mode 选择检索方式
            vector_config = paths.get("vector_search", {})
            hybrid_config = paths.get("hybrid_search", {})
            
            if search_mode == "hybrid":
                # 混合检索模式（关键词检索 + 向量检索综合排序）
                ranked_docs = _search_hybrid(
                    query,
                    all_docs,
                    top_k=top_k,
                    similarity_threshold=vector_config.get("similarity_threshold", 0.3),
                    keyword_weight=hybrid_config.get("keyword_weight", 0.5),
                    vector_weight=hybrid_config.get("vector_weight", 0.5),
                )
            elif search_mode == "vector" and vector_config.get("enable", False):
                # 向量检索模式（使用 TF-IDF 实现，无需外部模型）
                ranked_docs = _search_by_vector(
                    query,
                    all_docs,
                    top_k=top_k,
                    similarity_threshold=vector_config.get("similarity_threshold", 0.3),
                )
            else:
                # 关键词检索模式（默认）
                ranked_docs = _search_by_keyword(query, all_docs, top_k=top_k)
            
            # 将检索结果存入缓存
            if cache:
                cache.set_search(query, search_mode, top_k, use_global_memory, ranked_docs)
        
        # 步骤3：处理排序结果（应用容量限制和截断）
        docs = []              # 最终返回的文档列表
        any_truncated = False  # 是否有文档被截断的标记
        remaining = int(paths['max_chars'])  # 剩余可用字符数
        
        # 按检索排序顺序依次处理文档
        for doc in ranked_docs:
            original = doc['content']
            
            # 根据剩余字符数截取内容（防止超出 LLM 上下文窗口）
            included = original[:remaining] if remaining > 0 else ''
            
            # 判断是否被截断
            truncated = len(included) < len(original)
            any_truncated = any_truncated or truncated
            
            # 如果截取后有内容，添加到结果列表
            if included:
                docs.append({
                    'memory_id': doc['memory_id'],
                    'memory_type': doc.get('memory_type'),
                    'title': doc.get('title', doc['memory_id']),
                    'path': doc.get('path', ''),
                    'content': included,
                    'original_chars': len(original),
                    'included_chars': len(included),
                    'truncated': truncated,
                    '_bm25_score': doc.get('_bm25_score'),
                    '_vector_score': doc.get('_vector_score'),
                    '_hybrid_score': doc.get('_hybrid_score'),
                })
                
                # 更新剩余字符数
                remaining -= len(included)
        
        # 步骤4：构造返回结果
        status = "success" if docs else "no_match"  # 有结果为 success，无结果为 no_match
        result = {
            'status': status,
            'query': query,
            'search_mode': search_mode,
            'selected_memory_docs': docs,
            'max_memory_chars': paths['max_chars'],
            'total_chars': sum(item['included_chars'] for item in docs),
            'truncated': any_truncated,
            'errors': [],
        }
        
        # 步骤5：保存输出文件（如果指定了输出目录）
        if outdir:
            output_dir = Path(outdir)
            # 保存选中的记忆结果
            write_json(result, output_dir / 'selected_memory.json')
            # 追加写入操作日志
            append_jsonl({
                'timestamp': now_iso(),
                'operation': 'load',
                'status': status,
                'query': query,
                'top_k': top_k,
                'selected_ids': [item['memory_id'] for item in docs],
            }, output_dir / 'memory_log.jsonl')
        
        return result
    # ==================== 关键词检索模式结束 ====================

    # ==================== 模式二：精确 ID 查找模式 ====================
    # 触发条件：未传入 query，或传入了 selected_memory_ids
    # 适用场景：已知记忆 ID，需要精确加载特定上下文
    
    # 步骤1：构建有序的记忆 ID 列表
    ordered_ids = []
    
    # 优先加载全局记忆（如果启用），按 memory_id 排序保证确定性
    if use_global_memory:
        ordered_ids.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    
    # 追加用户指定的记忆 ID
    ordered_ids.extend(selected_memory_ids)
    
    # 去重并保持顺序（使用 dict.fromkeys 实现有序去重）
    ordered_ids = list(dict.fromkeys(ordered_ids))

    # 步骤2：依次加载每个记忆文档
    docs = []              # 成功加载的文档列表
    errors = []            # 加载失败的错误列表
    remaining = int(paths["max_chars"])  # 剩余可用字符数
    any_truncated = False  # 是否有文档被截断的标记
    
    for memory_id in ordered_ids:
        # 获取记忆元数据
        metadata = index.get(memory_id)
        
        # 错误处理：memory_id 不存在
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        
        # 获取文档相对路径
        relative_path = metadata.get("path")
        
        # 错误处理：路径元数据缺失
        if not isinstance(relative_path, str):
            errors.append({"memory_id": memory_id, "type": "InvalidMetadata", "message": "memory path is missing"})
            continue
        
        # 构建文档绝对路径
        document_path = (paths["root"] / relative_path).resolve()
        
        # 安全检查：防止路径穿越攻击
        try:
            document_path.relative_to(paths["root"].resolve())
        except ValueError:
            errors.append({"memory_id": memory_id, "type": "InvalidPath", "message": "memory path escapes root"})
            continue
        
        # 错误处理：文件不存在
        if not document_path.is_file():
            errors.append({"memory_id": memory_id, "type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"})
            continue
        
        # 读取文档内容
        original = read_text(document_path)
        
        # 应用容量限制，截取内容
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        
        # 如果截取后有内容，添加到结果列表
        if included:
            docs.append(
                {
                    "memory_id": memory_id,
                    "memory_type": metadata.get("memory_type"),
                    "title": metadata.get("title", memory_id),
                    "path": relative_path,
                    "content": included,
                    "original_chars": len(original),
                    "included_chars": len(included),
                    "truncated": truncated,
                }
            )
            remaining -= len(included)
    
    # 步骤3：确定状态码
    if errors and docs:
        status = "partial"  # 部分成功（有错误但也有成功加载的文档）
    elif errors:
        status = "error"    # 全部失败（只有错误）
    else:
        status = "success"  # 全部成功
    
    # 步骤4：构造返回结果
    result = {
        "status": status,
        "query": query,
        "selected_memory_docs": docs,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": errors,
    }
    
    # 步骤5：保存输出文件（如果指定了输出目录）
    if outdir:
        output_dir = Path(outdir)
        # 保存选中的记忆结果
        write_json(result, output_dir / "selected_memory.json")
        # 追加写入操作日志
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "selected_ids": [item["memory_id"] for item in docs],
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result
    # ==================== 精确 ID 查找模式结束 ====================


def _safe_conversation_id(conversation_id: str) -> str:
    """
    验证 conversation_id 的安全性，防止路径注入攻击。

    安全规则：
        - 必须是字符串类型
        - 只能包含字母、数字、点（.）、下划线（_）、连字符（-）
        - 禁止使用路径分隔符（/、\）等特殊字符

    参数：
        conversation_id: 对话 ID 字符串

    返回：
        验证通过的 conversation_id（原样返回）

    异常：
        ValueError: conversation_id 包含非法字符
    """
    # 使用正则表达式验证 conversation_id 格式
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


# ==================== 记忆更新与合并模块 ====================

def _parse_memory_markdown(content: str) -> dict:
    """
    将记忆文档的 Markdown 内容解析为结构化的部分。

    记忆文档格式：
        # Conversation conv_001
        - memory_id: `mem_conversation_conv_001`
        - conversation_id: `conv_001`
        - created_or_updated_at: `2026-06-25T14:46:36+08:00`

        ## Final Answer
        Agent 会结合模型、工具和记忆完成任务。

        ## Messages
        ```json
        [...]
        ```

        ## Trace
        ```json
        [...]
        ```

    参数：
        content: 记忆文档的 Markdown 内容字符串

    返回：
        包含以下字段的字典：
            - "title": 标题（如 "Conversation conv_001"）
            - "metadata": 元数据字典（memory_id, conversation_id, created_or_updated_at）
            - "final_answer": 最终回答内容
            - "messages": 消息历史（解析后的 JSON 数组）
            - "trace": 执行轨迹（解析后的 JSON 对象）
            - "raw_final_answer": 原始 Final Answer 文本（未处理）
            - "raw_messages": 原始 Messages JSON 字符串
            - "raw_trace": 原始 Trace JSON 字符串
    """
    parsed = {
        "title": "",
        "metadata": {},
        "final_answer": "",
        "messages": [],
        "trace": {},
        "raw_final_answer": "",
        "raw_messages": "",
        "raw_trace": "",
    }

    # 提取标题（第一行以 # 开头）
    lines = content.split("\n")
    if lines and lines[0].startswith("# "):
        parsed["title"] = lines[0][2:].strip()

    # 使用正则表达式提取各个部分
    # 提取元数据部分
    meta_match = re.search(r"- memory_id: `([^`]+)`\n- conversation_id: `([^`]+)`\n- created_or_updated_at: `([^`]+)`", content)
    if meta_match:
        parsed["metadata"] = {
            "memory_id": meta_match.group(1),
            "conversation_id": meta_match.group(2),
            "created_or_updated_at": meta_match.group(3),
        }

    # 提取 Final Answer 部分
    fa_match = re.search(r"## Final Answer\n\n(.+?)\n\n## Messages", content, re.DOTALL)
    if fa_match:
        parsed["raw_final_answer"] = fa_match.group(1).strip()
        parsed["final_answer"] = parsed["raw_final_answer"]

    # 提取 Messages 部分（JSON 代码块）
    msg_match = re.search(r"## Messages\n\n```json\n(.+?)\n```", content, re.DOTALL)
    if msg_match:
        parsed["raw_messages"] = msg_match.group(1).strip()
        try:
            parsed["messages"] = json.loads(parsed["raw_messages"])
        except json.JSONDecodeError:
            parsed["messages"] = []

    # 提取 Trace 部分（JSON 代码块）
    trace_match = re.search(r"## Trace\n\n```json\n(.+?)\n```", content, re.DOTALL)
    if trace_match:
        parsed["raw_trace"] = trace_match.group(1).strip()
        try:
            parsed["trace"] = json.loads(parsed["raw_trace"])
        except json.JSONDecodeError:
            parsed["trace"] = {}

    return parsed


def _compare_sections(old_parsed: dict, new_parsed: dict) -> dict:
    """
    比较新旧文档的各部分差异，返回分类后的差异报告。

    差异分类：
        - duplicate: 重复内容（新旧相同）
        - supplementary: 补充内容（新文档新增，不冲突）
        - conflicting: 冲突内容（新旧矛盾）

    参数：
        old_parsed: 旧文档解析结果（_parse_memory_markdown 返回的字典）
        new_parsed: 新文档解析结果（_parse_memory_markdown 返回的字典）

    返回：
        差异报告字典，包含：
            - "final_answer": Final Answer 部分的差异分析
            - "messages": Messages 部分的差异分析
            - "trace": Trace 部分的差异分析
            - "summary": 总体差异摘要（duplicate_count, supplementary_count, conflicting_count）
    """
    diff = {
        "final_answer": {"duplicate": [], "supplementary": [], "conflicting": []},
        "messages": {"duplicate": [], "supplementary": [], "conflicting": []},
        "trace": {"duplicate": [], "supplementary": [], "conflicting": []},
        "summary": {"duplicate_count": 0, "supplementary_count": 0, "conflicting_count": 0},
    }

    # 比较 Final Answer（按句子比较）
    old_fa = old_parsed.get("final_answer", "")
    new_fa = new_parsed.get("final_answer", "")

    if old_fa and new_fa:
        old_sentences = [s.strip() for s in re.split(r"[。！？；\n]+", old_fa) if s.strip()]
        new_sentences = [s.strip() for s in re.split(r"[。！？；\n]+", new_fa) if s.strip()]

        for sentence in new_sentences:
            if sentence in old_sentences:
                diff["final_answer"]["duplicate"].append(sentence)
                diff["summary"]["duplicate_count"] += 1
            else:
                is_conflict = False
                for old_sent in old_sentences:
                    if _detect_conflict(old_sent, sentence):
                        diff["final_answer"]["conflicting"].append(
                            {"old": old_sent, "new": sentence}
                        )
                        diff["summary"]["conflicting_count"] += 1
                        is_conflict = True
                        break
                if not is_conflict:
                    diff["final_answer"]["supplementary"].append(sentence)
                    diff["summary"]["supplementary_count"] += 1

    # 比较 Messages（按消息比较）
    old_messages = old_parsed.get("messages", [])
    new_messages = new_parsed.get("messages", [])

    old_msg_hashes = {_hash_message(m) for m in old_messages}

    for msg in new_messages:
        msg_hash = _hash_message(msg)
        if msg_hash in old_msg_hashes:
            diff["messages"]["duplicate"].append(msg)
            diff["summary"]["duplicate_count"] += 1
        else:
            is_conflict = False
            for old_msg in old_messages:
                if _detect_conflict(json.dumps(old_msg), json.dumps(msg)):
                    diff["messages"]["conflicting"].append({"old": old_msg, "new": msg})
                    diff["summary"]["conflicting_count"] += 1
                    is_conflict = True
                    break
            if not is_conflict:
                diff["messages"]["supplementary"].append(msg)
                diff["summary"]["supplementary_count"] += 1

    # 比较 Trace（按键比较）
    old_trace = old_parsed.get("trace", {})
    new_trace = new_parsed.get("trace", {})

    for key, new_value in new_trace.items():
        if key in old_trace:
            old_value = old_trace[key]
            if old_value == new_value:
                diff["trace"]["duplicate"].append({"key": key, "value": new_value})
                diff["summary"]["duplicate_count"] += 1
            elif _detect_conflict(str(old_value), str(new_value)):
                diff["trace"]["conflicting"].append(
                    {"key": key, "old": old_value, "new": new_value}
                )
                diff["summary"]["conflicting_count"] += 1
            else:
                diff["trace"]["supplementary"].append({"key": key, "value": new_value})
                diff["summary"]["supplementary_count"] += 1
        else:
            diff["trace"]["supplementary"].append({"key": key, "value": new_value})
            diff["summary"]["supplementary_count"] += 1

    return diff


def _detect_conflict(old_text: str, new_text: str) -> bool:
    """
    检测两段文本是否存在冲突。

    冲突检测规则：
        1. 比较数值型内容（数字），如果数值不同则认为冲突
        2. 比较反义词对（如 "支持"/"不支持", "是"/"不是"）
        3. 比较日期、时间等关键信息

    参数：
        old_text: 旧文本
        new_text: 新文本

    返回：
        True 如果检测到冲突，False 否则
    """
    # 提取数字进行比较
    old_nums = re.findall(r"-?\d+\.?\d*", old_text)
    new_nums = re.findall(r"-?\d+\.?\d*", new_text)

    # 如果两边都有数字且数字不同，可能存在冲突
    if old_nums and new_nums:
        old_num_set = set(old_nums)
        new_num_set = set(new_nums)
        # 如果新数字集合与旧数字集合没有交集，可能是冲突
        if new_num_set and not (new_num_set & old_num_set):
            return True

    # 检查反义词对
    antonym_pairs = [
        ("是", "不是"), ("有", "没有"), ("能", "不能"), ("可以", "不可以"),
        ("支持", "不支持"), ("包含", "不包含"), ("需要", "不需要"),
        ("成功", "失败"), ("正确", "错误"), ("true", "false"), ("True", "False"),
    ]
    for positive, negative in antonym_pairs:
        if (positive in old_text and negative in new_text) or \
           (negative in old_text and positive in new_text):
            return True

    return False


def _hash_message(message: dict) -> str:
    """
    生成消息的哈希值，用于快速比较消息是否相同。

    参数：
        message: 消息字典（包含 role, content, tool_calls 等字段）

    返回：
        消息的哈希字符串
    """
    # 按 key 排序后序列化，确保相同内容的消息哈希相同
    sorted_msg = json.dumps(message, sort_keys=True, ensure_ascii=False)
    return str(hash(sorted_msg))


def _merge_sections(old_parsed: dict, new_parsed: dict, diff: dict, strategy: str) -> tuple[str, dict]:
    """
    根据合并策略将新旧文档合并，生成合并后的 Markdown 内容和合并报告。

    支持的合并策略：
        - prefer_new: 默认，新内容优先，冲突时使用新内容
        - prefer_old: 旧内容优先，冲突时使用旧内容
        - keep_both: 保留双方内容，在冲突处添加标记
        - manual: 仅生成差异报告，不进行实际合并

    参数：
        old_parsed: 旧文档解析结果
        new_parsed: 新文档解析结果
        diff: 差异报告（_compare_sections 返回的字典）
        strategy: 合并策略字符串

    返回：
        元组 (merged_content, merge_report)
            - merged_content: 合并后的 Markdown 内容
            - merge_report: 合并过程报告，记录每个部分的处理方式
    """
    merge_report = {
        "strategy": strategy,
        "final_answer": {"action": "", "details": []},
        "messages": {"action": "", "details": []},
        "trace": {"action": "", "details": []},
        "merged_sections": [],
    }

    # 如果是 manual 策略，直接返回旧内容（不合并）
    if strategy == "manual":
        return old_parsed.get("raw_final_answer", "") + "\n\n", merge_report

    # 合并 Final Answer
    merged_fa = []
    if strategy == "prefer_new":
        merged_fa.append(new_parsed.get("final_answer", ""))
        merge_report["final_answer"]["action"] = "replace_with_new"
        merge_report["final_answer"]["details"] = ["使用新的 Final Answer"]
    elif strategy == "prefer_old":
        merged_fa.append(old_parsed.get("final_answer", ""))
        merge_report["final_answer"]["action"] = "keep_old"
        merge_report["final_answer"]["details"] = ["保留旧的 Final Answer"]
    elif strategy == "keep_both":
        merged_fa.append("## 原内容\n")
        merged_fa.append(old_parsed.get("final_answer", ""))
        merged_fa.append("\n\n## 更新内容\n")
        merged_fa.append(new_parsed.get("final_answer", ""))
        if diff["final_answer"]["conflicting"]:
            merged_fa.append("\n\n## ⚠️ 冲突内容\n")
            for conflict in diff["final_answer"]["conflicting"]:
                merged_fa.append(f"- 原: {conflict['old']}\n")
                merged_fa.append(f"- 新: {conflict['new']}\n")
        merge_report["final_answer"]["action"] = "keep_both"
        merge_report["final_answer"]["details"] = [
            f"重复句子: {len(diff['final_answer']['duplicate'])}",
            f"补充句子: {len(diff['final_answer']['supplementary'])}",
            f"冲突句子: {len(diff['final_answer']['conflicting'])}",
        ]

    # 合并 Messages
    merged_messages = []
    if strategy == "prefer_new":
        merged_messages = new_parsed.get("messages", [])
        merge_report["messages"]["action"] = "replace_with_new"
        merge_report["messages"]["details"] = ["使用新的 Messages"]
    elif strategy == "prefer_old":
        merged_messages = old_parsed.get("messages", [])
        merge_report["messages"]["action"] = "keep_old"
        merge_report["messages"]["details"] = ["保留旧的 Messages"]
    elif strategy == "keep_both":
        old_messages = old_parsed.get("messages", [])
        new_messages = new_parsed.get("messages", [])
        old_hashes = {_hash_message(m) for m in old_messages}
        for msg in old_messages:
            merged_messages.append(msg)
        for msg in new_messages:
            if _hash_message(msg) not in old_hashes:
                merged_messages.append(msg)
        merge_report["messages"]["action"] = "merge_unique"
        merge_report["messages"]["details"] = [
            f"原消息数: {len(old_messages)}",
            f"新消息数: {len(new_messages)}",
            f"合并后消息数: {len(merged_messages)}",
        ]

    # 合并 Trace
    merged_trace = {}
    if strategy == "prefer_new":
        merged_trace = new_parsed.get("trace", {})
        merge_report["trace"]["action"] = "replace_with_new"
        merge_report["trace"]["details"] = ["使用新的 Trace"]
    elif strategy == "prefer_old":
        merged_trace = old_parsed.get("trace", {})
        merge_report["trace"]["action"] = "keep_old"
        merge_report["trace"]["details"] = ["保留旧的 Trace"]
    elif strategy == "keep_both":
        merged_trace = dict(old_parsed.get("trace", {}))
        merged_trace.update(new_parsed.get("trace", {}))
        merge_report["trace"]["action"] = "merge_keys"
        merge_report["trace"]["details"] = [
            f"原键数: {len(old_parsed.get('trace', {}))}",
            f"新键数: {len(new_parsed.get('trace', {}))}",
            f"合并后键数: {len(merged_trace)}",
        ]

    # 生成合并后的 Markdown 内容
    now = now_iso()
    memory_id = old_parsed.get("metadata", {}).get("memory_id", "")
    conversation_id = old_parsed.get("metadata", {}).get("conversation_id", "")
    title = old_parsed.get("title", "")

    merged_content = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{now}`\n\n"
        "## Final Answer\n\n"
        f"{''.join(merged_fa)}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(merged_messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(merged_trace, ensure_ascii=False, indent=2)}\n```\n"
    )

    merge_report["merged_sections"] = ["final_answer", "messages", "trace"]

    return merged_content, merge_report


def update_memory(
    config_path: str,
    memory_id: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    merge_strategy: str = "prefer_new",
    outdir: str | None = None,
) -> dict:
    """
    更新指定的记忆文档，支持差异对比和智能合并。

    更新流程：
        1. 根据 memory_id 查找旧文档
        2. 读取新的输入文件（messages、trace、answer）
        3. 解析新旧文档为结构化部分
        4. 比较差异（重复、补充、冲突）
        5. 根据策略合并文档
        6. 写入合并后的 Markdown 文件
        7. 更新索引（保留创建时间，更新时间戳，记录合并历史）

    参数：
        config_path: 配置文件路径（memory.yaml）
        memory_id: 要更新的记忆 ID
        messages_path: 新的消息历史文件路径
        trace_path: 新的执行轨迹文件路径
        answer_path: 新的最终回答文件路径
        merge_strategy: 合并策略（prefer_new/prefer_old/keep_both/manual，默认 prefer_new）
        outdir: 输出目录路径（可选，用于保存结果文件）

    返回：
        包含以下字段的字典：
            - "status": 状态（success / no_match / error）
            - "memory_id": 更新的记忆 ID
            - "merge_strategy": 使用的合并策略
            - "diff_report": 差异分析报告
            - "merge_report": 合并过程报告
            - "updated_at": 更新时间
            - "path": 记忆文档路径
            - "index_path": 索引文件路径
    """
    # 步骤1：验证合并策略
    valid_strategies = ["prefer_new", "prefer_old", "keep_both", "manual"]
    if merge_strategy not in valid_strategies:
        raise ValueError(f"merge_strategy must be one of: {', '.join(valid_strategies)}")

    # 步骤2：解析配置文件
    paths = _memory_paths(config_path)

    # 步骤3：读取索引，查找旧文档
    index = _read_index(paths["index"])
    if memory_id not in index:
        result = {
            "status": "no_match",
            "memory_id": memory_id,
            "message": f"memory_id {memory_id} does not exist",
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "updated_memory.json")
            append_jsonl(
                {"timestamp": now_iso(), "operation": "update", "status": "no_match", "memory_id": memory_id},
                output_dir / "memory_log.jsonl",
            )
        return result

    # 步骤4：获取旧文档路径
    old_metadata = index[memory_id]
    relative_path = old_metadata.get("path")
    if not relative_path:
        result = {
            "status": "error",
            "memory_id": memory_id,
            "message": "old memory path is missing",
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "updated_memory.json")
        return result

    document_path = (paths["root"] / relative_path).resolve()
    if not document_path.is_file():
        result = {
            "status": "error",
            "memory_id": memory_id,
            "message": f"old memory file not found: {relative_path}",
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "updated_memory.json")
        return result

    # 步骤5：读取旧文档内容
    old_content = read_text(document_path)
    old_parsed = _parse_memory_markdown(old_content)

    # 步骤6：读取新的输入文件
    new_messages = read_json(messages_path)
    new_trace = read_json(trace_path)
    new_answer = read_text(answer_path).strip()

    # 步骤7：构建新文档的解析结果（模拟新文档的解析格式）
    save_type = old_metadata.get("memory_type", "conversation")
    conversation_id = old_metadata.get("conversation_id", "")
    new_title = f"{save_type.title()} {conversation_id}"
    new_parsed = {
        "title": new_title,
        "metadata": {
            "memory_id": memory_id,
            "conversation_id": conversation_id,
            "created_or_updated_at": now_iso(),
        },
        "final_answer": new_answer,
        "messages": new_messages,
        "trace": new_trace,
        "raw_final_answer": new_answer,
        "raw_messages": json.dumps(new_messages, ensure_ascii=False, indent=2),
        "raw_trace": json.dumps(new_trace, ensure_ascii=False, indent=2),
    }

    # 步骤8：比较差异
    diff_report = _compare_sections(old_parsed, new_parsed)

    # 步骤9：执行合并
    merged_content, merge_report = _merge_sections(old_parsed, new_parsed, diff_report, merge_strategy)

    # 步骤10：写入合并后的文件
    write_text(merged_content, document_path)

    # 步骤11：更新索引
    now = now_iso()
    merge_history = old_metadata.get("merge_history", [])
    merge_history.append({
        "timestamp": now,
        "strategy": merge_strategy,
        "diff_summary": diff_report["summary"],
    })

    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": old_metadata.get("memory_type"),
        "title": new_title,
        "summary": new_answer[:200],
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": old_metadata.get("created_at", now),
        "updated_at": now,
        "merge_history": merge_history,
    }
    write_json(index, paths["index"])

    # 步骤12：构造返回结果
    result = {
        "status": "success",
        "memory_id": memory_id,
        "merge_strategy": merge_strategy,
        "diff_report": diff_report,
        "merge_report": merge_report,
        "updated_at": now,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
    }

    # 步骤13：保存输出文件
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "updated_memory.json")
        write_json(diff_report, output_dir / "memory_diff.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "update",
                "status": "success",
                "memory_id": memory_id,
                "merge_strategy": merge_strategy,
                "diff_summary": diff_report["summary"],
            },
            output_dir / "memory_log.jsonl",
        )

    # 步骤14：失效相关缓存
    cache = _get_cache(config_path)
    cache.invalidate_doc(memory_id)
    cache.invalidate_search()

    return result
# ==================== 记忆更新与合并模块结束 ====================


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    """
    保存记忆文档的核心函数，将对话历史、执行轨迹和最终回答整合为记忆文档。

    保存流程：
        1. 验证参数合法性（conversation_id、save_type）
        2. 读取三个输入文件（messages、trace、answer）
        3. 生成唯一的 memory_id 和存储路径
        4. 生成结构化的 Markdown 文档内容
        5. 写入 Markdown 文件（原子操作）
        6. 更新 memory_index.json 索引
        7. 记录操作日志

    参数：
        config_path: 配置文件路径（memory.yaml）
        conversation_id: 对话 ID（用于生成唯一的记忆 ID 和文件名）
        save_type: 记忆类型（"conversation" 对话记忆 / "global" 全局记忆）
        messages_path: 消息历史文件路径（messages.json）
        trace_path: 执行轨迹文件路径（trace.json）
        answer_path: 最终回答文件路径（final_answer.md）
        outdir: 输出目录路径（可选，用于保存结果文件）

    返回：
        包含以下字段的字典：
            - "status": 状态（success）
            - "memory_id": 生成的唯一记忆 ID
            - "memory_type": 记忆类型
            - "conversation_id": 对话 ID
            - "title": 记忆标题
            - "summary": 记忆摘要（取 answer 的前 200 字符）
            - "path": 记忆文档的相对路径
            - "index_path": 索引文件名
            - "created_at": 创建时间（首次保存）或更新时间（覆盖保存）
            - "updated_at": 更新时间
            - "source_paths": 源文件路径（messages、trace、answer）

    输出文件（当 outdir 不为 None 时）：
        - saved_memory.json: 保存结果元数据
        - memory_log.jsonl: 操作日志（追加写入）

    记忆文档格式（Markdown）：
        # Conversation conv_001
        - memory_id: `mem_conversation_conv_001`
        - conversation_id: `conv_001`
        - created_or_updated_at: `2026-06-25T14:46:36+08:00`

        ## Final Answer
        Agent 会结合模型、工具和记忆完成任务。

        ## Messages
        ```json
        [...]
        ```

        ## Trace
        ```json
        [...]
        ```
    """
    # 步骤1：验证 conversation_id 的安全性（防止路径注入攻击）
    conversation_id = _safe_conversation_id(conversation_id)
    
    # 步骤2：验证 save_type 必须是 conversation 或 global
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    
    # 步骤3：解析配置文件，获取路径信息
    paths = _memory_paths(config_path)
    
    # 步骤4：读取三个输入文件
    messages = read_json(messages_path)  # 消息历史（JSON 数组）
    trace = read_json(trace_path)        # 执行轨迹（JSON 对象）
    answer = read_text(answer_path).strip()  # 最终回答（文本）
    
    # 步骤5：验证输入文件格式
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    
    # 步骤5.5：应用内容压缩（如果启用）
    compression_config = paths.get("compression", {})
    compression_info = {}
    if compression_config.get("enable", False):
        max_answer_chars = compression_config.get("max_answer_chars", 2000)
        max_messages_chars = compression_config.get("max_messages_chars", 3000)
        summary_length = compression_config.get("summary_length", 200)
        
        # 压缩最终回答
        answer, answer_compressed = _compress_answer(answer, max_answer_chars, summary_length)
        if answer_compressed:
            compression_info["answer"] = {
                "action": "compressed",
                "original_length": len(read_text(answer_path).strip()),
                "compressed_length": len(answer),
            }
        
        # 压缩消息历史
        messages, messages_compressed = _compress_messages(messages, max_messages_chars, summary_length)
        if messages_compressed:
            compression_info["messages"] = {
                "action": "compressed",
                "original_count": len(read_json(messages_path)),
                "compressed_count": len(messages),
            }
    
    # 步骤6：生成时间戳和唯一标识符
    now = now_iso()  # 当前时间（ISO 8601 格式）
    
    # 生成唯一的 memory_id：mem_{type}_{conversation_id}
    memory_id = f"mem_{save_type}_{conversation_id}"
    
    # 确定存储目录和相对路径
    if save_type == "conversation":
        target_dir = paths["conversations"]  # 对话记忆目录
        relative_dir = "conversations"       # 相对目录名
    else:
        target_dir = paths["global"]         # 全局记忆目录
        relative_dir = "global"              # 相对目录名
    
    # 构建目标文件路径
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    
    # 步骤7：生成记忆标题和摘要
    title = f"{save_type.title()} {conversation_id}"  # 首字母大写的类型 + 对话 ID
    summary = answer[:200]  # 摘要取最终回答的前 200 字符
    
    # 步骤8：生成 Markdown 文档内容
    markdown = (
        f"# {title}\n\n"                                    # 标题
        f"- memory_id: `{memory_id}`\n"                     # 唯一记忆 ID
        f"- conversation_id: `{conversation_id}`\n"          # 对话 ID
        f"- created_or_updated_at: `{now}`\n\n"              # 创建/更新时间
        "## Final Answer\n\n"                               # 最终回答标题
        f"{answer}\n\n"                                     # 最终回答内容
        "## Messages\n\n```json\n"                          # 消息历史标题（JSON 代码块）
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"  # 消息内容
        "## Trace\n\n```json\n"                             # 执行轨迹标题（JSON 代码块）
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"       # 轨迹内容
    )
    
    # 步骤9：写入 Markdown 文件（原子操作，防止文件损坏）
    write_text(markdown, target_path)
    
    # 步骤10：更新记忆索引
    index = _read_index(paths["index"])
    
    # 获取已存在的创建时间（如果是更新操作，保留原创建时间）
    existing = index.get(memory_id, {})
    created_at = existing.get("created_at", now)
    
    # 更新索引条目
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,       # 首次保存为 now，更新时保留原值
        "updated_at": now,              # 更新时间始终为当前时间
    }
    
    # 写入索引文件（原子操作）
    write_json(index, paths["index"])
    
    # 步骤11：构造返回结果
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
        "compression": compression_info,
    }
    
    # 步骤12：保存输出文件（如果指定了输出目录）
    if outdir:
        output_dir = Path(outdir)
        # 保存保存结果
        write_json(result, output_dir / "saved_memory.json")
        # 追加写入操作日志
        append_jsonl(
            {"timestamp": now, "operation": "save", "status": "success", "memory_id": memory_id},
            output_dir / "memory_log.jsonl",
        )
    
    # 步骤13：失效相关缓存
    cache = _get_cache(config_path)
    cache.invalidate_doc(memory_id)
    cache.invalidate_search()
    
    return result


def parse_bool(value: str) -> bool:
    """
    命令行参数布尔值解析函数，支持多种表示方式。

    支持的真值："true"、"1"、"yes"（不区分大小写）
    支持的假值："false"、"0"、"no"（不区分大小写）

    参数：
        value: 字符串形式的布尔值

    返回：
        解析后的布尔值（True/False）

    异常：
        argparse.ArgumentTypeError: 无法解析的布尔值
    """
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器，定义 B5 模块的 CLI 接口。

    支持的命令行参数：
        --config: 配置文件路径（必需）
        --select_memory_ids: 指定的记忆 ID 列表（精确查找模式）
        --use_global_memory: 是否使用全局记忆（True/False）
        --query: 用户查询字符串（检索模式）
        --top_k: 检索返回的前 K 个结果（默认3）
        --search_mode: 检索模式（keyword/vector，默认 keyword）
        --save_type: 保存类型（conversation/global）
        --save_input_path: 保存输入配置文件路径
        --update_memory_id: 要更新的记忆 ID（更新模式）
        --merge_strategy: 合并策略（prefer_new/prefer_old/keep_both/manual，默认 prefer_new）
        --clean: 清理过期记忆（清理模式）
        --batch_save: 批量保存输入配置文件路径（批量保存模式）
        --batch_delete: 要批量删除的记忆 ID 列表（批量删除模式）
        --list: 列出所有记忆文档（列表模式）
        --list_type: 列表模式下过滤记忆类型（conversation/global）
        --outdir: 输出目录路径（必需）

    九种运行模式：
        1. 精确 ID 查找：--select_memory_ids [ID1] [ID2] --use_global_memory true
        2. 关键词检索：--query "查询词" --top_k 3 --search_mode keyword
        3. 向量检索：--query "查询词" --top_k 3 --search_mode vector
        4. 混合检索：--query "查询词" --top_k 3 --search_mode hybrid
        5. 保存记忆：--save_type conversation --save_input_path input.json
        6. 更新记忆：--update_memory_id mem_xxx --save_input_path input.json --merge_strategy prefer_new
        7. 清理记忆：--clean
        8. 批量保存：--batch_save batch_input.json
        9. 批量删除：--batch_delete mem_xxx mem_yyy mem_zzz
        10. 列表记忆：--list [--list_type conversation]

    返回：
        argparse.ArgumentParser 对象
    """
    parser = argparse.ArgumentParser(description="Select, save or update local memory documents.")
    
    # 必需参数：配置文件路径
    parser.add_argument("--config", required=True)
    
    # 精确查找模式参数
    parser.add_argument("--select_memory_ids", nargs="*", help="指定要加载的记忆 ID 列表（精确查找模式）")
    parser.add_argument("--use_global_memory", type=parse_bool, help="是否加载全局记忆（True/False）")
    
    # 检索模式参数
    parser.add_argument("--query", help="用户查询字符串（检索模式）")
    parser.add_argument("--top_k", type=int, default=3, help="检索返回的前K个结果（默认3）")
    parser.add_argument("--search_mode", choices=["keyword", "vector", "hybrid"], default="keyword", 
                        help="检索模式：keyword(关键词检索)/vector(向量语义检索)/hybrid(混合检索)，默认 keyword")
    
    # 保存模式参数
    parser.add_argument("--save_type", choices=["conversation", "global"], help="记忆保存类型")
    parser.add_argument("--save_input_path", help="保存输入配置文件路径")
    
    # 更新模式参数
    parser.add_argument("--update_memory_id", help="要更新的记忆 ID（更新模式）")
    parser.add_argument("--merge_strategy", choices=["prefer_new", "prefer_old", "keep_both", "manual"], 
                        default="prefer_new", help="合并策略：prefer_new(新内容优先)/prefer_old(旧内容优先)/keep_both(保留双方)/manual(仅生成差异报告)")
    
    # 清理模式参数
    parser.add_argument("--clean", action="store_true", help="清理过期记忆（清理模式）")
    
    # 批量操作模式参数
    parser.add_argument("--batch_save", help="批量保存输入配置文件路径（批量保存模式）")
    parser.add_argument("--batch_delete", nargs="*", help="要批量删除的记忆 ID 列表（批量删除模式）")
    parser.add_argument("--list", action="store_true", help="列出所有记忆文档（列表模式）")
    parser.add_argument("--list_type", choices=["conversation", "global"], help="列表模式下过滤记忆类型")
    
    # 缓存操作参数
    parser.add_argument("--clear_cache", action="store_true", help="清除所有缓存（缓存操作）")
    parser.add_argument("--cache_stats", action="store_true", help="获取缓存统计信息（缓存操作）")
    
    # 必需参数：输出目录
    parser.add_argument("--outdir", required=True)
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    B5 模块的命令行入口函数，负责解析参数并调用对应的核心函数。

    执行流程：
        1. 解析命令行参数
        2. 根据参数判断运行模式（更新模式 / 保存模式 / 加载模式）
        3. 调用对应的核心函数（update_memory / save_memory / load_memory）
        4. 输出结果文件路径或错误信息

    参数：
        argv: 命令行参数列表（默认为 None，即使用 sys.argv）

    返回：
        退出码：0 表示成功，1 表示致命错误
    """
    # 解析命令行参数
    args = build_parser().parse_args(argv)
    
    try:
        # 将相对路径解析为绝对路径
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        
        # ==================== 清理模式 ====================
        # 触发条件：指定了 --clean
        if args.clean:
            # 调用清理记忆函数
            result = clean_memory(str(config_path), str(outdir))
            
            # 输出结果文件路径
            print(outdir / "cleaned_memories.json")
        
        # ==================== 批量保存模式 ====================
        # 触发条件：指定了 --batch_save
        elif args.batch_save:
            # 解析批量保存输入配置文件路径
            input_path = resolve_cli_path(args.batch_save)
            payload = read_json(input_path)
            
            # 获取配置文件所在目录（用于解析相对路径）
            base = input_path.parent
            
            # 构建批量输入列表
            batch_input = []
            for item in payload.get("batch", []):
                batch_input.append({
                    "conversation_id": item["conversation_id"],
                    "save_type": item.get("save_type", "conversation"),
                    "messages_path": str((base / item["messages_path"]).resolve()),
                    "trace_path": str((base / item["trace_path"]).resolve()),
                    "answer_path": str((base / item["answer_path"]).resolve()),
                })
            
            # 调用批量保存记忆函数
            result = batch_save_memory(str(config_path), batch_input, str(outdir))
            
            # 输出结果文件路径
            print(outdir / "batch_saved_memory.json")
        
        # ==================== 批量删除模式 ====================
        # 触发条件：指定了 --batch_delete
        elif args.batch_delete:
            # 调用批量删除记忆函数
            result = batch_delete_memory(str(config_path), args.batch_delete, str(outdir))
            
            # 输出结果文件路径
            print(outdir / "batch_deleted_memory.json")
        
        # ==================== 列表模式 ====================
        # 触发条件：指定了 --list
        elif args.list:
            # 调用列表记忆函数
            result = list_memory(str(config_path), args.list_type, str(outdir))
            
            # 输出结果文件路径
            print(outdir / "listed_memory.json")
        
        # ==================== 缓存操作模式 ====================
        # 触发条件：指定了 --clear_cache 或 --cache_stats
        elif args.clear_cache:
            # 清除所有缓存
            clear_cache(str(config_path))
            result = {"status": "success", "message": "cache cleared"}
            
            # 输出结果文件路径
            write_json(result, outdir / "cache_result.json")
            print(outdir / "cache_result.json")
        
        elif args.cache_stats:
            # 获取缓存统计信息
            result = get_cache_stats(str(config_path))
            
            # 输出结果文件路径
            write_json(result, outdir / "cache_stats.json")
            print(outdir / "cache_stats.json")
        
        # ==================== 更新模式 ====================
        # 触发条件：指定了 --update_memory_id
        elif args.update_memory_id:
            # 参数校验：--save_input_path 必须提供
            if not args.save_input_path:
                raise ValueError("--update_memory_id requires --save_input_path")
            
            # 解析保存输入配置文件路径
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            
            # 获取配置文件所在目录（用于解析相对路径）
            base = input_path.parent
            
            # 调用更新记忆函数
            result = update_memory(
                str(config_path),
                args.update_memory_id,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                args.merge_strategy,
                str(outdir),
            )
            
            # 输出结果文件路径
            print(outdir / "updated_memory.json")
        
        # ==================== 保存模式 ====================
        # 触发条件：指定了 --save_type 或 --save_input_path
        elif args.save_type or args.save_input_path:
            # 参数校验：--save_type 和 --save_input_path 必须同时提供
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            
            # 解析保存输入配置文件路径
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            
            # 参数校验：CLI 指定的 save_type 必须与配置文件中的一致
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            
            # 获取配置文件所在目录（用于解析相对路径）
            base = input_path.parent
            
            # 调用保存记忆函数
            result = save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            
            # 输出结果文件路径
            print(outdir / "saved_memory.json")
        
        # ==================== 加载模式 ====================
        # 触发条件：未指定保存模式和更新模式参数
        else:
            # 参数校验：必须指定 --select_memory_ids 或 --use_global_memory
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            
            # 调用加载记忆函数
            result = load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                args.top_k,
                str(outdir),
                args.search_mode,
            )
            
            # 输出结果文件路径
            print(outdir / "selected_memory.json")
        
        # 成功退出
        return 0
    
    # 异常处理：捕获所有异常并输出错误信息
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# 脚本入口：当作为主程序运行时调用 main 函数
if __name__ == "__main__":
    raise SystemExit(main())
