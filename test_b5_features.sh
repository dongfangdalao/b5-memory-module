#!/bin/bash
# B5模块功能测试脚本

echo "========================================"
echo "  B5模块功能测试"
echo "========================================"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE_DIR="$SCRIPT_DIR/src"
CONFIG_PATH="$SCRIPT_DIR/configs/memory.yaml"
OUTPUT_DIR="$SCRIPT_DIR/outputs/B5_memory/test_features"

mkdir -p "$OUTPUT_DIR"

# 准备测试数据
echo ""
echo "[0] 准备测试数据..."

python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --save_type conversation \
    --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_search_test_data_01.json" \
    --outdir "$OUTPUT_DIR/prepare" 2>/dev/null

python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --save_type conversation \
    --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_search_test_data_02.json" \
    --outdir "$OUTPUT_DIR/prepare" 2>/dev/null

python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --save_type conversation \
    --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_search_test_data_03.json" \
    --outdir "$OUTPUT_DIR/prepare" 2>/dev/null

echo "  测试数据已准备"

echo ""
echo "[1] 关键词检索（检索词：数据分析）..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode keyword \
    --use_global_memory true \
    --outdir "$OUTPUT_DIR/keyword" 2>/dev/null
python -c "
import json, os
result_path = os.path.join('$OUTPUT_DIR', 'keyword', 'selected_memory.json')
if os.path.exists(result_path):
    data = json.load(open(result_path))
    docs = data.get('selected_memory_docs', [])
    for i, doc in enumerate(docs[:3], 1):
        score = doc.get('_bm25_score', 'N/A')
        print('    %d. %s (BM25: %s)' % (i, doc['memory_id'], score))
else:
    print('    无结果')
"

echo ""
echo "[2] 向量检索（检索词：数据分析）..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode vector \
    --use_global_memory true \
    --outdir "$OUTPUT_DIR/vector" 2>/dev/null
python -c "
import json, os
result_path = os.path.join('$OUTPUT_DIR', 'vector', 'selected_memory.json')
if os.path.exists(result_path):
    data = json.load(open(result_path))
    docs = data.get('selected_memory_docs', [])
    for i, doc in enumerate(docs[:3], 1):
        score = doc.get('_vector_score', 0)
        print('    %d. %s (相似度: %.4f)' % (i, doc['memory_id'], score))
else:
    print('    无结果')
"

echo ""
echo "[3] 混合检索（检索词：数据分析）..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --query "数据分析" \
    --top_k 3 \
    --search_mode hybrid \
    --use_global_memory true \
    --outdir "$OUTPUT_DIR/hybrid" 2>/dev/null
python -c "
import json, os
result_path = os.path.join('$OUTPUT_DIR', 'hybrid', 'selected_memory.json')
if os.path.exists(result_path):
    data = json.load(open(result_path))
    docs = data.get('selected_memory_docs', [])
    for i, doc in enumerate(docs[:3], 1):
        bm25 = doc.get('_bm25_score', 0)
        vector = doc.get('_vector_score', 0)
        hybrid = doc.get('_hybrid_score', 0)
        print('    %d. %s (BM25: %.4f, 向量: %.4f, 综合: %.4f)' % (i, doc['memory_id'], bm25, vector, hybrid))
else:
    print('    无结果')
"

echo ""
echo "[4] 文档长度优化（压缩）..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --save_type conversation \
    --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_save_compress_test.json" \
    --outdir "$OUTPUT_DIR/compress" 2>/dev/null
[ -d "$SCRIPT_DIR/memory/conversations" ] && echo "  ✅ 通过" || echo "  ❌ 失败"

echo ""
echo "[5] 记忆更新与合并..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --save_type conversation \
    --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_save_update_test.json" \
    --outdir "$OUTPUT_DIR/update_save" 2>/dev/null
MEMORY_ID=$(python -c "
import json,os
index_path = os.path.join('$SCRIPT_DIR','memory','memory_index.json')
if os.path.exists(index_path):
    index = json.load(open(index_path))
    for m in index:
        if 'conv_update_test' in m:
            print(m)
            break
")
if [ -n "$MEMORY_ID" ]; then
    python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
        --update_memory_id "$MEMORY_ID" \
        --save_input_path "$SCRIPT_DIR/data/test_inputs/memory_update_input_new.json" \
        --outdir "$OUTPUT_DIR/update" 2>/dev/null
    [ -f "$OUTPUT_DIR/update/updated_memory.json" ] && echo "  ✅ 通过" || echo "  ❌ 失败"
else
    echo "  ❌ 失败"
fi

echo ""
echo "[6] 批量操作..."
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --batch_save "$SCRIPT_DIR/data/test_inputs/memory_batch_test_input.json" \
    --outdir "$OUTPUT_DIR/batch" 2>&1 >/dev/null
python "$CODE_DIR/b5_memory/__init__.py" --config "$CONFIG_PATH" \
    --list \
    --outdir "$OUTPUT_DIR/list" 2>&1 >/dev/null
[ -f "$OUTPUT_DIR/batch/batch_saved_memory.json" -a -f "$OUTPUT_DIR/list/listed_memory.json" ] && echo "  ✅ 通过" || echo "  ❌ 失败"

echo ""
echo "========================================"
echo "  测试完成"
echo "========================================"