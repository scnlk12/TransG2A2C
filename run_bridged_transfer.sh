#!/bin/bash
# BridgedSTGNN跨域迁移训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p saved_models
mkdir -p logs
mkdir -p results

echo "========================================="
echo "BridgedSTGNN 跨域流量迁移训练"
echo "========================================="

# ========== 配置参数 ==========
SOURCE_DATASET="PEMS07"
TARGET_DATASET="PEMS03"
BATCH_SIZE=64
AKR_EPOCHS=100
GKT_EPOCHS=50
DEVICE="cuda:0"

# ========== 检查数据 ==========
echo ""
echo "1. 检查数据集..."
if [ ! -f "data/${SOURCE_DATASET}/${SOURCE_DATASET}.npz" ]; then
    echo "❌ 错误: 源域数据集 ${SOURCE_DATASET} 不存在!"
    echo "请确保数据在: data/${SOURCE_DATASET}/${SOURCE_DATASET}.npz"
    exit 1
fi

if [ ! -f "data/${TARGET_DATASET}/${TARGET_DATASET}.npz" ]; then
    echo "❌ 错误: 目标域数据集 ${TARGET_DATASET} 不存在!"
    echo "请确保数据在: data/${TARGET_DATASET}/${TARGET_DATASET}.npz"
    exit 1
fi

echo "✓ 数据集检查通过:"
echo "  - 源域: ${SOURCE_DATASET}"
echo "  - 目标域: ${TARGET_DATASET}"

# ========== 检查/训练源域模型 ==========
echo ""
echo "2. 检查源域预训练模型..."
SOURCE_MODEL_PATH="saved_models/${SOURCE_DATASET}_gman_best.pth"

if [ ! -f "${SOURCE_MODEL_PATH}" ]; then
    echo "⚠️  未找到源域预训练模型，将从头训练..."
    echo "提示: 如果已有预训练模型，请放置在: ${SOURCE_MODEL_PATH}"

    read -p "是否现在训练源域模型? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "开始训练源域模型..."
        python train.py \
            --traffic_file data/${SOURCE_DATASET}/${SOURCE_DATASET}.npz \
            --batch_size ${BATCH_SIZE} \
            --max_epoch 200 \
            --patience 50 \
            --learning_rate 0.001 \
            --loss_func masked_mae \
            --save_model ${SOURCE_MODEL_PATH}

        if [ $? -ne 0 ]; then
            echo "❌ 源域模型训练失败!"
            exit 1
        fi
        echo "✓ 源域模型训练完成: ${SOURCE_MODEL_PATH}"
    else
        echo "⚠️  跳过源域训练，将使用随机初始化的源域编码器 (不推荐)"
        SOURCE_MODEL_PATH=""
    fi
else
    echo "✓ 找到源域预训练模型: ${SOURCE_MODEL_PATH}"
fi

# ========== 跨域迁移训练 ==========
echo ""
echo "3. 开始跨域迁移训练..."
echo "========================================="
echo "配置信息:"
echo "  - 源域: ${SOURCE_DATASET}"
echo "  - 目标域: ${TARGET_DATASET}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - AKR Epochs: ${AKR_EPOCHS}"
echo "  - GKT Epochs: ${GKT_EPOCHS}"
echo "  - 设备: ${DEVICE}"
echo "========================================="

python train_cross_domain.py \
    --source_dataset ${SOURCE_DATASET} \
    --target_dataset ${TARGET_DATASET} \
    --source_model_path ${SOURCE_MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --akr_epochs ${AKR_EPOCHS} \
    --gkt_epochs ${GKT_EPOCHS} \
    --embed_dim 128 \
    --topk 8 \
    --use_advanced_sampler \
    --use_cross_domain_contrast \
    --akr_lr 0.001 \
    --gkt_lr 0.0005 \
    --device ${DEVICE} \
    --save_dir saved_models \
    --log_dir logs \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ 跨域迁移训练失败!"
    exit 1
fi

echo ""
echo "✓ 跨域迁移训练完成!"

# ========== 测试评估 ==========
echo ""
echo "4. 在测试集上评估..."

python test.py \
    --model_path saved_models/bridged_${SOURCE_DATASET}_to_${TARGET_DATASET}_best.pth \
    --test_dataset ${TARGET_DATASET} \
    --metrics RMSE MAE MAPE

if [ $? -ne 0 ]; then
    echo "⚠️  测试评估失败，请检查模型路径"
else
    echo "✓ 测试评估完成!"
fi

# ========== 生成报告 ==========
echo ""
echo "5. 生成实验报告..."

REPORT_DIR="results/${SOURCE_DATASET}_to_${TARGET_DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${REPORT_DIR}

# 复制日志和模型
cp -r logs/* ${REPORT_DIR}/
cp saved_models/bridged_${SOURCE_DATASET}_to_${TARGET_DATASET}_*.pth ${REPORT_DIR}/

echo "✓ 实验报告已保存至: ${REPORT_DIR}"

# ========== 完成 ==========
echo ""
echo "========================================="
echo "✨ 所有流程完成! ✨"
echo "========================================="
echo "结果路径:"
echo "  - 模型: saved_models/"
echo "  - 日志: logs/"
echo "  - 报告: ${REPORT_DIR}"
echo ""
echo "下一步建议:"
echo "  1. 查看训练曲线: tensorboard --logdir=logs"
echo "  2. 可视化embedding: python scripts/visualize_embeddings.py"
echo "  3. 分析迁移效果: jupyter notebook analysis.ipynb"
echo "========================================="