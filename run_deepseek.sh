#!/bin/bash
#SBATCH --job-name=tpp_llm_deepseek-7b
#SBATCH --output=logs/tpp_llm_deepseek-7b_%j.out
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

## 1. 清除可能冲突的模块
module unload cuda 2>/dev/null || true

## 2. 加载环境+初始化conda
module load miniforge3/25.11.0-1
module load cuda/12.8
eval "$(conda shell.bash hook)"

## 3. 激活conda环境
conda activate TTP-LLM 

## 4. 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## 5. 设置Python路径
export PYTHONPATH=/data/run01/scxi244/EVENT/eventsfm/code/cy/TPP-LLM/src:$PYTHONPATH
export TQDM_DISABLE=1

## 6. 定义要使用的seed数组
declare -a seeds=(375 605 4967 8664 9231)

## 7. 定义数据集配置数组
# 格式: "数据集名称:数据路径:事件类型数量:训练batch_size:评估batch_size"
declare -a datasets=(
    "taxi:/data/run01/scxi244/EVENT/eventsfm/data/taxi:8:8:16"
    "taobao:/data/run01/scxi244/EVENT/eventsfm/data/taobao:9:4:8"
    "amazon:/data/run01/scxi244/EVENT/eventsfm/data/amazon:18:4:8"
    "retweet:/data/run01/scxi244/EVENT/eventsfm/data/retweet:9:4:8"
)

## 8. 模型配置
MODEL_NAME="deepseek-7b"
MODEL_PATH="/data/run01/scxi244/EVENT/LLM/deepseek-7b/"

## 9. 外层循环：遍历每个seed
for seed in "${seeds[@]}"; do
    echo ""
    echo "##################################################"
    echo "开始使用SEED: $seed 进行训练"
    echo "使用模型: $MODEL_NAME"
    echo "模型路径: $MODEL_PATH"
    echo "##################################################"
    echo ""

    ## 10. 内层循环：遍历每个数据集
    for dataset_config in "${datasets[@]}"; do
        # 解析配置
        IFS=':' read -r dataset_name data_path num_event_types train_bs eval_bs <<< "$dataset_config"
        
        echo ""
        echo "=========================================="
        echo "SEED: $seed | 开始训练数据集: $dataset_name"
        echo "数据路径: $data_path"
        echo "事件类型数量: $num_event_types"
        echo "训练batch size: $train_bs"
        echo "评估batch size: $eval_bs"
        echo "=========================================="
        echo ""
        
        # 检查数据路径是否存在
        if [ ! -d "$data_path" ]; then
            echo "警告: SEED: $seed | 数据路径不存在: $data_path，跳过该数据集"
            continue
        fi
        
        # 执行训练
        python /data/run01/scxi244/EVENT/eventsfm/code/cy/TPP-LLM/scripts/train_tpp_llm.py \
          --model_path "$MODEL_PATH" \
          --data_path "$data_path" \
          --num_event_types "$num_event_types" \
          --temporal_emb_type positional \
          --temporal_emb_first \
          --num_integral_samples 20 \
          --quant_type 4bit \
          --peft_type lora \
          --lora_rank 16 \
          --lora_modules q_proj k_proj v_proj o_proj \
          --train_batch_size "$train_bs" \
          --eval_batch_size "$eval_bs" \
          --learning_rate 1e-4 \
          --lr_scheduler_type constant \
          --num_train_epochs 20 \
          --warmup_ratio 0 \
          --beta_type 1 \
          --beta_time 1 \
          --device cuda \
          --seed $seed
        
        # 检查训练是否成功
        TRAIN_EXIT_CODE=$?
        if [ $TRAIN_EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✓ SEED: $seed | 数据集 $dataset_name 训练完成"
        else
            echo ""
            echo "✗ SEED: $seed | 数据集 $dataset_name 训练失败，退出码: $TRAIN_EXIT_CODE"
        fi
        
        # 训练后清理GPU内存
        cleanup_gpu
        
        echo ""
        echo "SEED: $seed | 等待 10 秒后继续下一个数据集..."
        sleep 10
    done
    
    echo ""
    echo "=========================================="
    echo "SEED: $seed | 所有数据集训练完成！"
    echo "=========================================="
    echo ""
    echo "等待 30 秒后开始下一个SEED的训练..."
    sleep 30
done

echo ""
echo "=========================================="
echo "所有SEED的训练任务全部完成！"
echo "=========================================="