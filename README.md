# HomeBench 复现与微调指南

Reproduction of HomeBench,https://github.com/BITHLP/HomeBench/tree/main

本项目包含 HomeBench 论文代码的修复版本，支持 Qwen2.5 等模型的监督微调 (SFT)、分布式推理测试以及指标评估。

请确保您位于项目的**根目录**下执行所有指令 (例如 `/root/autodl-tmp/HomeBenchProduction`)。

## 1. 环境准备

确保已安装必要的依赖库，并正确设置了 HF 镜像（代码中已内置 `HF_ENDPOINT` 设置）。

核心文件结构说明：

```
HomeBenchReproduction/
├── code/
│   ├── sft_fixed.py        # 修复版微调脚本 (解决了 Loss=0 和 Processor 报错问题)
│   ├── model_test_fixed.py # 修复版推理脚本 (支持 DDP 多卡并行，支持 Zero-shot/Few-shot/RAG/SFT)
│   ├── eval_fixed.py       # 修复版评估脚本 (支持多种错误类型分析)
│   └── rag_dataset_generation.py # RAG 向量检索数据集生成脚本
├── dataset/                # 数据集存放目录
├── models/                 # 原始模型权重目录
├── model_output/           # (自动生成) 微调后的 LoRA 权重保存目录
└── output/                 # (自动生成) 推理结果和评估报告保存目录
```

## 2. 第一步：监督微调 (SFT)

使用 `code/sft_fixed.py` 对基座模型进行 LoRA 微调。

### 运行指令

```
# 单卡训练
torchrun --nproc_per_node=1 code/sft_fixed.py --model_name qwen --batch_size 4 --grad_accum 16

# 多卡训练 (DDP)
torchrun --nproc_per_node=2 code/sft_fixed.py --model_name qwen --batch_size 4 --grad_accum 16
```

## 3. 第二步：模型推理测试

使用 `code/model_test_fixed.py` 进行推理。该脚本通过不同参数支持所有核心实验模式。

### 实验 A：监督微调模型测试 (SFT, Main Result)

评估经过 SFT 训练后的模型性能。通常使用 Zero-shot 提示（因为模型已内化规则）。

**关键参数：** `--use_finetuned`

```
torchrun --nproc_per_node=1 code/model_test_fixed.py \
    --model_name qwen \
    --use_finetuned \
    --batch_size 16 \
    --cuda_devices "0"
```

- **输出文件**：`output/qwen_sft_zero_shot_test_result.json`

### 实验 B：原始基座模型基线 (Baselines)

评估未经过微调的原始模型能力。**不要**添加 `--use_finetuned` 参数。

#### 1. Zero-shot Baseline (基于提示词的测试)

最基础的基线，直接询问模型。

```
torchrun --nproc_per_node=1 code/model_test_fixed.py \
    --model_name qwen \
    --batch_size 16 \
    --cuda_devices "0"
```

- **输出文件**：`output/qwen_zero_shot_test_result.json`

#### 2. Few-shot Baseline (少样本上下文学习 ICL)

在 Prompt 中加入示例，测试模型的模仿学习能力。

**关键参数：** `--use_few_shot`

```
torchrun --nproc_per_node=1 code/model_test_fixed.py \
    --model_name qwen \
    --use_few_shot \
    --batch_size 16 \
    --cuda_devices "0"
```

- **输出文件**：`output/qwen_few_shot_test_result.json`

#### 3. RAG Baseline (检索增强生成)

利用向量检索精简上下文，测试模型处理长文本的能力。 *(注意：必须先运行附录中的 RAG 数据生成步骤)*

**关键参数：** `--use_rag`

```
torchrun --nproc_per_node=1 code/model_test_fixed.py \
    --model_name qwen \
    --use_rag \
    --batch_size 16 \
    --cuda_devices "0"
```

- **输出文件**：`output/qwen_rag_test_result.json`

### 通用参数说明

| 参数               | 说明                                | 建议值                      |
| ------------------ | ----------------------------------- | --------------------------- |
| `--model_name`     | 模型名称                            | `qwen` 等                   |
| `--use_finetuned`  | **[关键]** 是否加载微调后的 adapter | 测试 SFT 模型时必选         |
| `--use_few_shot`   | 开启少样本模式                      | 仅用于基线测试              |
| `--use_rag`        | 开启 RAG 模式                       | 仅用于基线测试              |
| `--batch_size`     | 推理 Batch Size                     | A800/A100 可设 64           |
| `--nproc_per_node` | 进程数                              | **必须等于** 使用的显卡数量 |

## 4. 第三步：结果评估

使用 `code/eval_fixed.py` 对生成的 JSON 结果文件进行指标计算（EM, Precision, Recall, F1）。

### 运行指令示例

```
# 评估 SFT 结果
python code/eval_fixed.py --result_file output/qwen_sft_zero_shot_test_result.json

# 评估 Zero-shot 基线结果
python code/eval_fixed.py --result_file output/qwen_zero_shot_test_result.json
```

## 附录：模型下载

```
hf auth login
export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT
export HF_TOKEN=YOUR_HF_TOKEN
echo $HF_TOKEN
```

**输出：** 模型将在 `models/` 目录下生成文件。

## 附录：RAG 数据集生成

如果您需要进行基于 RAG 的实验，请先运行此步骤生成带有检索上下文的测试集。

```
# 使用 Qwen2.5-7B 作为 Embedding 模型
python code/rag_dataset_generation.py --model_name qwen --cuda_devices "0"
```

**输出：** 脚本将在 `dataset/` 目录下生成文件，例如 `dataset/Qwen2.5-7B-Instruct_rag_test_data.json`。