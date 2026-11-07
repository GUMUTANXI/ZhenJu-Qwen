---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: apache-2.0
language:
  - zh
tags:
  - lora
  - chinese
  - grammatical-error-correction
---

# 诊句 · ZhenJu LoRA

诊句用 Qwen3-1.7B 识别中文语病、解释改法并生成改正句。这个 LoRA 适配器在 FCGEC 上训练，可通过命令行或 Gradio 加载。

[中文首页](../../README.md) · [English overview](../../README.en.md) · [完整评测](../../docs/details.md)

| 项目 | 内容 |
|---|---|
| 任务 | 判断句子是否有语病，归入 7 类错误，说明修改方法并给出改正句 |
| 训练数据 | FCGEC 主训练切分（25,605 句） |
| 配置 | r=16，alpha=32，3 轮，学习率 2e-4，约占全部参数的 1% |
| 测试集结果 | 检测准确率 0.721，类型 Macro-F1 0.515，字符级改正 F0.5 0.249 |

仓库收录生成上述结果的 LoRA 权重。加载时使用原始基础模型。 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

## 输入输出

使用仓库 `src/zhenju/prompts.py` 里的提示模板（需要关闭思考模式），模型按四行回答：

```text
判断：病句
类型：搭配不当
修改：把“改变”改为“提高”
改正：改革开放40年来，人民的物质生活水平有了明显的提高。
```

## 评测范围

评测使用 FCGEC 官方验证集保留的 1,998 句，结果来自单次运行。语料主要来自语文考试题，适配器可用于教学中的语病初筛和修改提示。
