## Synthetic Data Enhances Mathematical Reasoning of Language Models
This project explored using synthetic mathematical data: linear algebra and abstract algebra theorems and calculation generated by [Gretel.AI](https://console.gretel.ai/navigator)

## Base Models on Benchmarks 
| Benchmarks                | Model                    | Accuracy |
|---------------------------|--------------------------|----------|
| MMLU Abstract Algebra     | GPT-3.5-Turbo (LLM)      | 22.00%   |
| Linear Algebra Theorem QA | GPT-3.5-Turbo (LLM)      | 9.62%    |
| Linear Algebra QA         | GPT-3.5-Turbo (LLM)      | 31.84%   |
|                           | LLaMA-2-7B (SLM)         | 5.83%    |
|                           | LLaMA-2-13B (SLM)        | 8.07%    |
|                           | Mistral-7B-v0.1 (SLM)    | 14.80%   |
|                           | BLOOM 7B1 (SLM)          | 0.90%    |
| MATH Linear Algebra       | GPT-3.5-Turbo (LLM)      | 8.60%    |
|                           | LLaMA-2-7B (SLM)         | 0.30%    |
|                           | LLaMA-2-13B (SLM)        | 1.05%    |
|                           | Mistral-7B-v0.1 (SLM)    | 1.95%    |
|                           | BLOOM 7B1 (SLM)          | 0.00%    |

## Finetuned Models on Benchmarks (Highest Performance)
All finetuned version could be found in [Huggingface](https://huggingface.co/Charlie-Han-01).
| Benchmarks                | Finetuned Models         | Accuracy |
|---------------------------|--------------------------|----------|
| MMLU Abstract Algebra     | GPT-3.5-Turbo (LLM)      | 26.00%   |
| Linear Algebra Theorem QA | GPT-3.5-Turbo (LLM)      | 25.00%   |
| Linear Algebra QA         | GPT-3.5-Turbo (LLM)      | 34.98%   |
|                           | LLaMA-2-7B (SLM)         | 8.52%    |
|                           | LLaMA-2-13B (SLM)        | 12.11%   |
|                           | Mistral-7B-v0.1 (SLM)    | 26.91%   |
|                           | BLOOM 7B1 (SLM)          | 2.24%    |
| MATH Linear Algebra       | GPT-3.5-Turbo (LLM)      | 10.65%   |
|                           | LLaMA-2-7B (SLM)         | 1.55%    |
|                           | LLaMA-2-13B (SLM)        | 2.45%    |
|                           | Mistral-7B-v0.1 (SLM)    | 7.85%    |
|                           | BLOOM 7B1 (SLM)          | 0.45%    |

## Files
* Code: the evalution process of pre-trained models and finetuned models on benchmarks, and corresponding visualizations.
* Data: Including finetuning data and benchmarks. There exists different formats of data such as csv and json. And we used data_combine.py and data_convert.py to acquire the final required data for model finetuning and evaluation.
* Figures: Collected figures and citations.
* Math QA paper: Overleaf Latex version of the paper. 


