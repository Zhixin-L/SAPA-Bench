<div align="center">
  <h1>📲<i>Mind the Third Eye!</i> Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents</h1> 
        If our project helps you, please give us a star ⭐ on GitHub to support us. 🥸🥸


[![arXiv](https://img.shields.io/badge/arXiv-2508.19493-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2508.19493) [![hf_checkpoint](https://img.shields.io/badge/🤗-SAPA--Bench-9C276A.svg)](https://huggingface.co/datasets/OmniQuest/SAPA-Bench) 
</div>

## 🔥 News
* **`2025-08-28`** 🎉 🌟 We are happy to release the SAPA-Bench. You can find the SAPA-Bench from [![hf_checkpoint](https://img.shields.io/badge/🤗-SAPA--Bench-9C276A.svg)](https://huggingface.co/datasets/OmniQuest/SAPA-Bench).



## TODO
- [ ] Release the SAPA-Bench.
- [ ] Release the latest evaluation code.
- [ ] ···

## 📖SAPA-Bench Overview

Smartphones offer great convenience but also collect vast amounts of personal information.  
With the rise of **MLLM-powered smartphone agents**, automation performance has improved significantly—yet at the cost of **extensive access to sensitive user data**.  

To systematically evaluate this issue, we introduce the **first large-scale benchmark** (7,138 scenarios) for **privacy awareness in smartphone agents**. Each scenario is annotated with:  
- 🔑 **Privacy Type** (e.g., *Account Credentials*)  
- ⚠️ **Sensitivity Level**  
- 📍 **Location**  

We benchmarked **seven mainstream smartphone agents** and found:  
- Overall privacy awareness (RA) remains **below 60%**, even with explicit hints.  
- **Closed-source agents** generally perform better; **Gemini 2.0-flash** achieved the highest RA (67%).  
- Privacy detection strongly correlates with **sensitivity level**—higher sensitivity makes scenarios more identifiable.  

👉 Our results highlight the urgent need to rethink the **utility–privacy tradeoff** in the design of smartphone agents.  




## 🛠️Evaluation 
```shell

```

## 🌟 Star History


## 📑 Citation
If you find **SAPA-Bench** useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{lin2025sapa,
  title   = {Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents},
  author  = {Lin, Zhixin and Li, Jungang and Pan, Shidong and Shi, Yibo and Yao, Yue and Xu, Dongliang},
  journal = {arXiv preprint arXiv:2508.19493},
  year    = {2025}
}
```
