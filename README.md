<div align="center">
  <h2>
    <a href="https://github.com/Zhixin-L/SAPA-Bench">
      SAPA-Bench: Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents
    </a>
  </h2>
</div>

<p align="center">
  <a href="https://scholar.google.com/citations?user=dSQdVooAAAAJ">Zhixin Lin</a><sup>1</sup>,
  <a href="https://github.com/LJungang">Jungang Li</a><sup>2,3</sup>,
  <a href="https://shidongpan.github.io/">Shidong Pan</a><sup>4</sup>,
  <a href="https://openreview.net/profile?id=~Yibo_Shi2">Yibo Shi</a><sup>5</sup>,
  <a href="https://scholar.google.com/citations?user=dSQdVooAAAAJ">Yue Yao</a><sup>1†</sup>,
  <a href="https://faculty.sdu.edu.cn/xudongliang/zh_CN/index.htm">Dongliang Xu</a><sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>Shandong University&nbsp;&nbsp;
  <sup>2</sup>HKUST(GZ)&nbsp;&nbsp;
  <sup>3</sup>HKUST&nbsp;&nbsp;
  <sup>4</sup>Columbia University&nbsp;&nbsp;
  <sup>5</sup>Xi’an Jiaotong University
</p>

<p align="center">
  <em>† Corresponding Author</em>
</p>

<div align="center">
  <p align="center">
    &nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2508.19493"><b>Paper</b></a>&nbsp&nbsp |
    &nbsp&nbsp🏠 <a href="https://zhixin-l.github.io/SAPA-Bench"><b>Project Page</b></a>&nbsp&nbsp |
    🤗 <a href="https://huggingface.co/datasets/OmniQuest/SAPA-Bench"><b>Dataset</b></a>&nbsp&nbsp
  </p>

  <p align="center">
    If you find this work useful, please consider starring ⭐ the repository to support our research.
  </p>
</div>


## 📰 News
* **`2025-11-18`** 🌟 We have realeased the SAPA-Bench on the [![hf_checkpoint](https://img.shields.io/badge/🤗-SAPA--Bench-9C276A.svg?style=flat-square)](https://huggingface.co/datasets/OmniQuest/SAPA-Bench) today. Feel free to try it out, and don't forget to leave us a ⭐ — it really means a lot to us!
* **`2025-11-09`** 🎉🎉🎉 Our work has been accepted by AAAI 2026!
* **`2025-08-28`** 🎉 🌟 We are happy to release the SAPA-Bench. You can find the SAPA-Bench from [![hf_checkpoint](https://img.shields.io/badge/🤗-SAPA--Bench-9C276A.svg?style=flat-square)](https://huggingface.co/datasets/OmniQuest/SAPA-Bench).



## 👉 TODO
- [x] Release the SAPA-Bench.
- [ ] Release the latest evaluation code.
- [ ] ···

## 📖 SAPA-Bench Overview

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

 Our results highlight the urgent need to rethink the **utility–privacy tradeoff** in the design of smartphone agents.  




## 🛠️ Evaluation 
```shell
bash run.sh
```
Note: For evaluation details, please refer to the Eval README.
## 🌟 Star History


## 📑 Citation
If you find **SAPA-Bench** useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{lin2025sapa,
  title      = {Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents},
  author     = {Lin, Zhixin and Li, Jungang and Pan, Shidong and Shi, Yibo and Yao, Yue and Xu, Dongliang},
  booktitle  = {The Fortieth Annual AAAI Conference on Artificial Intelligence},
  year       = {2026},
}
```
