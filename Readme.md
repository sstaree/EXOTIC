# [CVPR-26] EXOTIC: External Vision Guided Incomplete Multi-view Classification
<p align="center">
  🎆 Welcome to the repo of EXOTIC! 🎆  
</p>
<p align="center">
  Paper Link: <a href="https://ojs.aaai.org/index.php/AAAI/article/view/35485">Click Here</a>  
</p>
<p align="center">
  If you find this work helpful, please support us with your stars⭐ and cite our paper!    
</p>
<p align="center">
  This repo contains the official PyTorch implementation of NLC!      
</p>

## 🔥 News
• [2026/02/22] 🔥 Our paper has been accepted by CVPR2026!
## 📖 Introduction
### Motivation
Incomplete multi-view data, often caused by sensor failures or occlusions, suffers from limited semantic information due to missing views. Existing Incomplete Multi-View Classification (IMVC) methods primarily rely on mining internal supervision from partially observed data, which inherently constrains their performance ceiling. The key motivation is to break through this limitation by incorporating external knowledge as additional semantic guidance for missing view completion.
### Method
The proposed Noisy Label Calibration (NLC) method includes:<br>
  &nbsp;&nbsp;1.External Knowledge Construction – builds a vision knowledge library using unlabeled images and a pre-trained vision-language model (BLIP) to extract rich semantic representations.<br>
  &nbsp;&nbsp;2.Knowledge Filtering – employs a dual-channel decoupling architecture to adaptively select task-relevant external knowledge for each instance while avoiding interference with representation learning.<br>
  &nbsp;&nbsp;3.Knowledge Purification – introduces category alignment loss and and knowledge alignment loss  to reconcile external knowledge with internal representations and mitigate semantic conflicts.<br>
  &nbsp;&nbsp;4.External Completion – leverages purified external knowledge to impute missing views in the second channel, followed by cross-entropy classification loss for final prediction.<br>
![image](https://github.com/sstaree/EXOTIC/blob/22b0d4a7ff34c0177de002210ea7ecff19f75028/img/framework_01.png)

### Dataset
• All dataset can be downloaded at: <br>
https://pan.baidu.com/s/1uLPfx5lMMqMdYtFUav7jcQ?pwd=6666 <br>
passowrd: 6666 
## 💖 Acknowledgements
We would like to present our sincere thanks to the authors for their contributions to the community!
## 📝 Citation
If you find this repo helpful, please consider citing this paper:<br>
```bibtex
waiting!!!
```
## 📧 Contact
If you have any questions, please feel free to contact the author:<br>
Shi-Lin Xu: xushilin990@gmail.com.


