# Awesome-Zero-Shot-Learning
Selected papers I've read in the field of zero shot learning. Not a complete list of all accepted papers. I'll mainly focus on those papers with open-source implementations and most interesting to me. 

Note: This list may contain understanding bias and personal preference. All that papers without available code will not provide a link to it.
## Table of Contents
+ [Papers](#Papers)
+ [Datasets](#Datasets)

### Papers
#### arXiv
+ **CIZSL++**: CIZSL++: Creativity Inspired Generative Zero-Shot Learning. Mohamed Elhoseiny, Kai Yi, Mohamed Elfeki. [[paper]](https://arxiv.org/pdf/2101.00173.pdf) [[code]](https://github.com/Elhoseiny-VisionCAIR-Lab/CIZSL.v2)

#### ICLR 2021
+ **CN-ZSL**: Class Normalization for Zero Shot Learning. Ivan Skorokhodov, Mohamed Elhoseiny. [[paper]](https://openreview.net/forum?id=7pgFL2Dkyyy)  [[Code]](https://github.com/universome/nm-zsl)
       
       Note: In the paper, the authors investigated basic normalization strategies and proposed the novel class normalization. Besides, they introduced a more general continual zero-shot learning setting. But generally, there is a lot of space to improve on that.

#### NeurIPS 2020
+ **Composer** Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition. D.~Huynh and E.~Elhamifar. [[paper]](https://hbdat.github.io/pubs/neurips20_CompositionZSL_final.pdf) [[code]](https://github.com/hbdat/neurIPS20_CompositionZSL)
+ **APN** Attribute Prototype Network for Zero-Shot Learning. Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata. [[empty github repo, no further consideration on this paper]]()

#### ECCV 2020
+ **f-VAEGAN**: Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification. Sanath Narayan*, Akshita Gupta*, Fahad Shahbaz Khan, Cees G. M. Snoek, Ling Shao. [[arXiv]](https://arxiv.org/abs/2003.07833) [[Code]](https://github.com/akshitac8/tfvaegan)
+ **LsrGAN**: Leveraging Seen and Unseen Semantic Relationships for Generative Zero-Shot Learning. Maunil R Vyas, Hemanth Venkateswara, Sethuraman Panchanathan. [[arXiv]](https://arxiv.org/abs/2007.09549) [[Code]](https://github.com/Maunil/LsrGAN)
            
      Note: This is a paper based on GAZSL and the loss design is interesting and easy to follow. However, this is for transductive zero shot learning, which uses image features of unseen classes at training step.
+ TBD

#### CVPR 2020
+ **Hyperbolic-ZSL**: Shaoteng Liu, Jingjing Chen, Liangming Pan, Chong-Wah Ngo, Tat-Seng Chua, Yu-Gang Jiang. Hyperbolic Visual Embedding Learning for Zero-Shot Recognition. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Hyperbolic_Visual_Embedding_Learning_for_Zero-Shot_Recognition_CVPR_2020_paper.pdf)[[Code]](https://github.com/ShaoTengLiu/Hyperbolic_ZSL)

      Note: The most important part in this paper is the evaluations on ImageNet, which has hierarchical structures of labels. However, the processed ImageNet feature data was not provided and no response from the authors yet. I havn't tested the code for this reason so I'm not very sure this implementation can achieve reported results.
      
+ Dat Huynh, Ehsan Elhamifar. Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.pdf)
+ Dat Huynh, Ehsan Elhamifar. A Shared Multi-Attention Framework for Multi-Label Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_A_Shared_Multi-Attention_Framework_for_Multi-Label_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Suchen Wang, Kim-Hui Yap, Junsong Yuan, Yap-Peng Tan. Discovering Human Interactions with Novel Objects via Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Discovering_Human_Interactions_With_Novel_Objects_via_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Shaobo Min, Hantao Yao, Hongtao Xie, Chaoqun Wang, Zheng-Jun Zha, Yongdong Zhang. Domain-aware Visual Bias Eliminating for Generalized Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Min_Domain-Aware_Visual_Bias_Eliminating_for_Generalized_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Jiamin Wu, Tianzhu Zhang, Zheng-Jun Zha, Jiebo Luo, Yongdong Zhang, Feng Wu. Self-supervised Domain-aware Generative Network for Generalized Zero-shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Self-Supervised_Domain-Aware_Generative_Network_for_Generalized_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Rohit Keshari, Richa Singh, Mayank Vatsa. Generalized Zero-Shot Learning Via Over-Complete Distribution. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Keshari_Generalized_Zero-Shot_Learning_via_Over-Complete_Distribution_CVPR_2020_paper.pdf)
+ Yunlong Yu, Zhong Ji, Jungong Han, Zhongfei Zhang. Episode-based Prototype Generating Network for Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Episode-Based_Prototype_Generating_Network_for_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Pengkai Zhu, Hanxiao Wang, Venkatesh Sligrama. Don't Even Look Once: Synthesizing Features for Zero-Shot Detection. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Dont_Even_Look_Once_Synthesizing_Features_for_Zero-Shot_Detection_CVPR_2020_paper.pdf)
+ Zongyan Han, Zhenyong Fu, Jian Yang. Learning the Redundancy-free Features for Generalized Zero-Shot Object Recognition. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Han_Learning_the_Redundancy-Free_Features_for_Generalized_Zero-Shot_Object_Recognition_CVPR_2020_paper.pdf)
+ Rohit Keshari, Richa Singh, Mayank Vatsa. Generalized Zero-Shot Learning Via Over-Complete Distribution. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Keshari_Generalized_Zero-Shot_Learning_via_Over-Complete_Distribution_CVPR_2020_paper.pdf)

#### ICCV 2019
+ **CIZSL**: Mohamed Elhoseiny, Mohamed Elfeki. Creativity Inspired Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Elhoseiny_Creativity_Inspired_Zero-Shot_Learning_ICCV_2019_paper.pdf) [[arXiv]](https://arxiv.org/abs/1904.01109) [[Code]](https://github.com/mhelhoseiny/CIZSL)

      Note: Very interesting paper on designing a creative loss which can help properly formulate deviation from generating features similar to existing classes while balancing the desirable transfer learning signal.

#### CVPR 2019
+ **DGP**: Michael Kampffmeyer, Yinbo Chen, Xiaodan Liang, Hao Wang, Yujia Zhang, and Eric P. Xing. Rethinking Knowledge Graph Propagation for Zero-Shot Learning. [[arXiv]](https://arxiv.org/abs/1805.11724) [[Code]](https://github.com/cyvius96/DGP)

#### CVPR 2018
+ **GAZSL**: Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng, Ahmed Elgammal. "A Generative Adversarial Approach for Zero-Shot Learning From Noisy Texts". [[arXiv]](https://arxiv.org/abs/1712.01381) [[Code]](https://github.com/EthanZhu90/ZSL_GAN)

      Note: This paper is one of the earlierst works working on textual-based CUBird and NABird. And the proposed two heads discriminator and the pipeline are widely accepted.

#### PAMI 2018
+ **GBU**: Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata. "Zero-shot learning-A comprehensive evaluation of the good, the bad and the ugly". [[arXiv]](https://arxiv.org/abs/1707.00600) [[Project]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)

      Note: This paper maybe one of the most influential papers in zero-shot learning.

## Datasets
+ TBD
