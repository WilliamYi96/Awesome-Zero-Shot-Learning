# Awesome-Zero-Shot-Learning
Selected papers I've read in the field of zero shot learning. Not a complete list of all accepted papers. I'll mainly focus on those papers with open-source implementations and most interesting to me. 

**Note**: *This list may contain understanding bias and personal preference. Paper link will not be provided for the one without an available code.*
## Table of Contents
+ [Papers](#Papers)
+ [Datasets](#Datasets)                                 

### Papers      
#### Survey
+ **GBU (PAMI-18)**: Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata. "Zero-shot learning-A comprehensive evaluation of the good, the bad and the ugly". [[arXiv]](https://arxiv.org/abs/1707.00600) [[Project]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)
+ **TIST19**: Wei Wang, Vincent W. Zheng, Han Yu, and Chunyan Miao. A Survey of Zero-Shot Learning: Settings, Methods, and Applications. [[paper]](https://dl.acm.org/doi/10.1145/3293318)
+ **arXiv20**: Farhad Pourpanah, Moloud Abdar, Yuxuan Luo, Xinlei Zhou, Ran Wang, Chee Peng Lim, and Xi-Zhao Wang. A Review of Generalized Zero-Shot Learning Methods. [[arxiv]](https://arxiv.org/abs/2011.08641)

#### arXiv
+ **GRaWD**: Imaginative Walks: Generative Random Walk Deviation Loss for Improved Unseen Learning Representation. Mohamed Elhoseiny, Divyansh Jha, Kai Yi, Ivan Skorokhodov. [[paper]](https://arxiv.org/abs/2104.09757) [[code]](https://github.com/Vision-CAIR/GRaWD)
+ **CIZSL++**: CIZSL++: Creativity Inspired Generative Zero-Shot Learning. Mohamed Elhoseiny, Kai Yi, Mohamed Elfeki. [[paper]](https://arxiv.org/pdf/2101.00173.pdf) [[code]](https://github.com/Elhoseiny-VisionCAIR-Lab/CIZSL.v2)

#### ICCV 2021
+ **FREE**: Feature Refinement for Generalized Zero-Shot Learning. Shiming Chen, Wenjie Wang, Beihao Xia, Qinmu Peng, Xinge You, Feng Zheng, Ling Shao. [[paper]](https://arxiv.org/abs/2107.13807) [[code]](https://github.com/shiming-chen/FREE)
+ **SDGZSL**: Semantic Disentangling Generalized Zero-Shot Learning. Zhi Chen, Ruihong Qiu, Sen Wang, Zi Huang, Jingjing Li, Zheng Zhang. [[paper]](https://arxiv.org/abs/2101.07978) [[No code - 210805]]()
#### CVPR 2021
+ **GEM-ZSL**: Goal-Oriented Gaze Estimation for Zero-Shot Learning. Yang Liu, Lei Zhou, Xiao Bai, Yifei Huang, Lin Gu, Jun Zhou, Tatsuya Harada. [[paper]](https://arxiv.org/abs/2103.03433) [[code]](https://github.com/osierboy/GEM-ZSL)
+ **CE-GZSL**: Contrastive Embedding for Generalized Zero-Shot Learning. Zongyan Han, Zhenyong Fu, Shuo Chen, Jian Yang. [[paper]](https://arxiv.org/abs/2103.16173) [[code]](https://github.com/Hanzy1996/CE-GZSL)
+ **CGE**: Learning Graph Embeddings for Compositional Zero-Shot Learning. Muhammad Ferjad Naeem, Yongqin Xian, Federico Tombari, Zeynep Akata. [[paper]](https://arxiv.org/abs/2102.01987) [[code]](https://github.com/ExplainableML/czsl)
+ **CompCos**: Open World Compositional Zero-Shot Learning. Massimiliano Mancini, Muhammad Ferjad Naeem, Yongqin Xian, Zeynep Akata [[paper]](https://arxiv.org/abs/2101.12609) [[code]](https://github.com/ExplainableML/czsl)
+ **GCM-CF**: Counterfactual Zero-Shot and Open-Set Visual Recognition. Zhongqi Yue, Tan Wang, Qianru Sun, Xian-Sheng Hua, Hanwang Zhang [[paper]](https://arxiv.org/abs/2103.00887) [[code]](https://github.com/yue-zhongqi/gcm-cf)
+ **STHS**: Hardness Sampling for Self-Training Based Transductive Zero-Shot Learning. Liu Bo, Qiulei Dong, Zhanyi Hu. [[empty github repo - 210805]]()

#### ICLR 2021
+ **CN-ZSL**: Class Normalization for Zero Shot Learning. Ivan Skorokhodov, Mohamed Elhoseiny. [[paper]](https://openreview.net/forum?id=7pgFL2Dkyyy)  [[Code]](https://github.com/universome/nm-zsl)
       
   
       Note: In the paper, the authors investigated basic normalization strategies and proposed the novel class normalization. Besides, they introduced a more general continual zero-shot learning setting. But generally, there is a lot of space to improve on that.
   
+ **IPN**: Lu Liu, Tianyi Zhou, Guodong Long, Jing Jiang, Xuanyi Dong, Chengqi Zhang. Isometric Propagation Network for Generalized Zero-shot Learning. [[No code - 210805]]()
+ **AGZSL**: Yu-Ying Chou, Hsuan-Tien Lin, Tyng-Luh Liu. Adaptive and Generative Zero-Shot Learning. [[paper]](https://openreview.net/forum?id=ahAUv8TI2Mz) [[code]](https://github.com/anonmous529/AGZSL)
+ Yunhao Ge, Sami Abu-El-Haija, Gan Xin, Laurent Itti. Zero-shot Synthesis with Group-Supervised Learning. [[No code - 210805]]()
#### NeurIPS 2020
+ **Composer**: Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition. D.~Huynh and E.~Elhamifar. [[paper]](https://hbdat.github.io/pubs/neurips20_CompositionZSL_final.pdf) [[code]](https://github.com/hbdat/neurIPS20_CompositionZSL)
+ **APN**: Attribute Prototype Network for Zero-Shot Learning. Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata. [[empty github repo, no further consideration on this paper-210311]]()

#### ECCV 2020
+ **f-VAEGAN**: Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification. Sanath Narayan*, Akshita Gupta*, Fahad Shahbaz Khan, Cees G. M. Snoek, Ling Shao. [[arXiv]](https://arxiv.org/abs/2003.07833) [[Code]](https://github.com/akshitac8/tfvaegan)
+ **LsrGAN**: Leveraging Seen and Unseen Semantic Relationships for Generative Zero-Shot Learning. Maunil R Vyas, Hemanth Venkateswara, Sethuraman Panchanathan. [[arXiv]](https://arxiv.org/abs/2007.09549) [[Code]](https://github.com/Maunil/LsrGAN)
              
  
      Note: This is a paper based on GAZSL and the loss design is interesting and easy to follow. However, this is for transductive zero shot learning, which uses the semantic features of unseen classes at training step.
+ **OOD**: A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning. Xingyu Chen, Xuguang Lan, Fuchun Sun, Nanning Zheng. [[paper]](https://arxiv.org/abs/2008.04872) [[code]](https://github.com/Chenxingyu1990/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning)    

       Note: Currently, reproduced results are far below the reported results following the released code. [21--3-16] .
   
+ **RGEN**: Region Graph Embedding Network for Zero-shot Learning - Guo-Sen Xie, Li Liu, Fan Zhu, Fang Zhao, Zheng Zhang, Yazhou Yao, Jie Qin, Ling Shao. [[No code-210311]]()

#### CVPR 2020
+ **Hyperbolic-ZSL**: Shaoteng Liu, Jingjing Chen, Liangming Pan, Chong-Wah Ngo, Tat-Seng Chua, Yu-Gang Jiang. Hyperbolic Visual Embedding Learning for Zero-Shot Recognition. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Hyperbolic_Visual_Embedding_Learning_for_Zero-Shot_Recognition_CVPR_2020_paper.pdf)[[Code]](https://github.com/ShaoTengLiu/Hyperbolic_ZSL)

      Note: The most important part in this paper is the evaluations on ImageNet, which has hierarchical structures of labels. However, the processed ImageNet feature data was not provided and no response from the authors yet. I havn't tested the code for this reason so I'm not very sure this implementation can achieve reported results.

+ **DVBE**: Shaobo Min, Hantao Yao, Hongtao Xie, Chaoqun Wang, Zheng-Jun Zha, Yongdong Zhang. Domain-aware Visual Bias Eliminating for Generalized Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Min_Domain-Aware_Visual_Bias_Eliminating_for_Generalized_Zero-Shot_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/mboboGO/DVBE)
+ **DAZLE**: Dat Huynh, Ehsan Elhamifar. Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.pdf) [[code]](https://github.com/hbdat/cvpr20_DAZLE)
+ **LESA**: Dat Huynh, Ehsan Elhamifar. A Shared Multi-Attention Framework for Multi-Label Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_A_Shared_Multi-Attention_Framework_for_Multi-Label_Zero-Shot_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/hbdat/cvpr20_DAZLE)
+ **RFF-GZSL**: Learning the Redundancy-Free Features for Generalized Zero-Shot Object Recognition. Zongyan Han, Zhenyong Fu, Jian Yang. [[No code-210311]]()
+ **SDGN**: Jiamin Wu, Tianzhu Zhang, Zheng-Jun Zha, Jiebo Luo, Yongdong Zhang, Feng Wu. Self-supervised Domain-aware Generative Network for Generalized Zero-shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Self-Supervised_Domain-Aware_Generative_Network_for_Generalized_Zero-Shot_Learning_CVPR_2020_paper.pdf) [[No code - 210311]]()
+ **E-PGN**: Yunlong Yu, Zhong Ji, Jungong Han, Zhongfei Zhang. Episode-based Prototype Generating Network for Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Episode-Based_Prototype_Generating_Network_for_Zero-Shot_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/yunlongyu/EPGN)

       Note: Code issue - "nan" loss and "0.0" accurracy, [[issue#5]](https://github.com/yunlongyu/EPGN/issues/5).

+ Rohit Keshari, Richa Singh, Mayank Vatsa. Generalized Zero-Shot Learning via Over-Complete Distribution. [[No code - 210311]]()

+ Below partial related:
+ Suchen Wang, Kim-Hui Yap, Junsong Yuan, Yap-Peng Tan. Discovering Human Interactions with Novel Objects via Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Discovering_Human_Interactions_With_Novel_Objects_via_Zero-Shot_Learning_CVPR_2020_paper.pdf)
+ Pengkai Zhu, Hanxiao Wang, Venkatesh Sligrama. Don't Even Look Once: Synthesizing Features for Zero-Shot Detection. [[CVF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Dont_Even_Look_Once_Synthesizing_Features_for_Zero-Shot_Detection_CVPR_2020_paper.pdf)


#### ICCV 2019
+ **CIZSL**: Mohamed Elhoseiny, Mohamed Elfeki. Creativity Inspired Zero-Shot Learning. [[CVF]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Elhoseiny_Creativity_Inspired_Zero-Shot_Learning_ICCV_2019_paper.pdf) [[arXiv]](https://arxiv.org/abs/1904.01109) [[Code]](https://github.com/mhelhoseiny/CIZSL)

  ~~~
  Note: Very interesting paper on designing a creative loss which can help properly formulate deviation from generating features similar to existing classes while balancing the desirable transfer learning signal.
  ~~~

+ **AttentionZSL**: Yang Liu, Jishun Guo, Deng Cai, Xiaofei He. Attribute Attention for Semantic Disambiguation in Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Attribute_Attention_for_Semantic_Disambiguation_in_Zero-Shot_Learning_ICCV_2019_paper.pdf) [[Code]](https://github.com/ZJULearning/AttentionZSL)

+ **cvcZSL**: Kai Li, Martin Renqiang Min, Yun Fu. Rethinking Zero-Shot Learning: A Conditional Visual Classification Perspective. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Rethinking_Zero-Shot_Learning_A_Conditional_Visual_Classification_Perspective_ICCV_2019_paper.pdf) [[Code]](https://github.com/kailigo/cvcZSL) 

+ Senthil Purushwalkam, Maximilian Nickel, Abhinav Gupta, Marc'Aurelio Ranzato. Task-Driven Modular Networks for Zero-Shot Compositional Learning. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Purushwalkam_Task-Driven_Modular_Networks_for_Zero-Shot_Compositional_Learning_ICCV_2019_paper.pdf) [[Code]](https://github.com/facebookresearch/taskmodularnets)

+ **TCN**: Huajie Jiang, Ruiping Wang, Shiguang Shan, Xilin Chen. Transferable Contrastive Network for Generalized Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Transferable_Contrastive_Network_for_Generalized_Zero-Shot_Learning_ICCV_2019_paper.pdf) [[No code-210311]]() 

+ Yizhe Zhu, Jianwen Xie, Bingchen Liu, Ahmed Elgammal. Learning Feature-to-Feature Translator by Alternating Back-Propagation for Generative Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_Learning_Feature-to-Feature_Translator_by_Alternating_Back-Propagation_for_Generative_Zero-Shot_Learning_ICCV_2019_paper.pdf) [[Code]](https://github.com/EthanZhu90/ZSL_ABP)

+ Yannick Le Cacheux, Herve Le Borgne, Michel Crucianu. Modeling Inter and Intra-Class Relations in the Triplet Loss for Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Le_Cacheux_Modeling_Inter_and_Intra-Class_Relations_in_the_Triplet_Loss_for_ICCV_2019_paper.pdf) [[No code-210311]]() 

#### CVPR 2019
+ **DGP**: Michael Kampffmeyer, Yinbo Chen, Xiaodan Liang, Hao Wang, Yujia Zhang, and Eric P. Xing. Rethinking Knowledge Graph Propagation for Zero-Shot Learning. [[arXiv]](https://arxiv.org/abs/1805.11724) [[Code]](https://github.com/cyvius96/DGP)
+ **CADA-VAE**: Edgar Schönfeld, Sayna Ebrahimi, Samarth Sinha, Trevor Darrell, Zeynep Akata. Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schonfeld_Generalized_Zero-_and_Few-Shot_Learning_via_Aligned_Variational_Autoencoders_CVPR_2019_paper.pdf) [[Code]](https://github.com/edgarschnfld/CADA-VAE-PyTorch)
+ Bin Tong, Chao Wang, Martin Klinkigt, Yoshiyuki Kobayashi, Yuuichi Nonaka. Hierarchical Disentanglement of Discriminative Latent Features for Zero-shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tong_Hierarchical_Disentanglement_of_Discriminative_Latent_Features_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[No code-210311]]()
+ **SABR-T**: Akanksha Paul, Naraynan C Krishnan, Prateek Munjal. Semantically Aligned Bias Reducing Zero Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Paul_Semantically_Aligned_Bias_Reducing_Zero_Shot_Learning_CVPR_2019_paper.pdf) [[No code-210311]]() 
+ **Gzsl-VSE**: Pengkai Zhu, Hanxiao Wang, Venkatesh Saligrama. Generalized Zero-Shot Recognition based on Visually Semantic Embedding. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Generalized_Zero-Shot_Recognition_Based_on_Visually_Semantic_Embedding_CVPR_2019_paper.pdf) [[No code-210311]]() 
+ **COSMO**: Yuval Atzmon, Gal Chechik. Adaptive Confidence Smoothing for Generalized Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Atzmon_Adaptive_Confidence_Smoothing_for_Generalized_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/yuvalatzmon/COSMO) 
+ **AREN**: Guo-Sen Xie, Li Liu, Xiaobo Jin, Fan Zhu, Zheng Zhang, Jie Qin, Yazhou Yao, Ling Shao. Attentive Region Embedding Network for Zero-shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Attentive_Region_Embedding_Network_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/gsx0/Attentive-Region-Embedding-Network-for-Zero-shot-Learning)
+ Tristan Hascoet, Yasuo Ariki, Tetsuya Takiguchi. On Zero-Shot Learning of generic objects.  [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hascoet_On_Zero-Shot_Recognition_of_Generic_Objects_CVPR_2019_paper.pdf) [[Code]](https://github.com/TristHas/GOZ)
+ Zhengming Ding, Hongfu Liu. Marginalized Latent Semantic Encoder for Zero-Shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Marginalized_Latent_Semantic_Encoder_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[No code-210311]]() 
+ **PrEN**: Meng Ye, Yuhong Guo. Progressive Ensemble Networks for Zero-Shot Recognition. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Progressive_Ensemble_Networks_for_Zero-Shot_Recognition_CVPR_2019_paper.pdf) [[No code-210311]]() 
+ **GDAN**: He Huang, Changhu Wang, Philip S. Yu, Chang-Dong Wang. Generative Dual Adversarial Network for Generalized Zero-shot Learning. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Generative_Dual_Adversarial_Network_for_Generalized_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/stevehuanghe/GDAN)
+ **PQZSL**: Jin Li, Xuguang Lan, Yang Liu, Le Wang, Nanning Zheng. Compressing Unknown Classes with Product Quantizer for Efficient Zero-Shot Classification. [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Unknown_Images_With_Product_Quantizer_for_Efficient_Zero-Shot_Classification_CVPR_2019_paper.pdf) [[No code-210311]]() 
+ **LisGAN**: Jingjing Li, Mengmeng Jin, Ke Lu, Zhengming Ding, Lei Zhu, Zi Huang. Leveraging the Invariant Side of Generative Zero-Shot Learning.  [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Leveraging_the_Invariant_Side_of_Generative_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/lijin118/LisGAN)
+ **gmnZSL**: Mert Bulent Sariyildiz, Ramazan Gokberk Cinbis. Gradient Matching Generative Networks for Zero-Shot Learning.  [[CVF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sariyildiz_Gradient_Matching_Generative_Networks_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) [[Code]](https://github.com/mbsariyildiz/gmn-zsl) 

#### NeurIPS 2019

- Jian Ni, Shanghang Zhang, Haiyong Xie. Dual Adversarial Semantics-Consistent Network for Generalized Zero-Shot Learning.  [[CVF]](https://proceedings.neurips.cc/paper/2019/file/c46482dd5d39742f0bfd417b492d0e8e-Paper.pdf) [[No code-210311]]() 
- **VSC**: Ziyu Wan, Dongdong Chen, Yan Li, Xingguang Yan, Junge Zhang, Yizhou Yu, Jing Liao. Transductive Zero-Shot Learning with Visual Structure Constraint.   [[CVF]](https://proceedings.neurips.cc/paper/2019/file/5ca359ab1e9e3b9c478459944a2d9ca5-Paper.pdf) [[Code]](https://github.com/raywzy/VSC)
- **SGAL**: Hyeonwoo Yu, Beomhee Lee. Zero-shot Learning via Simultaneous Generating and Learning. [[CVF]](https://proceedings.neurips.cc/paper/2019/file/19ca14e7ea6328a42e0eb13d585e4c22-Paper.pdf) [[Code]](https://github.com/bogus2000/zero-shot_SGAL) 
- **SGMA**: Yizhe Zhu, Jianwen Xie, Zhiqiang Tang, Xi Peng, Ahmed Elgammal. Semantic-Guided Multi-Attention Localization for Zero-Shot Learning.  [[arXiv]](https://arxiv.org/pdf/1903.00502.pdf) [[No code-210311]]() 

#### CVPR 2018
+ **GAZSL**: Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng, Ahmed Elgammal. "A Generative Adversarial Approach for Zero-Shot Learning From Noisy Texts". [[arXiv]](https://arxiv.org/abs/1712.01381) [[Code]](https://github.com/EthanZhu90/ZSL_GAN)

      Note: This paper is one of the earlierst works working on textual-based CUBird and NABird. And the proposed two heads discriminator and the pipeline are widely accepted.



## Datasets
+ TBD
