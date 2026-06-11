Evaluating Mamba-based Backbones for Land-Use Land-Cover Applications

Abstract


Introduction

Land-use land-cover (LULC) segmentation is a fine-grained inference task where images of a geographical area taken from a top-down perspective are segmented into distinct regions corresponding to distinct land-cover classes, such as vegetation, roads, and water bodies at pixel-level precision [1] [2] [3]. LULC segmentation remains a popular field of geospatial computer vision due to its importance for automated geospatial data analysis, which is difficult to do manually and requires expertise.

Image segmentation tasks in general require backbone models that can extract reliable local and global features, after which a segmentation head, such as DeepLabV3+ [4] and UPerNet [5], produces the per-pixel classification. While historically backbone models mainly utilize convolutions [6] [7] [8] with downsampling to capture multi-level features, recent methods in feature extraction integrate transformer blocks and Mamba state-space-model (SSM) blocks to enhance modelling of global dependencies between different regions of the image [9] [10] [11]. Notably, MambaOut [35] demonstrated that while removing the SSM layer from a visual Mamba model surpassed all other Mamba-based models at that time, the performance still falls short of state-of-the-art Mamba models when used as a backbone for object detection and segmentation tasks, demonstrating Mamba’s strength in autoregressive long-sequence modelling for finer-inference vision tasks.

However, the disparity that sets apart LULC segmentation from general segmentation is that LULC segmentation uses geospatial imagery taken from a top-down perspective, while general segmentation tasks such as COCO [26] and ADE20K [27] consist of images of everyday objects and environments taken from a more natural angle, which may impact the feature extraction capability of backbone models [12] [13]. Moreover, these general segmentation tasks are more often used as a benchmark for various backbone models, which puts into question how these backbone models perform in LULC segmentation. This puts into question the performance of recent Mamba-based and Mamba-hybrid backbone models when used for LULC segmentation when compared to a purely convolutional model e.g. ResNet [24] and ConvNeXt [25].

In this study, we evaluate two recent Mamba-based backbones alongside the ResNet and ConvNeXt purely convolutional backbones: VMamba [14] which incorporates cross-scan mechanism, and MambaVision [15] which incorporates the original Mamba scan alongside Transformer blocks to enhance global dependency modelling. ResNet and ConvNeXt is used as a baseline as both models use purely convolutions and residual blocks, in order to isolate the benefits that may be produced by integrating Mamba into backbone architectures.
We evaluate these backbone models on the ISPRS Potsdam dataset [28], which is popular as a geospatial segmentation benchmark. Further details of the backbone models and the Potsdam dataset are outlined in Section 3. Through this research, we hope to provide insights on the viability of current Mamba-based and Mamba-Transformer hybrid backbone models for LULC segmentation, as well as provide a baseline performance of current backbone-agnostic segmentation heads for LULC segmentation for future research and modifications.



II. Related Work (todo: add metrics for each citation, possible rewrite and expansion)

Since its introduction as a sequence modelling model in natural language processing (NLP), the Mamba architecture has been widely adopted in computer vision as a backbone for image classification and segmentation tasks.  Vision Mamba (Vim) [11] proposed a bidirectional Mamba block for efficient visual representation learning with significantly reduced memory footprint. Additionally, EfficientVMamba [16] proposed atrous selective scan for lightweight visual processing, and PlainMamba [17] explored non-hierarchical architectures, both achieving strong results on ImageNet and downstream segmentation benchmarks.

Recent studies in LULC segmentation have also experimented with integrating Mamba into their methodology due to its efficiency in capturing long-range spatial dependencies in large-scale geospatial imagery. RS-Mamba [19] proposed an omnidirectional scanning mechanism similar to VMamba specifically designed for dense prediction in large remote sensing images. UNetMamba [21] designed an efficient Mamba-based encoder-decoder architecture for high-resolution remote sensing images. Beyond LULC segmentation, Mamba has also been applied to related remote sensing tasks such as spatiotemporal change detection [18] [32] [33], demonstrating the versatility and viability of Mamba in various geospatial vision tasks.






III. Experimental Method

A. VMamba

VMamba [14] is a visual backbone that introduces the 2D Selective Scan (SS2D) module with a four-direction cross-scan mechanism to bridge 1D state space operations and 2D spatial data, enabling efficient global context modeling of feature maps with a complexity of O(HW). While the original Mamba SSM uses a single unidirectional selective-scan to compute global dependencies within a token sequence in NLP, the inductive bias is not compatible when applied to 2D spatial tokens where dependencies between tokens can come from all spatial directions. To counteract this, SS2D performs a cross scan mechanism in order to better match the inductive bias in 2D vision data. The resulting model achieves 83.9% Top-1 accuracy on ImageNet-1K, which is better than Vim [34] that attained 80.5% Top-1 accuracy on the same dataset. For segmentation tasks, VMamba is paired with UPerNet head, where VMamba-T achieves 3.4 higher mIoU than Swin-T and 1.9 higher than ConvNeXt-T at single-scale data on ADE20K.

B. MambaVision
The more recent MambaVision backbone [15] integrates minimally-modified Mamba blocks alongside convolution and transformer blocks. In the first and second level, only convolutional blocks are used in order to model fine details and short-range dependencies. Subsequent layers utilize a combination of Transformer and Mamba blocks for modelling global dependencies. Compared to the original Mamba blocks, the authors proposed replacing the causal convolution with regular convolution to better handle 2D spatial data and adding a symmetric path without Mamba to make up the information lost caused by Mamba. The outputs of the two paths are then concatenated with the Mamba-path outputs and projected to the original dimensionality. The resulting model achieves state-of-the-art performance in terms of both Top-1 accuracy and throughput in ImageNet-1K. Compared to VMamba’s performance as evaluated by the authors, MambaVision-B against VMamba-B attains 0.3 higher Top-1 accuracy (84.2 vs 83.9) with almost 6x the throughput (3670 vs 645 img/sec) with slightly higher number of parameters (97.7M vs 89.0M).
The model is also evaluated for segmentation, in which the MambaVision-B backbone with UPerNet head achieves 49.1 mIoU on ADE20K single-scale, which is better than previous convolution-based and Transformer-based backbones.
 





C. Dataset



Figure 1. The RGB image patches of the ISPRS Potsdam dataset.

The dataset used is the ISPRS Potsdam 2D Semantic labelling dataset [19], which contains 38 high-resolution image patches, each patch is a true orthophoto (TOP) with size 6000 x 6000 in pixels and has a ground sampling distance of 5 cm. The image patches can be seen in Figure 1. The dataset also contains the digital surface model
(DSM) of the image patches that provides height estimation, however, the DSM patches will not be used for training and inference, as DSMs are an entirely different image domain, which would make ImageNet transfer learning less
effective. For the TOP patches, only the red, green, and blue bands are used for model training and inference to maintain consistency in the number of color channels with ImageNet.

Table 1. The land cover classes in the Potsdam dataset labels. TODO: add class weighting of each class

The TOPs have been classified manually into the six most common land cover classes, which are shown in Table 1. The proportions of each class are obtained upon further analysis, and are also shown.

D. Dataset processing

For the train-val-test split, since there are no provided/official splits, we randomly split the dataset at the high-resolution patch level into roughly 80/10/10 ratio for training set, validation and testing set respectively with seed 42 for all experiments. This results in 30 training patches, 4 validation patches and 4 testing patches. 

However, such a high-resolution image is impractical as an inference input due to massive memory costs, therefore each high-resolution patch is patchified further into 256x256 resolution mini-patches with a stride of 128, resulting in 
63480 training, 8464 validation and 8464 testing mini–patches. The stride ensures that the resulting LULC segmentation map produced by high-resolution patch inference is not disjoint between each mini-patch. When producing the inference result of a given high-resolution patch, the logits of the overlapping regions are averaged before softmax. 

The following augmentations are applied to the image-label pair: RandomRotation(degrees=[90, 180, 270, 360]),	RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5). Additionally, for the RGB image the augmentation     PhotoMetricDistortion(brightness_delta=16,	contrast_range=(0.8,1.2), saturation_range=(0.8, 1.2), hue_delta=9) is applied. These augmentations are applied to teach the model rotation, flip, and color distortion invariance, as well as reduce overfitting. Finally, in all splits the images are normalized with ImageNet statistics, which allows for more stable training when using ImageNet-pretrained backbones.


E. Training Setup

For training all of our models, we used the mmsegmentation [29] framework which provides standardized runners for image segmentation training and evaluation. 
The backbones used are ResNet50/ResNet101 and ConvNeXt-S/-B for the purely convolutional models and non-Mamba baseline, VMamba-S/-B, for Mamba-based model and MambaVision-S/-B for Mamba-Transformer hybrid model. The segmentation heads chosen are DeepLabV3+ [30] and UPerNet [31], as DeepLabV3+ is a backbone-agnostic segmentation head widely used in various segmentation tasks with demonstrated strong performance in PASCAL VOC 2012, and UPerNet is used as the segmentation head of choice in VMamba and MambaVision original papers.
The models are trained for 100 epochs on 256×256 mini-patches with a batch size of 32 mini-patches, with early stopping (patience=10, min_delta=0.0001) monitoring the maximum validation mIoU. The loss function used is Cross-Entropy loss with inverse frequency class weighting as shown in Figure 1 to encourage the model to segment rare classes, and the optimizer used is AdamW(lr=5e-5, weight_decay=0.01) with a linear warmup of 1000 iterations followed by a scheduler that linearly decays the	 learning rate to 0. Deep supervision is employed using an FCN auxiliary head at the third level of the backbone to improve gradient flow in earlier levels, with a loss multiplier of 0.4. A dropout ratio of 0.1 is applied in both the decode and auxiliary heads, with SyncBN used throughout.. BF16 automatic mixed precision (AMP) is used to speed up computations and reduce memory usage for forward and backward passes with minimal performance impact caused by lower floating-point precision. We used a seed of 42 for all experiments to ensure reproducibility as well as ensuring no seed tuning is performed. Through the mentioned hyperparameters, we aim to make evaluation of each model as fair as possible.  

IV. Results and Discussion
Quantitative Results
	For the metrics that are used to measure the performance of various backbone-head combinations on the testing set, we used mean Intersection-over-Union, mean F1-score, overall accuracy (OA), recall score and precision score. The metrics are shown in Figure ….

	Given the metrics of each model… 





IoU, F1, OA, Recall, Prec
Inference speed (img/sec), Training memory usage, number of parameters
Figures: per class accuracy & IoU, accuracy/IoU vs training memory usage, accuracy/IoU vs inference speed


Qualitative Results
Show 1-2 examples, explain what each model does best (fine grained details?)
Ablation Study
Ablations
Turn off deep supervision


V. Conclusion
Conclude
We show that …. We conclude
Future work to build upon this research
Future work should …

References:
[1] J. Li, Y. Cai, Q. Li, M. Kou, and T. Zhang, "A review of remote sensing image segmentation by deep learning methods," Int. J. Digit. Earth, vol. 17, no. 1, 2024, Art. no. 2328827.
[2] L. Huang, B. Jiang, S. Lv, Y. Liu, and Y. Fu, "Deep-learning-based semantic segmentation of remote sensing images: A survey," IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens., vol. 17, pp. 8370–8396, 2024.

[3] J. Lv, Q. Shen, M. Lv, Y. Li, L. Shi, and P. Zhang, "Deep learning-based semantic segmentation of remote sensing images: A review," Front. Ecol. Evol., vol. 11, Art. no. 1201125, Jul. 2023.

[4] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, "Encoder-decoder with atrous separable convolution for semantic image segmentation," in Proc. Eur. Conf. Comput. Vis. (ECCV), Munich, Germany, Sep. 2018, pp. 801–818.

[5] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, "Unified perceptual parsing for scene understanding," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Salt Lake City, UT, USA, Jun. 2018, pp. 864–872.

[6] A. Younesi, M. Ansari, M. Fazli, A. Ejlali, M. Shafique, and J. Henkel, "A comprehensive survey of convolutions in deep learning: Applications, challenges, and future trends," IEEE Access, vol. 12, pp. 41180–41218, 2024.

[7] I. Ulku and E. Akagunduz, "A survey on deep learning-based architectures for semantic segmentation on 2D images," Appl. Artif. Intell., vol. 36, no. 1, 2022, Art. no. 2032924.

[8] L. Jiao, M. Wang, X. Liu, L. Li, F. Liu, Z. Feng, S. Yang, and B. Hou, "Multiscale deep learning for detection and recognition: A comprehensive survey," IEEE Trans. Neural Netw. Learn. Syst., vol. 36, no. 4, pp. 5900–5920, 2024.

[9] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical vision transformer using shifted windows," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Montreal, QC, Canada, Oct. 2021, pp. 10012–10022.

[10] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," in Proc. Conf. Lang. Model. (COLM), 2024.

[11] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," in Proc. Int. Conf. Mach. Learn. (ICML), 2024.

[12] J. Kang, S. Tariq, H. Oh, and S. S. Woo, "A survey of deep learning-based object detection methods and datasets for overhead imagery," IEEE Access, vol. 10, pp. 20118–20134, 2022.

[13] M. Zhang, H. Singh, L. Chok, and R. Chunara, "Segmenting across places: The need for fair transfer learning with satellite imagery," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), New Orleans, LA, USA, Jun. 2022, pp. 2915–2924.

[14] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, and Y. Liu, "VMamba: Visual state space model," in Adv. Neural Inf. Process. Syst. (NeurIPS), Vancouver, Canada, Dec. 2024.

[15] A. Hatamizadeh and J. Kautz, "MambaVision: A hybrid Mamba-Transformer vision backbone," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2025.

[16] X. Pei, T. Huang, and C. Xu, "EfficientVMamba: Atrous selective scan for lightweight visual Mamba," in Proc. 39th AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 6, 2025.

[17] C. Yang, Z. Chen, M. Espinosa, L. Ericsson, Z. Wang, J. Liu, and E. J. Crowley, "PlainMamba: Improving non-hierarchical Mamba in visual recognition," in Proc. British Mach. Vis. Conf. (BMVC), 2024.

[18] H. Chen, J. Song, C. Han, J. Xia, and N. Yokoya, "ChangeMamba: Remote sensing change detection with spatiotemporal state space model," IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–20, 2024, Art no. 4409720.

[19] S. Zhao, H. Chen, X. Zhang, P. Xiao, L. Bai, and W. Ouyang, "RS-Mamba for large remote sensing image dense prediction," IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–14, 2024.

[20] X. Ma, X. Zhang, and M.-O. Pun, "RS3Mamba: Visual state space model for remote sensing image semantic segmentation," IEEE Geosci. Remote Sens. Lett., vol. 21, pp. 1–5, 2024.

[21] E. Zhu, Z. Chen, D. Wang, H. Shi, X. Liu, and L. Wang, "UNetMamba: An efficient UNet-like Mamba for semantic segmentation of high-resolution remote sensing images," IEEE Geosci. Remote Sens. Lett., vol. 22, pp. 1–5, 2024.

[22] Q. Zhu, Y. Fang, Y. Cai, C. Chen, and L. Fan, "Rethinking scanning strategies with Vision Mamba in semantic segmentation of remote sensing imagery: An experimental study," IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 17, pp. 18223–18234, 2024.

[23] F. Rottensteiner, G. Sohn, M. Gerke, J. D. Wegner, U. Breitkopf, and J. Jung, "Results of the ISPRS benchmark on urban object detection and 3D building reconstruction," ISPRS J. Photogramm. Remote Sens., vol. 93, pp. 115–128, 2014.

[24] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 2015, arXiv. doi: 10.48550/ARXIV.1512.03385.

[25] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” 2022, arXiv. doi: 10.48550/ARXIV.2201.03545. 

[26] T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” 2014, arXiv. doi: 10.48550/ARXIV.1405.0312. 

[27] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, pages 5122–5130, 2017.

[28] Isprs potsdam dataset. https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx 

[29] ‘MMSegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark’, 2020. [Online]. Available: https://github.com/open-mmlab/mmsegmentation.

[30] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation,” 2018, arXiv. doi: 10.48550/ARXIV.1802.02611. 

[31] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified Perceptual Parsing for Scene Understanding,” 2018, arXiv. doi: 10.48550/ARXIV.1807.10221. 

[32] T. Gozalie, A. S. Karnyoto, E. Irwansyah, and B. Pardamean, “ChangeMambaVision: Adapting Mambavision for Building Change Detection,” 2026 International Conference on Current Research in Artificial Intelligence and Data Science (ICCRAIDS). IEEE, pp. 1–7, Apr. 10, 2026. doi: 10.1109/iccraids67816.2026.11519636. 

[33] H. Zhang, K. Chen, C. Liu, H. Chen, Z. Zou, and Z. Shi, “CDMamba: Incorporating Local Clues into Mamba for Remote Sensing Image Binary Change Detection,” 2024, arXiv. doi: 10.48550/ARXIV.2406.04207. 

[34] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. In ICML, 2024.

[35] W. Yu and X. Wang, ‘MambaOut: Do We Really Need Mamba for Vision?’, arXiv[cs.CV]. 2024.
