Land-use land-cover (LULC) segmentation is a fine-grained inference task where given a top-down (true orthophoto) image of a geographical area, the entire image is segmented into distinct regions corresponding to distinct land-cover classes, such as vegetation, roads, and water bodies [1] [2] [3].

Image segmentation tasks in general require backbone models that can extract reliable local and global features, after which a segmentation head, such as DeepLabV3+ [4] and UPerNet [5], produces the per-pixel classification. While historically backbone models mainly utilize convolutions [6] [7] [8] with downsampling to capture multi-level features, recent methods in feature extraction integrate transformer- and Mamba-based blocks to enhance modelling of global dependencies between different regions of the image [9] [10] [11].

However, one disparity that sets apart LULC segmentation from general segmentation is that LULC segmentation uses geospatial imagery taken from a top-down perspective, while general segmentation tasks such as COCO [coco_dataset] and Cityscapes [cityscapes_dataset] consist of images of everyday objects and environments taken from a more natural angle, which may impact the feature extraction capability of backbone models [12] [13]. Moreover, these general segmentation tasks are more often used as a benchmark for various backbone models, which puts into question how these backbone models perform in LULC segmentation.

In this study, we evaluate two recent Mamba-based backbones alongside the ResNet [resnet] backbone: VMamba [14] which incorporates cross-scan mechanism, and MambaVision [15] which incorporates the original Mamba scan alongside Transformer blocks to enhance global dependency modelling. Through this research, we hope to provide insights on the viability of current Mamba-based backbone models for LULC segmentation.

II. Related Work

Since its introduction, the Mamba architecture has been widely adopted in computer vision as a backbone for image classification and segmentation tasks. VMamba [14] introduced a cross-scan mechanism to handle 2D spatial dependencies within the selective state space framework, demonstrating competitive performance on ImageNet and ADE20K. Vision Mamba (Vim) [11] proposed a bidirectional Mamba block for efficient visual representation learning with significantly reduced memory footprint. MambaVision [15] further advanced this direction by integrating Mamba with Transformer self-attention in a hybrid architecture. Additionally, EfficientVMamba [16] proposed atrous selective scan for lightweight visual processing, and PlainMamba [17] explored non-hierarchical architectures, both achieving strong results on ImageNet and downstream segmentation benchmarks.

Recent studies in LULC segmentation have also experimented with integrating Mamba into their methodology due to its efficiency in capturing long-range spatial dependencies in large-scale geospatial imagery. RS-Mamba [19] proposed an omnidirectional scanning mechanism specifically designed for dense prediction in large remote sensing images. RS3Mamba [20] introduced a visual state space model for remote sensing image semantic segmentation. UNetMamba [21] designed an efficient Mamba-based encoder-decoder architecture for high-resolution remote sensing images, while Zhu et al. [22] conducted an experimental study rethinking scanning strategies with Vision Mamba for remote sensing semantic segmentation. Beyond LULC segmentation, Mamba has also been applied to related remote sensing tasks such as spatiotemporal change detection [18], demonstrating the versatility of state space models in geospatial analysis.

III. Experimental Method

A. Backbone Models

VMamba [14] is a visual backbone that introduces the 2D Selective Scan (SS2D) module with a four-direction cross-scan mechanism to bridge 1D state space operations and 2D spatial data, enabling efficient global context modeling with linear computational complexity.

The more recent MambaVision backbone [15] integrates Mamba-based selective state space blocks in early stages with Transformer self-attention in later stages, combining the computational efficiency of Mamba with the global dependency modeling capability of attention mechanisms.

B. Dataset

Potsdam is a widely-used benchmark dataset released as part of the ISPRS 2D Semantic Labeling Challenge [23]. It consists of 38 high-resolution true orthophoto tiles at 5 cm ground sampling distance, each 6000 x 6000 pixels, with annotations for six land-cover classes: impervious surfaces, building, low vegetation, tree, car, and clutter/background.

C. Training Setup

For this research, the hyperparameters used is as follows:

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
