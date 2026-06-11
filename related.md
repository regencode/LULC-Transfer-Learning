II. Related Work

Since Mamba was first used for vision tasks, researchers have studied whether Mamba provides 
a performance boost compared to standard convolutional models. Yu and Wang 35 showed through MambaOut that 
Mamba-based models do not outperform convolutions on image classification, but they do 
show clear gains on detection and segmentation tasks where long-range dependencies matter. 
Hong et al. X also compared VMamba against carefully matched convolutional networks and found 
that VMamba's design works better than gated convolutions when the overall architecture is 
kept the same.

Mamba has been increasingly popular in remote sensing vision tasks.
Bao et al. X surveyed Vision Mamba methods applied to remote sensing, 
covering different backbone designs and scanning strategies for segmentation, object detection, and change detection. Zhu et al. 22 focused specifically on scanning strategies, testing different scanning 
directions for Vision Mamba on remote sensing segmentation datasets and showing that the choice of 
scan direction affects segmentation quality.

The closest work to ours is Wasalathilaka et al
. X, who benchmarked VMamba, MambaVision, and Spatial-Mamba on LoveDA 
and ISPRS Potsdam using a single fixed decoder. They found that scaling up within 
the same backbone family gives only small improvements, and that boundary errors are the 
main source of mistakes. However, since they used only one decoder, it 
remains unclear whether their findings hold across different segmentation heads.
Additionally, we paired the chosen backbone models with widely-used decoders in DeepLabV3+ and UPerNet 
which are often used for subsequent research in the development of new backbone architectures as well
as actual applications due to accessibility and proven performance in various segmentation tasks.

In this paper, 
we build on their work by testing Mamba-based backbones against convolutional baselines (
ResNet, ConvNeXt) with two segmentation heads (DeepLabV3+ and UPerNet). By 
changing both the backbone and the decoder, we can see whether Mamba's 
advantages stay consistent across different segmentation heads.
