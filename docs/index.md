# Deep learning project report
## Introduction

<p align="justify">
In this blog post, we present the results of our analysis and reproduction of the paper "Soft Masking for Cost-Constrained Channel Pruning."  by R. Humble,  M. Shen, J.A. Latorre, E. Darve and J. Alvarez presented at the Computer Vision–ECCV 2022 17th European Conference. The work was conducted as the project for the CS4240 Deep Learning course 2022/23 at TU Delft.
</p>

<p align="justify">
The paper introduces a new technique for structured channel pruning which significantly accelerates inference time for convolutional neural networks (CNNs) without major drops in final test accuracy, even for high fractions of the network being pruned.
</p>

<p align="justify">
The core of the new approach relies on regularly rewiring the network sparsity, through the use of a soft mask for the network weights. This allows previously pruned channels to later be reintroduced into the network, instead of being permanently removed.
The channel pruning task is formulated as a global resource allocation problem, where the goal is to minimize the drop in accuracy while keeping the cost constraint (i.e. inference time) below a certain value.
</p>

<p align="justify">
Moreover, the paper introduces a new batch normalization scaling approach to lessen the effect of large gradient magnitudes caused by the pruning of many channels.
To verify the new method’s accuracy and inference time improvements, the authors test it on the ImageNet and PASCAL VOC datasets for ResNet, MobileNet and SSD architectures. The new method, referred to as Soft Masking for cost-constrained Channel Pruning (SMCP) outperforms prior pruning approaches (Figure 1), achieving up to an additional 20% speedup at the same Top-1  accuracy level, or up to 0.6% Top-1 accuracy improvement at the same frames per second (FPS) for ResNet50 and ResNet101 on ImageNet.
</p>

<p align="justify">
Our work focused on testing the effectiveness of the SMCP method on a scaled-down model architecture, ResNet18, and a new dataset: CIFAR10. Our goal was to evaluate whether we could obtain comparable results to those reported in the paper. We analyzed the results for different pruning ratios and learning rates **!!!add or change the hypeparams actually used!!!**. Moreover, we compared the results from the pruned ResNet18 to those obtained when training without pruning. Finally, we also performed an ablation study by training with the SMCP method, without warmup epochs. 
</p>


<p align="center" style="margin-top: 10px; margin-bottom: 10px;">
  <img src="https://raw.githubusercontent.com/NVlabs/SMCP/main/SMCP_teaser.JPG" alt="Comparison of the performance between SMCP and other channel pruning techniques. On the left plot the cost is    measured in FPS, on the right plot in FLOPs." width="50%"/>
  <p align="center"><em>Figure 1: Comparison of the performance between SMCP and other channel pruning techniques.</em></p>
</p>



## Implementation

<p align="justify">
Initially, we attempted to run the original authors’ code (available on github at https://github.com/NVlabs/SMCP), on a docker container, utilizing the environment specified by the authors. However, this approach resulted in encountering several issues related to the versions of Python packages utilized. Despite reaching out to the authors for a complete list of the package versions utilized during the project, we were not provided with such information. Many of the packages utilized in the original codebase of the paper, have since undergone changes or modifications, requiring also specific versions of other packages to run. Although we initially attempted to resolve all those issues, we soon realized it was a time-consuming and inefficient strategy. As such, we opted for reimplementing the model training component ourselves. Fortunately, we were able to reuse most of the original code for the SMCP method component, without the need for any major  change.
</p>

<p align="justify">
In our new code variant of the model architecture and training, we exploited the classes and functions of the Pytorch-lighting and the lighting-bolts packages, used also in the original codebase. We reimplemented only what was strictly necessary to conduct our analysis, eliminating any extra configuration or model personalization options.
</p>

<p align="justify">
As discussed in the Introduction section, there were two major differences between our reimplementation and the original research project. Firstly, we utilized a simpler model architecture, namely ResNet18.  This architecture is 18 layers deep, which is significantly less complex than the ResNet50 adopted by the original authors. Secondly, we utilized a   significantly smaller dataset, CIFAR10. This dataset contains only 10 classes, with 6000 images per class, in comparison to the original dataset (i.e ImageNet), which contains over a million images and around 1000 classes. The two elements together allowed for a considerably faster training, further detail will follow in the Result section.
</p>

<p align="justify">
The code was implemented on Google Colab, exploiting the computational power of the free GPU provided.
The model was trained on **!!!n!!!** epochs. Training time was around 30 minutes. For each different SMCP method hyperparameters configuration, we logged in a csv file the respective train and test loss at each epoch, as well as the Top-1 and Top-5 class accuracy. To measure the FPS, we reutilized the function provided in the authors’ codebase.
</p>

## Results

## Discussion

## Distribution of the efforts

## Conclusion

## References
Humble, R., Shen, M., Latorre, J. A., Darve, E., & Alvarez, J. (2022, November). “Soft Masking for Cost-Constrained Channel Pruning”. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XI (pp. 641-657). Cham: Springer Nature Switzerland.

