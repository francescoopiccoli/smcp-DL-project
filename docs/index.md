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
Our work focused on testing the effectiveness of the SMCP method on a scaled-down model architecture, ResNet18, and a new dataset: CIFAR10. Our goal was to evaluate whether we could obtain comparable results to those reported in the paper. We analyzed the results for different pruning ratios and learning rates. Moreover, we compared the results from the pruned ResNet18 to those obtained when training without pruning. Finally, we also performed an ablation study by training with the SMCP method, without warmup epochs. 
</p>


<p align="center" style="margin-top: 10px; margin-bottom: 10px;">
  <img src="https://raw.githubusercontent.com/NVlabs/SMCP/main/SMCP_teaser.JPG" alt="Comparison of the performance between SMCP and other channel pruning techniques. On the left plot the cost is    measured in FPS, on the right plot in FLOPs." width="50%"/>
  <p align="center"><em>Figure 1: Comparison of the performance between SMCP and other channel pruning techniques from the paper. [1] </em></p>
</p>



## Implementation

<p align="justify">
Initially, we attempted to run the original authors’ code (available on github at https://github.com/NVlabs/SMCP), on a docker container, utilizing the environment specified by the authors. However, this approach resulted in encountering several issues related to the versions of Python packages utilized. Despite reaching out to the authors for a complete list of the package versions utilized during the project, we were not provided with such information. Many of the packages utilized in the original codebase of the paper, have since undergone changes or modifications, requiring also specific versions of other packages to run. Although we initially attempted to resolve all those issues, we soon realized it was a time-consuming and inefficient strategy. As such, we opted for reimplementing the model training component ourselves. Fortunately, we were able to reuse most of the original code for the SMCP method component, without the need for any major  change.
</p>

<p align="justify">
In our new code variant of the model architecture and training, we exploited the classes and functions of the Pytorch-lighting and the lighting-bolts packages, used also in the original codebase. We reimplemented only what was strictly necessary to conduct our analysis, eliminating any extra configuration or model personalization options (which is instead present in the original codebase).
</p>

<p align="justify">
As discussed in the Introduction section, there were two major differences between our reimplementation and the original research project. Firstly, we utilized a simpler model architecture, namely ResNet18.  This architecture is 18 layers deep, which is significantly less complex than the ResNet50 adopted by the original authors. Secondly, we utilized a   significantly smaller dataset, CIFAR10. This dataset contains only 10 classes, with 6000 images per class, in comparison to the original dataset (i.e ImageNet), which contains over a million images and around 1000 classes. The two elements together allowed for a considerably faster training, further detail will follow in the Result section.
</p>

<p align="justify">
The code was implemented on Google Colab, exploiting the computational power of the free GPU provided.
The model was trained on 21 epochs. Training time was around 25/30 minutes. For each different SMCP method hyperparameters configuration, we logged in a csv file the respective train and test loss at each epoch, as well as the Top-1 and Top-5 class accuracy. To measure the FPS, we reutilized the function provided in the authors’ codebase. 

As we scaled down the model, different hyperparameters required to be reset and adapted, such changes were made in the way that seemed most apt and meaningful to us after studying the codebase, the paper and trying different setups, however we acknowledge this as a limitation given our limited expertise in the field and domain, and as we’ll discuss in the later sections we believe this could have influenced the results obtained.
</p>

## Results
### Pruning ratios 
<p align="justify">
We ran several tests comparing the performance (top-1 accuracy and FPS) for different pruning ratios (from 0% to 90%) to see how the performance varies, and we obtained the values shown in Figure 2.
The overall trend is in line with that of Figure 1, namely we observe a drop in accuracy and an increase in FPS as the pruning ratio increases. The accuracy values are very similar to those of Figure 1, the FPS values are much higher, this is due to the smaller scale of our model. 
It is hard to assess whether the increase in FPS is in the same order as the one in Figure 1, as we do not have information on the pruning ratios utilized for that figure, however the increase in FPS seems to be limited compared to the decrease in accuracy, which would not encourage the utilization of the SMCP method, except for some specific values (which could be strictly related to the dataset and hyperparameters utilized), such as from no pruning to 10% <code>pruning_ratio</code>. 
It is difficult however to draw general conclusions on whether SMCP does not scale to smaller architectures (or it does but only for very small pruning ratio values) , as we tested only a specific scenario, and many factors could have contributed to the different performance, such as the hyperparameters chosen for the smaller architecture, the fitting between the architecture chosen and the dataset, and the smaller dataset being used.
</p>

<p align="center" style="margin-top: 10px; margin-bottom: 10px;">
  <img src="https://raw.githubusercontent.com/francescoopiccoli/smcp-DL-project/main/Images/top1_fps_plot.png" alt="Top-1 Acc / FPS values for different pruning ratios.." width="45%"/>
  <p align="center"><em>Figure 2: Top-1 Acc / FPS values for different pruning ratios.</em></p>
</p>

### Ablation study: Running without warmup epochs
<p align="justify">
In the paper, before applying the SMCP method, a number of warmup epochs (around 1/9 of the total epochs) is set, during these epochs, the model is trained without applying SMCP, we tested whether removing the warmup phase would affect the accuracy and fps performance.

In particular we tested the scenario with a very high pruning_ratio (0.8), to amplify the possible difference between warmup and no warmup case. 
We obtained the following values:
</p>
<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Top-1 Acc</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>no-warmup</td>
      <td>0,7427</td>
      <td>7736</td>
    </tr>
    <tr>
      <td>warmup</td>
      <td>0,6736</td>
      <td>7737</td>
    </tr>
  </tbody>
</table>

<p align="justify">
Our ablation study showed that the classifier performs better when there is no warmup. This is not the result we expected. Since the intuition behind the warmup, to get a sense of which weights are important, is sound it was surprising to find this did not lead to improved performance. There are a few reasons why this might be the case however. Firstly, the warmup period might lead to overfitting in some way. This is not a likely reason, but the pruning could lead to increased resilience during inference. A second related reason is that learned redundant features could lead to decreased performance, and pruning could be alleviated. This result is highly unexpected, and defies expectations. Varying different hyperparameters with and without a warmup would answer some of the questions raised by this result. Since there seem to be no tradeoffs to not using a warmup, and it increases accuracy, it seems that the results of the paper could potentially have been understated. 
</p>

## Discussion and Conclusion

<p align="justify">
We managed to partially replicate some of the results in the paper, with several limitations and differences, as discussed for Figure 2 in the Implementation section. We encountered several technical issues and making the code run took plenty of time, this also limited our analysis and our ability to reproduce and investigate the paper results. Specifically, we deem that the high number of hyperparameters which required to be reset (due to the new architecture and dataset) could have influenced the results we obtained, and it would have required more tests to find proper values for such hyperparameters. 
</p>


## Distribution of the efforts
<p align="justify">
Francesco worked partly on trying to fix the dependencies and environment issues of the authors' code, in the writing of the blog post and the setup of the repository, on running the ablation study, and contributed partly on making the new model run.

Marcus worked mainly on reproducing the paper, adapting it to the new dataset, and experimenting with different models.
</p>

## References
[1] Humble, R., Shen, M., Latorre, J. A., Darve, E., & Alvarez, J. (2022, November). “Soft Masking for Cost-Constrained Channel Pruning”. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XI (pp. 641-657). Cham: Springer Nature Switzerland.
