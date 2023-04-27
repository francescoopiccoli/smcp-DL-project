<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group 73: Deep Learning (CS4240) Project
The project aims at reproducing the results of the <em>Soft Masking for Cost-Constrained Channel Pruning (SMCP)</em> method, proposed by Humble et al. (2022) in the [paper](https://arxiv.org/pdf/2211.02206.pdf) <em>"Soft Masking for Cost-Constrained Channel Pruning"</em> on a scaled-down architecture, specifically: a ResNet-18 model trained and tested on the CIFAR10 dataset. We utilized the code for the pruning developed by the authors, available at [this](https://github.com/NVlabs/SMCP) Github repo.

The main goals of our project are:
- Apply the SCMP method on a scaled-down architecture, and on a smaller dataset, and compare the results to those obtained in the paper.
- Test different hyperparameters settings, specifically for pruning-ratio, rewiring frequencies, and learning rate schedule, compare the results.
- Running an ablation study by removing the warmup epochs.

## Authors

Marcus Plesner

Francesco Piccoli (ID: 5848474)

Aranya Sinha

## User Manual

Download the <code>training.ipynb</code> Jupyter Notebook and upload it on Colab, alternatively run it in your editor by removing colab-specific code cells, download also the [repository](https://github.com/NVlabs/SMCP) of the authors and place it in the same folder as the notebook, in case you are running from colab, upload the original authors' repository folder on your google drive.

## Results
![a](https://raw.githubusercontent.com/francescoopiccoli/smcp-DL-project/main/Images/top1_acc_pruning_ratio.png) | ![a](https://raw.githubusercontent.com/francescoopiccoli/smcp-DL-project/main/Images/top5_acc_pruning_ratio.png)

Results obtained for different pruning ratios.
