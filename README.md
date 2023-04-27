<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group 73: Deep Learning (CS4240) Project
The project aims at training and reproducing the Soft Masking for Cost-Constrained Channel Pruning (SMCP) method, proposed by Humble et al. (2022) in the [paper](https://arxiv.org/pdf/2211.02206.pdf) Soft Masking for Cost-Constrained Channel Pruning on a ResNet-18 architecture and on the CIFAR10 dataset. Reutilizing the code for the pruning developed by the authors, available at [this](https://github.com/NVlabs/SMCP) Github repo.

The main goals of our project were:
- Apply the SCMP method on a scaled-down architecture, and on a smaller dataset, and compare the results to those obtained in the paper
- Test different hyperparameters settings, specifically for pruning-ratio, rewiring frequencies, and learning rate schedule
- Running an ablation study by removing the warmup epochs

## Authors

Marcus Plesner

Francesco Piccoli (ID: 5848474)

Aranya Sinha

## User Manual

Download the <code>training.ipynb</code> Jupyter Notebook and upload it on Colab, alternatively run it in your editor by removing colab-spefific cells. Donwload also the [repository](https://github.com/NVlabs/SMCP) of the authors and place it in the same folder as the notebook.
