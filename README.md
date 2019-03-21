# Semi Supervised Learning Using GANs

* SSL with GANs is found to be useful when doing classification with limited amount of labeled data.
* The unlabeled samples can be used in a semi-supervised setting to boost performance.
* Even with limited number of labeled images, the SSL GAN is able to perform better than the supervised baseline. 
* The loss function used is of the form specified in the paper "Improved Techniques for Training GANs" [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)

### Results for MNIST

| No. of labeled samples per class | Accuracy - SSL GAN | Accuracy - Supervised |
|:------------:|:------------:|:------------:|
10 | 0.7220 ± 0.0247 | 0.6403 ± 0.0203 |
50 | 0.8985 ± 0.0609 | 0.8610 ± 0.0127 |
100 | 0.9325 ± 0.0269 | 0.9218 ± 0.0067 |
250 | 0.9693 ± 0.0149 | 0.9550 ± 0.0088 |
500 | 0.9760 ± 0.0065 | 0.9698 ± 0.0034 |
750 | 0.9818 ± 0.0038 | 0.9795 ± 0.0026 | 
1000 | 0.9813 ± 0.0010 | 0.9830 ± 0.0012 |


![graph](https://raw.githubusercontent.com/theidentity/Improved-GAN-PyTorch/master/graphs/ssl_sup_compare.png "MNIST Comparison graph")


### Results for CIFAR10

| No. of labeled samples per class | Accuracy - SSL GAN | Accuracy - Supervised |
|:------------:|:------------:|:------------:|
10 | 0.3430	± 0.0552 | 0.1808 ± 0.0245 |
250 | 0.6070 ± 0.1061 |0.4288 ± 0.0229 |
1000 | 0.7655 ±	0.0389 | 0.6500	± 0.0358 |


![graph](https://raw.githubusercontent.com/theidentity/Improved-GAN-PyTorch/master/graphs/cifar_ssl_sup_compare.png "CIFAR-10 Comparison graph")


### Use Cases
* Radar Image analysis : [A Deep Convolutional Generative Adversarial Networks (DCGANs)-Based Semi-Supervised Method for Object Recognition in Synthetic Aperture Radar (SAR) Images](https://www.semanticscholar.org/paper/A-Deep-Convolutional-Generative-Adversarial-Method-Vuku%C5%A1i%C4%87-Yang/97a19fbc1cfb1afdf546624abf2a4742b3d3dd2b?navId=paper-header)
* Medical Diagnostics : [Semi-Supervised Deep Learning for Abnormality Classification in Retinal Images](https://arxiv.org/abs/1812.07832)
* In general, semi-supervised learning is useful when you have unlabeled samples (which cannot be made use of by supervised models) - usually when you cannot label large number of data - due to cost associated / unavailability of experts for the task.
