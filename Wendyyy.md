Abstract
============

   Semantic segmentation requires a high number of pixel-level annotations to learn and train accurate models. This paper reports the progress on improving the mIoU and Flops based on a small training dataset. Using several model structures and optimizing strategies, this report gives a summary and analysis on deep learning by describing and observing the characteristics of different approaches based on the experiment results. The report firstly introduces the concept and performance of baseline model, and then reviews the performance by modifying the training parameters and hidden layers based on baseline model. Secondly, the report records the performances on U-Net, FCN and other convolutional neural networks by combining several backbones. Thirdly, a higher performance solution is given by using transfer learning. Finally, the visualization result is summarized, limitation and challenges are discussed. The entire experiment results can be viewed through this link.

1.Introduction 
============

The convolutional neural network (CNN) architectures are the most common
used deep learning frameworks. The given baseline model is defined as
SegNetwork class based on CNN which includes 5 stages of hidden layers
as a backbone. For each stage, it contains convolutional layers, ReLU
layers and pooling layers; the last stage also uses dropout layers. The
forward method is defined inside of the lass, which is a mapping that
converts an input tensor into an output tensor for a prediction. It
calls all of the layers that defined in the constructor. The
understanding of each layer type's definition and usage is discussed
below:

1.1 Convolutional layer
-------------------

Convolutional layers consist a group of filters, referred as kernels,
which are applied to an input picture. The convolutional layer's output
is a feature map which represents the input picture with the filter
applied. In the baseline model, a 2D convolutional layer has been
applied on a signal input using multiple input planes with a square
kernel size, equal padding and stride.

1.2 ReLU layer
----------

ReLU layer is one of the most common choices for hidden layers, as it is
simple and effective. It reduces the probability that the computing
needed to run the neural network would rise exponentially. The
computational cost of adding more ReLUs rises linearly as the CNN's size
scales.

1.3 Pooling layer
-------------

The pooling layer decreases the spatial dimension, which facilitates
processing and uses less memory. Pooling speeds up training and also
contributes to a decrease in the number of parameters. Max pooling and
average pooling are the two primary forms of pooling. A 2D max pooling
is applied in baseline model, which operates for 2D spatial data and
picks the maximum value from the feature map. The highest value for each
input channel over an input window of the size specified by pool size is
used to downsample the input along its spatial dimensions (height and
width). Steps are taken along each dimension to move the window.

In order to minimize the size of the input before it is fed into a fully
connected layer, pooling layers are often utilised after convolutional
layers.

1.4 Dropout layer
-------------

To avoid overfitting, the dropout layer randomly converts input units to
0 with a frequency of rate for each channel during training (a channel
indicates a 2D feature map). Each single channel will be separately
zeroed out on every forwards call with a certain probability.

2.Improvement based on baseline model
===================================

We have tested modifications on various parameters and the hidden layers
based on baseline model. By doing this, we get a general idea of
choosing the right parameters for the rest of testing process.

Table 1 summarizes the mIoU results by only modifying one type of
parameter or hidden layer each time. From the table, we can clearly see
that the optimizer Adam performs much better than SGD.

![WechatIMG21](https://user-images.githubusercontent.com/64782437/176138110-ea7e69c4-0bf8-40e5-a5cc-10e342e864b7.jpeg)

For learning rates, we have tested a range of values between 1.0 and
1e-6. A strategy called step-wise decay has been applied to modify the
learning rate during training. We design this strategy to lower the
learning rate according to a predetermined rate, then decay the rate by
a number of percentage after each certain size of epochs.
Figure1 shows the changes of learning rate by using step-wise decay method. By using this
strategy, it helps the model to be converged faster and might improve
the mIoU when if meet the right rate.
Figure2 shows the result of using step-wise decay method to find the decent learning
rate[1].

![image](https://user-images.githubusercontent.com/64782437/176140763-fc73ab4c-44b2-4741-87b7-da98942c276d.png)
![image](https://user-images.githubusercontent.com/64782437/176140815-7ea86b9a-7d6a-4b6a-8dc3-eff2fbdab336.png)

From Table1, we observe that increase the batch size does not bring that much difference on the
final results. Moreover, enlarge the number of epochs can improve the final mIoU within a small range.

Besides of the small changes above, we also modify the hidden layers by
adding Batch Normalization and more convolutional layers. In certain
circumstances, Batch Normalization eliminates the necessity for Dropout
and enables us to employ considerably greater learning rates and less
cautious initialisation. According to the paper results, when apply on a
cutting-edge image classification model, Batch Normalization outperforms
the original model by a wide margin and obtains the same mIoU with 14
times smaller training steps[2]. Therefore, we insert the
functions between convolutional layers and pooling layers. As we can see
from Table1, the mIoU has been improved a bit more compare with other
modification results. Then, we add more convolutional layers and switch
the positions, but it does not perform as well as Batch Normalization.

![image](https://user-images.githubusercontent.com/64782437/176140959-02afc9ac-1f56-4421-89c2-a509256053c7.png)

After testing and observing the results by applying small changes on
baseline model, we decide to keep using Adam as the optimizer.
Table2 shows our best
mIoU achievements using ablation studies. We find that when increase the
batch size, add Batch Normalization and more convolutional layers at the
same time, we achieve the highest mIoU. As the next step, we move on to
test different networks and involve ablation studies to pick the best
performing backbone for each network.

3.Improvement based on several networks 
=====================================

The following discussion is focusing on the testing results with
different networks and backbones. We mainly focus on ResNet as our
backbone as there are enough sources online though GitHub.

3.1 FCN
---

The baseline model uses convolutional networks with fully connected
layers to get the categories of images. Considering for semantic
segmentation, we need to input the original image of any size, and the
final output is changed from a category to a category for each pixel of
the whole image. The answer to this problem is given in the paper
\"Fully Convolutional Networks for Semantic
Segmentation\"[3]. By involving the fully-connected layer,
the input to the network will accept arbitrary sizes and the convolution
will result in a probability map. Finally, solving the problem that the
size of the image after convolution is smaller than the original one,
upsampling is performed by bilinear interpolation or transposed
convolution to recover the size of the probability map to get our
segmented image.

After testing on two backbones - ResNet50 and ResNet101, FCN + ResNet101
this combination reaches an obvious improvement, which up to 0.41 in
mIoU. However, the computational cost is a bit large, which is around 6
times to the original.

3.2 U-Net
-----

In the paper \"U-Net: CNNs for Biomedical Image Segmentation\", in
addition to the down-sampling and up-sampling processes mentioned in
FCN, U-Net introduces the \"copy and crop\"[4]. This
solves the problem of information loss in the downsampling process in
FCN, which further improves the mIoU of segmentation.

According to the analysis above, we have implemented and tested the
U-Net performance based on ResNet18 and ResNet101 backbones separately.
From Table3, we can see
that U-Net + ResNet101 has a higher mIoU but also a larger computational
cost.

 ![image](https://user-images.githubusercontent.com/64782437/176141421-97d8489d-b295-40c5-9958-ff98a3531a86.png)


3.3 UPerNet
-------

Consider convolution makes the visible field of view of the network
small, the paper \"Pyramid Scene Parsing Network\" gives a multi-scale
pooling solution, which allows us to obtain more contextual information
and thus improve the segmentation mIoU[5]. We have tried
several ways to apply the PSPNet by training our data, but we could not
successfully develop and load this network. After searching on more
deliverable networks, in paper "Unified Perceptual Parsing for Scene
Understanding", we find a new network approach called UPerNet which is
built up based on the Feature Pyramid Network (FPN) and PSPNet. The
authors apply a Pyramid Pooling Module (PPM) from PSPNet on the final
layer of the backbone network[6]. Empirically, they
discover that the PPM, by offering useful global prior representations,
is extremely compatible with the FPN design.

Table3 shows the outputs by combining ResNet50 with UPerNet, which achieves a higher mIoU.
Comparing with FCN's and U-Net's computational cost, it is more Flops, but still around 5 times larger than the baseline model.

3.4 DeepLabV3
---------

While searching for a better solution, we find that the method in the
DeepLab paper has been reproduced by many people. By looking up the
problems solved of using DeepLab network, we discover that DeepLab can
fix the problem of downsampling with null convolution[7].
It also solves the limitation of blurred predicted images as the
conditional random fields of view to capture context using multiscale
null convolution (ASPP module). From paper "Lightweight semantic
segmentation algorithm based on MobileNetV3 network", we find
MobileNetV3 network is a very efficient approach. Therefore, we have
tested the results by using ResNet and also MobileNetV3 as backbone on
DeepLabV3 model[8].

From Table3, the highest mIoU is using DeepLabV3 + ResNet101, which is 0.46. However, by
comparing the Flops values, DeepLabV3 + MobileNetV3 Large is very efficient, which is 4 times smaller than the original baseline. We will discuss the computational cost more in detail below.

3.5 Overview on mIoU
----------------

From Table4, we observe
that DeepLabV3 + ResNet101 has the best performance in mIoU. Therefore,
we have involved with ablation study on this model as the testing
results showing in Table5. In conclusion, increasing the number of epochs to a
large amount will not always lead a good result. In order to increase
the performance of a model, a correct learning rate is much more
important and essential.

![image](https://user-images.githubusercontent.com/64782437/176141772-26745151-131c-4280-beea-bdd3566bc340.png)

![image](https://user-images.githubusercontent.com/64782437/176141852-9cbb3113-ad56-4513-b2b5-9e51d19377f2.png)


3.6 Overview on Flops
-----------------

![image](https://user-images.githubusercontent.com/64782437/176141907-3e2e9ee7-b1d1-42d3-9177-136b48337bbc.png)


In order to improve the total model's Flops, we have tried to modify the
backbone. From Table6,
using MobileNetV3 Large as a backbone has decreased the Flops to only
35.85G, which is 4 times smaller than the baseline. The other approach
we have discovered is ENet which is developed based on the SegNet as a
baseline. By analyzing the results from paper - "ENet: A Deep Neural
Network Architecture for Real-Time Semantic Segmentation", ENet has
fewer parameters and cheaper in computation time cost than other
networks[9]. Table6 shows the entire networks' Flops in an ascending
order. ENet is the fastest one with only 6.64G in Flops.

Moreover, by reading "Deep transfer learning for military object
recognition under small training set condition" this paper, we got some
hints to use pretrained models which have been trained on a large
dataset[10]. This technique called transfer learning. The key
advantages of transfer learning are resource conservation and increased
effectiveness while developing new models. Since the majority of the
model will have already been trained, it can also aid when only
unlabeled datasets are available for model training. After applying this
transfer learning strategy on the DeepLabV3Plus + ResNet101 model, we
achieve a higher mIoU and lower Flops (only 286.90G). More details will
be covered in the flowing session.

4.Implementation - transfer learning
==================================
![image](https://user-images.githubusercontent.com/64782437/176142123-3b123f91-05ab-4707-86e6-39e3695fb2a0.png)

epochs.](latex/img/Picture3.png){#fig:pic3 width="\\linewidth"}
After observing the above testing result, we pick the DeepLabV3Plus +
ResNet101 as our final model architecture. As we have observed our
dataset with several dataset, Cityscapes dataset is highly matched. It
is a large-scale dataset which has 5000 images with high quality
pixel-level labels. Based on the previous idea (transfer learning), we
choose a pretrained model which is trained on the Cityscapes dataset.

In our implementation, we have downloaded the model path file and load
it. Then we define a CNN as our baseline.

From Figure3 we can
clearly see that when compute around 125 epochs, the loss curve has a
large bump which affects the rest of loss values and the final mIoU
result. In the case, if we adjust the learning rate appropriately, we
might achieve a even higher mIoU in total.

5.Visualization results
=====================

The fully visualization results has been uploaded, it is visible through
this link. In Figure4, we only
pick 3 visualization images. By comparing the results, in final model
(right hand side), the objects' edges are more clear and each region is
also segmented according to the outlines.

![image](https://user-images.githubusercontent.com/64782437/176142332-8d908aac-a5ea-474a-9c3d-2984575da19c.png)

6.Conclusion
==========

We have met several limitations during the entire process:

-   Manage and optimize the augmentation functions.

-   High-level requirement on GPU while computing a large amount of
    epochs.

-   Lack of pretrained model resources from the pytorch libary.
    Especially, for KITTI, Cityscpaes those dataset.

All in all, our final solution "DeepLabV3Plus" maintains a good mIoU and
the Flops values as well. Our experimental results clearly demonstrate
the improving procedures from focusing on a baseline model up to
implementing a transfer learning strategy.

As our next step, we want to explore more strategies to achieve a better
result (mm segmentation or weakly-supervised learning strategy).

References
==========

[1] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos,
Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image
segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern
analysis and machine intelligence, 40(4):834–848, 2017. 
[2] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. PMLR, 2015. 
[3] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully
convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3431–3440, 2015. 
[4] Adam Paszke, Abhishek Chaurasia, Sangpil Kim, and Eugenio Culurciello. Enet: A deep neural network architecture for real-time semantic segmentation. arXiv preprint
arXiv:1606.02147, 2016. 
[5] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241.
Springer, 2015. 
[6] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and
Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer
vision (ECCV), pages 418–434, 2018. 
[7] Zhi Yang, Wei Yu, Pengwei Liang, Hanqi Guo, Likun Xia,
Feng Zhang, Yong Ma, and Jiayi Ma. Deep transfer learning
for military object recognition under small training set condition. Neural Computing and Applications, 31(10):6469–
6478, 2019. 
[8] Kaichao You, Mingsheng Long, Jianmin Wang, and
Michael I Jordan. How does learning rate decay help modern
neural networks? arXiv preprint arXiv:1908.01878, 2019. 
[9] Yongjun Zhang and Xia Chen. Lightweight semantic segmentation algorithm based on mobilenetv3 network. In 2020
International Conference on Intelligent Computing, Automation and Systems (ICICAS), pages 429–433, 2020. 
[10] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang
Wang, and Jiaya Jia. Pyramid scene parsing network. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2881–2890, 2017. 
