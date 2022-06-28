Introduction {#sec:intro}
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

Convolutional layer
-------------------

Convolutional layers consist a group of filters, referred as kernels,
which are applied to an input picture. The convolutional layer's output
is a feature map which represents the input picture with the filter
applied. In the baseline model, a 2D convolutional layer has been
applied on a signal input using multiple input planes with a square
kernel size, equal padding and stride.

ReLU layer
----------

ReLU layer is one of the most common choices for hidden layers, as it is
simple and effective. It reduces the probability that the computing
needed to run the neural network would rise exponentially. The
computational cost of adding more ReLUs rises linearly as the CNN's size
scales.

Pooling layer
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

Dropout layer
-------------

To avoid overfitting, the dropout layer randomly converts input units to
0 with a frequency of rate for each channel during training (a channel
indicates a 2D feature map). Each single channel will be separately
zeroed out on every forwards call with a certain probability.

Improvement based on baseline model {#sec:improving process}
===================================

We have tested modifications on various parameters and the hidden layers
based on baseline model. By doing this, we get a general idea of
choosing the right parameters for the rest of testing process.

Table 1 summarizes the mIoU results by only modifying one type of
parameter or hidden layer each time. From the table, we can clearly see
that the optimizer Adam performs much better than SGD.

::: {#tab:t1}
  Modification     mIoU
  --------------- ------
  Optimizer       
  Adam             0.28
  SGD              0.02
  Learning rate   
  3e-4             0.28
  3e-3             0.05
  3e-5             0.18
  1e-4             0.24
  Batch size      
  4                0.28
  16               0.25
  32               0.21
  Epochs          
  180              0.28
  500              0.28
  1000             0.30
  Hidden layer    
  BatchNorm2d      0.29
  Conv2d           0.29

  : Testing on several parameter values.
:::

For learning rates, we have tested a range of values between 1.0 and
1e-6. A strategy called step-wise decay has been applied to modify the
learning rate during training. We design this strategy to lower the
learning rate according to a predetermined rate, then decay the rate by
a number of percentage after each certain size of epochs.
[1](#fig:pic1){reference-type="ref" reference="fig:pic1"} shows the
changes of learning rate by using step-wise decay method. By using this
strategy, it helps the model to be converged faster and might improve
the mIoU when if meet the right rate.
[2](#fig:pic2){reference-type="ref" reference="fig:pic2"} shows the
result of using step-wise decay method to find the decent learning
rate[@you2019does].

![Learning rate decays each 10
epochs.](latex/img/Picture1.png){#fig:pic1 width="\\linewidth"}

![A comparison between decaying the learning rate and a decent learning
rate.](latex/img/Picture2.png){#fig:pic2 width="\\linewidth"}

From [1](#tab:t1){reference-type="ref" reference="tab:t1"}, we observe
that increase the batch size does not bring that much difference on the
final results. Moreover, enlarge the number of epochs can improve the
final mIoU within a small range.

Besides of the small changes above, we also modify the hidden layers by
adding Batch Normalization and more convolutional layers. In certain
circumstances, Batch Normalization eliminates the necessity for Dropout
and enables us to employ considerably greater learning rates and less
cautious initialisation. According to the paper results, when apply on a
cutting-edge image classification model, Batch Normalization outperforms
the original model by a wide margin and obtains the same mIoU with 14
times smaller training steps[@ioffe2015batch]. Therefore, we insert the
functions between convolutional layers and pooling layers. As we can see
from Table1, the mIoU has been improved a bit more compare with other
modification results. Then, we add more convolutional layers and switch
the positions, but it does not perform as well as Batch Normalization.

::: {#tab:t2}
  Increase batch   Conv2d   BatchNorm2d   mIoU
  ---------------- -------- ------------- ------
  N                N        Y             0.29
  N                Y        Y             0.32
  Y                N        Y             0.32
  Y                Y        Y             0.33

  : Ablation study. 3 approaches to improve baseline mIoU.
:::

After testing and observing the results by applying small changes on
baseline model, we decide to keep using Adam as the optimizer.
[2](#tab:t2){reference-type="ref" reference="tab:t2"} shows our best
mIoU achievements using ablation studies. We find that when increase the
batch size, add Batch Normalization and more convolutional layers at the
same time, we achieve the highest mIoU. As the next step, we move on to
test different networks and involve ablation studies to pick the best
performing backbone for each network.

Improvement based on several networks {#sec:progress2}
=====================================

The following discussion is focusing on the testing results with
different networks and backbones. We mainly focus on ResNet as our
backbone as there are enough sources online though GitHub.

FCN
---

The baseline model uses convolutional networks with fully connected
layers to get the categories of images. Considering for semantic
segmentation, we need to input the original image of any size, and the
final output is changed from a category to a category for each pixel of
the whole image. The answer to this problem is given in the paper
\"Fully Convolutional Networks for Semantic
Segmentation\"[@long2015fully]. By involving the fully-connected layer,
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

U-Net
-----

In the paper \"U-Net: CNNs for Biomedical Image Segmentation\", in
addition to the down-sampling and up-sampling processes mentioned in
FCN, U-Net introduces the \"copy and crop\"[@ronneberger2015u]. This
solves the problem of information loss in the downsampling process in
FCN, which further improves the mIoU of segmentation.

According to the analysis above, we have implemented and tested the
U-Net performance based on ResNet18 and ResNet101 backbones separately.
From [3](#tab:t3){reference-type="ref" reference="tab:t3"}, we can see
that U-Net + ResNet101 has a higher mIoU but also a larger computational
cost.

::: {#tab:t3}
  Network         Backbone             mIoU   Flops
  --------------- -------------------- ------ ---------
  Baseline        /                    0.28   132.85G
  ENet            /                    0.32   6.64G
  UNet            ResNet18             0.31   615.83G
  UNet            ResNet101            0.33   767.33G
  FCN             ResNet50             0.40   530.28G
  FCN             ResNet101            0.41   808.52G
  UperNet         ResNet50             0.43   662.16G
  UperNet         ResNet101            0.39   752.64G
  DeepLabV3       ResNet50             0.44   620.55G
  DeepLabV3       ResNet101            0.46   898.79G
  DeepLabV3       MobileNet V3 Large   0.34   35.85G
  DeepLabV3Plus   ResNet101            0.50   286.90G

  : Overview on different networks with several backbones.
:::

UPerNet
-------

Consider convolution makes the visible field of view of the network
small, the paper \"Pyramid Scene Parsing Network\" gives a multi-scale
pooling solution, which allows us to obtain more contextual information
and thus improve the segmentation mIoU[@zhao2017pyramid]. We have tried
several ways to apply the PSPNet by training our data, but we could not
successfully develop and load this network. After searching on more
deliverable networks, in paper "Unified Perceptual Parsing for Scene
Understanding", we find a new network approach called UPerNet which is
built up based on the Feature Pyramid Network (FPN) and PSPNet. The
authors apply a Pyramid Pooling Module (PPM) from PSPNet on the final
layer of the backbone network[@xiao2018unified]. Empirically, they
discover that the PPM, by offering useful global prior representations,
is extremely compatible with the FPN design.

[3](#tab:t3){reference-type="ref" reference="tab:t3"} shows the outputs
by combining ResNet50 with UPerNet, which achieves a higher mIoU.
Comparing with FCN's and U-Net's computational cost, it is more Flops,
but still around 5 times larger than the baseline model.

DeepLabV3
---------

While searching for a better solution, we find that the method in the
DeepLab paper has been reproduced by many people. By looking up the
problems solved of using DeepLab network, we discover that DeepLab can
fix the problem of downsampling with null convolution[@chen2017deeplab].
It also solves the limitation of blurred predicted images as the
conditional random fields of view to capture context using multiscale
null convolution (ASPP module). From paper "Lightweight semantic
segmentation algorithm based on MobileNetV3 network", we find
MobileNetV3 network is a very efficient approach. Therefore, we have
tested the results by using ResNet and also MobileNetV3 as backbone on
DeepLabV3 model[@9402816].

From [3](#tab:t3){reference-type="ref" reference="tab:t3"}, the highest
mIoU is using DeepLabV3 + ResNet101, which is 0.46. However, by
comparing the Flops values, DeepLabV3 + MobileNetV3 Large is very
efficient, which is 4 times smaller than the original baseline. We will
discuss the computational cost more in detail below.

Overview on mIoU
----------------

From [4](#tab:t4){reference-type="ref" reference="tab:t4"}, we observe
that DeepLabV3 + ResNet101 has the best performance in mIoU. Therefore,
we have involved with ablation study on this model as the testing
results showing in [5](#tab:t5){reference-type="ref"
reference="tab:t5"}. In conclusion, increasing the number of epochs to a
large amount will not always lead a good result. In order to increase
the performance of a model, a correct learning rate is much more
important and essential.

::: {#tab:t4}
  Network         Backbone             mIoU
  --------------- -------------------- ------
  Baseline        /                    0.28
  UNet            ResNet18             0.31
  ENet            /                    0.32
  UNet            ResNet101            0.33
  DeepLabV3       MobileNet V3 Large   0.34
  UPerNet         ResNet101            0.39
  FCN             ResNet50             0.40
  FCN             ResNet101            0.41
  UPerNet         ResNet50             0.43
  DeepLabV3       ResNet50             0.44
  DeepLabV3       ResNet101            0.46
  DeepLabV3Plus   ResNet101            0.50

  : Network mIoU. mIoU comparison in the ascending order.
:::

::: {#tab:t5}
  Adjust learning rate   Increase epochs   Backbone   mIoU      
  ---------------------- ----------------- ---------- ------ -- --
  N                      N                 Y          0.43      
  N                      Y                 Y          0.42      
  Y                      N                 Y          0.46      
  Y                      Y                 Y          0.40      

  : Ablation Study. DeepLabV3 network with ResNet101 as backbone.
:::

Overview on Flops
-----------------

::: {#tab:t6}
  Network         Backbone             Flops
  --------------- -------------------- ---------
  ENet            /                    6.64G
  DeepLabV3       MobileNet V3 Large   35.85G
  Baseline        /                    132.85G
  DeepLabV3Plus   ResNet101            286.90G
  FCN             ResNet50             530.28G
  UNet            ResNet18             615.83G
  DeepLabV3       ResNet50             620.55G
  UPerNet         ResNet50             662.16G
  UPerNet         ResNet101            752.64G
  UNet            ResNet101            767.33G
  FCN             ResNet101            808.52G
  DeepLabV3       ResNet101            898.79G

  : Network Flops. Flops comparison in the ascending order.
:::

In order to improve the total model's Flops, we have tried to modify the
backbone. From [6](#tab:t6){reference-type="ref" reference="tab:t6"},
using MobileNetV3 Large as a backbone has decreased the Flops to only
35.85G, which is 4 times smaller than the baseline. The other approach
we have discovered is ENet which is developed based on the SegNet as a
baseline. By analyzing the results from paper - "ENet: A Deep Neural
Network Architecture for Real-Time Semantic Segmentation", ENet has
fewer parameters and cheaper in computation time cost than other
networks[@paszke2016enet]. [6](#tab:t6){reference-type="ref"
reference="tab:t6"} shows the entire networks' Flops in an ascending
order. ENet is the fastest one with only 6.64G in Flops.

Moreover, by reading "Deep transfer learning for military object
recognition under small training set condition" this paper, we got some
hints to use pretrained models which have been trained on a large
dataset[@yang2019deep]. This technique called transfer learning. The key
advantages of transfer learning are resource conservation and increased
effectiveness while developing new models. Since the majority of the
model will have already been trained, it can also aid when only
unlabeled datasets are available for model training. After applying this
transfer learning strategy on the DeepLabV3Plus + ResNet101 model, we
achieve a higher mIoU and lower Flops (only 286.90G). More details will
be covered in the flowing session.

Implementation - transfer learning {#sec:code implementation}
==================================

![A graph of the loss curve according to
epochs.](latex/img/Picture3.png){#fig:pic3 width="\\linewidth"}

After observing the above testing result, we pick the DeepLabV3Plus +
ResNet101 as our final model architecture. As we have observed our
dataset with several dataset, Cityscapes dataset is highly matched. It
is a large-scale dataset which has 5000 images with high quality
pixel-level labels. Based on the previous idea (transfer learning), we
choose a pretrained model which is trained on the Cityscapes dataset.

In our implementation, we have downloaded the model path file and load
it. Then we define a CNN as our baseline.

From [3](#fig:pic3){reference-type="ref" reference="fig:pic3"} we can
clearly see that when compute around 125 epochs, the loss curve has a
large bump which affects the rest of loss values and the final mIoU
result. In the case, if we adjust the learning rate appropriately, we
might achieve a even higher mIoU in total.

Visualization results {#sec:results}
=====================

The fully visualization results has been uploaded, it is visible through
this
[link](https://drive.google.com/drive/folders/1-XCCT6df10bgXP7mcUvbMbLsS9FoHdKM?usp=sharing).
In [4](#fig:pic4){reference-type="ref" reference="fig:pic4"}, we only
pick 3 visualization images. By comparing the results, in final model
(right hand side), the objects' edges are more clear and each region is
also segmented according to the outlines.

![Segmentation results on U-Net, DeepLabV3+ResNet101,
DeepLabV3Plus+ResNet101 (from left to
right).](latex/img/Picture4.png){#fig:pic4 width="\\linewidth"}

Conclusion {#sec:conclusion}
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
