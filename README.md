# A-machine-learning-approach-to-visual-perception-of-forest-trails-for-mobile-robots

This is reproduction based on the paper

# source
|  |  |
|:---------|:---------|
| Paper | http://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf |
| Dataset | http://people.idsia.ch/~giusti/forest/web/ |

# Abstract
Abstract — We study the problem of perceiving forest or moun- tain trails from a single monocular image acquired from the viewpoint of a robot traveling on the trail itself. Previous literature focused on trail segmentation, and used low-level features such as image saliency or appearance contrast; we propose a different approach based on a Deep Neural Network used as a supervised image classifier. By operating on the whole image at once, our system outputs the main direction of the trail compared to the viewing direction. Qualitative and quantitative results computed on a large real-world dataset (which we provide for download) show that our approach outperforms alternatives, and yields an accuracy comparable to the accuracy of humans that are tested on the same image classification task. Preliminary results on using this information for quadrotor control in unseen trails are reported. To the best of our knowledge, this is the first paper that describes an approach to perceive forest trials which is demonstrated on a quadrotor micro aerial vehicle.

# Requirements
- Python
- Numpy
- Opencv
- Chainer
- Cupy

# Simple result

| ColumnNet | Main/Accuracy | Val/Accuracy |
|:---------|:---------|:---------|
| 90 epoch | 87.59% | 83.15% |
| 540 epoch | 95.72% | 74.21% |
