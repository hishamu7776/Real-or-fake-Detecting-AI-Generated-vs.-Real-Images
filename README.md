# Real or fake: Detecting AI-Generated vs. Authentic Images

Artificial Intelligence (AI) has made significant advances in the generation of synthetic images, generating questions about the veracity of image data. Detecting AI-generated images has become essential for maintaining data integrity. In recent years, remarkable technological advancements have enabled the generation of images of such exceptional quality that it has become difficult for human observers to distinguish them from actual photographs.

## Objective

The primary objective of this project is to investigate various classification problem approaches in an effort to gain insight into the performance of various architectures. This project seeks to analyse the results of each classification method by experimenting with a variety of models, ranging from simple convolutional neural network (CNN) architectures to more complex ones such as ALEXNET, VGG19, InceptionResnetV2, and others. Through this exhaustive investigation, a deeper understanding of each architecture's performance characteristics can be attained.

## Dataset

The CIFAKE dataset consists of 120,000 images, carefully curated to include both real and AI-generated examples. This dataset serves as the foundation for training, validation, and testing our classification models. Due to its comprehensive nature, it provides a robust benchmark for evaluating the performance of different architectures.

## Methodology

We begin by exploring the CIFAKE dataset to understand its characteristics and distributions. he dataset consisted of a total of 120,000 images, evenly divided between real and synthetic images, with 60,000 images in each category. The objective is to classify synthetic images and real images. Thus, the issue at hand can be framed as a binary classification problem, distinguishing between real and synthetic images.

### Binary Image Classification

The Binary Image Classification algorithms are used to predict the binary class or label of a given input image. A trained model utilizes learned features extracted from input images and subsequently processes them to determine whether the image belongs to a real class or a synthetic class. Various Convolutional Neural Network (CNN) architectures were utilized in this study to accomplish the objective.

### Model Selection

We experiment with various CNN architectures, CNN architectures mainly built with three type of layers, convolution layers, pooling layers and fully connected layers. The following are some of the techniques explored in this project.
*Basic CNN Methods*
*Dilated Network*
*Inception module*
*AlexNet Architecture*
*InceptionResNetV2 architecture*
*VGG19*

### Training and Evaluation

Each model is trained on a portion of the dataset and evaluated using appropriate performance metrics to assess its accuracy, precision, recall, and F1 score.

### Comparison and Analysis

We compare the performance of different architectures and analyze their strengths and weaknesses in distinguishing between real and AI-generated images.




