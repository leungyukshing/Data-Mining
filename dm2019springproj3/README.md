## DataMining Course Project3

This is an image classification task. Given 21048 training images, please predict the cloth label for 4500 test images.

---

## File Structure

To run notebooks in this repository, you have to download dataset first and then place `data/` 

Dataset: [Google Drive](https://drive.google.com/open?id=1Zhl1NtHnMjW8G0-5o1pKC_faqtvSPKT8)

Our structure is:

```
:.
|-- data
|-- model
|-- result
```

---

## Used Network

I have tried various network in this project in order to reach a better performance. And the result shows that I can get my best score by using ResNet-50 with pretrained parameters and some data augmentation.

Pretrained parameters can be obtained by setting pretrain to true. Here I focus on data augmentation.

---

## Data Augmentation

We need to specify the size of input. Because the provided data is larger than the standard input size of ResNet-50, we need to adjust the network to receive a 350*350 input picture. Also, by glancing the sample first, we know that the cloth is in the middle of the picture, so it's better to crop the central part of a given picture.

---

## Hyper-parameters

I use Adam as the optimizer in this project, and learning-rate decay is implemented by using lr-scheduler. Because pretrained parameters are used, the network converges quickly, costing about 15 epochs to reach an accuracy about 90% on the validation set.

---

## Future Work

We may set some fixed point concerning the cloth in order to make a classification more accurately. But entering these fixed points' coordinates into the network, we may get a better score!



For any question or advice, please contact me!