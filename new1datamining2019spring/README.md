## DataMining Course Project1

This is a binary classification task. Given 200k data samples as train set, each of 32 features, please predict the label (0 or 1) for the whole test set containing 180k data samples.

---

## File Structure

To run theses notebooks, you have to download the dataset and place `data/` in the root file. And then create a `result/` to store result files.

```
.root
|-data
|-result
```

---

## Methods

In this project, I use several methods to give prediction, including Decision Tree, KNN, MLP, RandomForest, SVM and XGB. All of these methods are provided by sklearn library, which is easy to use.

In order to improve accuracy, model bagging is used, which is proven to have a better outcome. Also, there are some trick about processing data, like ignoring those irrelevant attributes.



Dataset: [Google Drive](https://drive.google.com/open?id=1SH9fj0MaMB2BHZ3slGu-y7zRQb52bZFK)



For any question or suggestion, please contact me!