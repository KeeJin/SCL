# SCL
## Shopee Code League 2020
This repository details the competitions I participated in in Shopee Code League 2020.

### 1. Product Detection
In this competition (hosted on Kaggle), a multiple image classification model needs to be built. There are ~100k images within 42 different categories, including essential medical tools like masks, protective suits and thermometers, home & living products like air-conditioner and fashion products like T-shirts, rings, etc. For the data security purpose the category names will be desensitized. The evaluation metrics is top-1 accuracy. The competition duration was 2 weeks long.

I built a classification model via transfer learning of the EfficientNet B6 model with pretrained imagenet weights, then finetuned it by adding extra hidden layers, and unfreezing a few additional top layers for further training. The data pipeline and training was constructed with Tensorflow 2.

##### Link to competition here: https://www.kaggle.com/c/shopee-product-detection-student/overview
##### Results
My model achieved approximately 0.8 validation loss, and a 77.218% classification accuracy rate on the test set.

##### What can be improved
To improve on my classification accuracy, I could implement a stacked ensemble model with a bilinear layer to combine softmax results from multiple results. Also, throughout the competition, I was using Google Colab's free backend GPU. Learning how to use their backend TPU instead would have reduced my training times substantially, and may allow me to further train and finetune my model.
