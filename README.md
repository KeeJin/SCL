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

### 2. Sentiment Analysis
In this competition (hosted on Kaggle), a multiple product review sentiment classification model needs to be built. There are ~150k product reviews from different categories, including electronics, furniture, home & living products like air-conditioner and fashion products like T-shirts, rings, etc. For data security purposes, the review ids will be desensitized. The evaluation metrics is top-1 accuracy. The competition duration was 2 weeks long.

I tried building 3 sentence classification models, first with a basic LSTM model, followed by a BERT model (transfer learning) and finally the XLM-RoBERTa model (transfer learning). The data pipeline and training was constructed with Tensorflow 2. I started out the competition not knowing about text augmentation, and only figured out how to implement it when I started on the XLM-RoBERTa model. I used the nlpaug library (https://pypi.org/project/nlpaug/) to generate contextualised word embeddings.

##### Link to competition here: https://www.kaggle.com/c/student-shopee-code-league-sentiment-analysis
##### Results
My best model was the XLM-RoBERTa model with text augmentation:
LSTM score: 0.21174
BERT score: 0.44205
XLM-RoBERTa (with text augmentation) score: 0.52550

##### What can be improved
I could have tried more models with my enlarged dataset (from text augmentation) to try to produce better ensembled outputs. Another learning point I got from the top few teams was to alter the final logits array to better model the actual test distribution, which was extremely skewed.
