# SCL
## Shopee Code League 2020
This repository details the competitions I participated in in Shopee Code League 2020.

### 1. Product Detection
In this competition (hosted on Kaggle), a multiple image classification model needs to be built. There are ~100k images within 42 different categories, including essential medical tools like masks, protective suits and thermometers, home & living products like air-conditioner and fashion products like T-shirts, rings, etc. For the data security purpose the category names will be desensitized. The evaluation metrics is top-1 accuracy. The competition duration was 2 weeks long.

I built a classification model via transfer learning of the EfficientNet B6 model with pretrained imagenet weights, then finetuned it by adding extra hidden layers, and unfreezing a few additional top layers for further training. The data pipeline and training was constructed with Tensorflow 2.

##### Link to competition here: https://www.kaggle.com/c/shopee-product-detection-student/overview
##### Results
My model achieved approximately 0.8 validation loss, and a 77.218% classification accuracy rate on the test set.  
Position: 195/823

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
Position: 39/356

##### What can be improved
I could have tried more models with my enlarged dataset (from text augmentation) to try to produce better ensembled outputs. Another learning point I got from the top few teams was to alter the final logits array to better model the actual test distribution, which was extremely skewed.

### 3. Marketing Analytics
In this competition (hosted on Kaggle), we were provided with data related to marketing emails (Electronic Direct Mail) that were sent to Shopee users over a certain period. Based on the data provided, we must predict whether each user will open an email sent to him/her. The evaluation metrics used was the Matthews Correlation Coefficient (MCC). The competition duration was 1 week long.

I built a Sequential Dense network with tensorflow.keras with a sigmoid activation function at the final layer to implement binary logistic regression. The data pipeline and training was constructed with Tensorflow 2. I then did an ensemble of my best 2 results and tweaked the logistic regression threshold to optimise my results. My final threshold used was in the ranges of 0.7 to 0.75.  

I also tried using the keras-tuner (https://github.com/keras-team/keras-tuner) to search for optimal hyperparameters, but it did not work out due to the lack of compute.

##### Link to competition here: https://www.kaggle.com/c/student-shopee-code-league-marketing-analytics
##### Results
Ensembled score: 0.52935  
Position: 26/368

##### What can be improved
Initially I did not update my model metrics to monitor MCC over Classification accuracy, which resulted in a huge discrepancy between my cross-validation accuracy and the actual score obtained. This took up quite a bit of my time before I realised my mistake! Also, I could have spent more time analysing the data, to remove variables that were highly correlated with each other. I could also have attempted simpler models from sklearn, such as Gradient Boosting Classifier.
