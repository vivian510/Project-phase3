# ABC MULTISTATE BANK CUSTOMER CHURN

![image](https://github.com/user-attachments/assets/a873833d-a84c-452d-bb4a-50f7021a93ca)

## BACKGROUND

>ABC Multistate bank is facing a challenge of customer churn.In the Banking industry,customer retention is more cost effective than acquiring new customer.Churn results in loss of revenue affecting the bank's reputation.
>Understanding the reason behind the churning can help the financial institutions design targeted interventions like improving the customer service.
>Data driven approach using Machine Learning are used  which provide accurate predictions and actionable insights improving customer retention.

## BUSINESS PROBLEM

>The bank aims to predict which customers are likely to churn and identify the key factors contributing to it.
>The goal is to minimize the churn rates and improving customer loyalty and satisfaction by implementing customer retention strategies

## DATA UNDERSTANDING

>The project uses data from https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset
>The dataset contains Demographic,financial and behavioural information about the customers.
>The Key features include: 
  Demographics: Age,Gender,Country
  Financial Metrics: Credit score,Credit card,Estimated salary
  Customer behaviour: Active member,tenure,Number of products
  Target variable: Churn(1=Churned,0=Retained)

### Library used

>To conduct analysis and predictions,we used python libraries and employed various libraries shown below:
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,auc,roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

### DATA CLEANING

>There was no missing values and no duplicates
>Dropping unused columns e.g Column_id

### Exploratory Data Analysis(EDA)

>Performing EDA to understand data and visualize patterns.

![image](https://github.com/user-attachments/assets/89d8e8b3-4ad2-4c7c-9f4b-c967a7a0cd7c)

>1 represents the churned and 0 represents the retained

![image](https://github.com/user-attachments/assets/4316afee-b8b0-4e46-a3c1-50d16f4e14f2)

There is a high correlation between features like age,tenure ,balance with churn variable

## PREPROCESSING DATA

a) Categorical Features

> Preprocess data by  Encoding categorical features like country and gender
> We used pd.get dummies to One Hot Encode as it converts categorical variables into multiple binary columns making them compatible with the algorithm.
 
b)  Numerical Features
> scaling numerical features,the data is scaled to ensure compatibility among features.

c) Defining Features and Targets
>  Feature and Target partitioning. Separating features(X) and Target variable(y).

d) Handling class imbalance using SMOTE

> We used SMOTE so as to handle Class imbalance which causes bias towards majority class as it dominates the dataset leading to high performance but poor performance in minority class.

e) Train_Test Split.Splitting data into training and testing sets.
We split data to train and test so as to evaluate model performance on the unseen data.
 
### MODELING

>Train multiple classification model.e.g Logistic regression,Random Forest and Decision Trees.

a) Logistic Rebression

precision    recall  f1-score   support

           0       0.80      0.78      0.79      2426
           1       0.78      0.80      0.79      2352

    accuracy                           0.79      4778
   macro avg       0.79      0.79      0.79      4778
weighted avg       0.79      0.79      0.79      4778

![image](https://github.com/user-attachments/assets/9f9d53d0-8221-4754-950f-9932540c4b23)

>The high values of True Positives(1887)  and True Negatives(1873) above shows the model is correctly predicting positives and negatives effectively.
False positives(553) and False Negatives(465) indicates where the model is making errors.

b) Decision Trees

precision    recall  f1-score   support

           0       0.81      0.77      0.79      2426
           1       0.78      0.82      0.79      2352

    accuracy                           0.79      4778
   macro avg       0.79      0.79      0.79      4778
weighted avg       0.79      0.79      0.79      4778

![image](https://github.com/user-attachments/assets/bd8d869b-c8c5-4aa3-9400-119383b28f3b)

>The high values of True Positives(1898)  and True Negatives(1895) above shows the model is correctly predicting positives and negatives effectively.
False positives(531) and False Negatives(454) indicates where the model is making errors.>

c) Random Forest Classifier

 precision    recall  f1-score   support

           0       0.86      0.85      0.86      2426
           1       0.85      0.86      0.85      2352

    accuracy                           0.86      4778
   macro avg       0.86      0.86      0.86      4778
weighted avg       0.86      0.86      0.86      4778

![image](https://github.com/user-attachments/assets/78724197-b67a-4055-91af-541d4450add7)

>The high values of True Positives(2020)  and True Negatives(2073) above shows the model is correctly predicting positives and negatives effectively.
False positives(353) and False Negatives(332) indicates where the model is making errors.

![image](https://github.com/user-attachments/assets/1d273ee9-8556-4532-944d-2dddda27e581)

>from the above,AUC_ROC is the best performing classification metrivcs with 0.93.
>ROC_AUC of 0.93  indicates  a higher Area Under the Curve (AUC)  hence showing the model has a strong ability to distinguish between the positive and negative classes.High positive rate indicate the model correctly identifies a high propotion of positive instances.
>The Random Guess provide baseline for comparison.
>

7. Model Evaluation.Evaluate model performance using classification metrics like accuracy, precision, recall, F1 score,         confusion matrix, and AUC-ROC.
