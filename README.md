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
