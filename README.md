# Private Data Generation Toolbox

The goal of this toolbox is to make private generation of synthetic data samples accessible to machine learning practitioners. It currently implements 3 state of the art generative models that can generate differentially private synthetic
data. We evaluate the models on 7 public datasets from domains where privacy of sensitive data is paramount. Users can benchmark the models on the existing datasets or feed a new sensitive dataset as an input and get a synthetic dataset 
as the output which can be distributed to third parties with strong differential privacy guarantees.


## Models : 
**PATE-GAN** : PATE-GAN : Generating Synthetic Data with Differential Privacy Guarantees. ICLR 2019

**DP-WGAN** : Implementation of private Wasserstein GAN using noisy gradient descent moments accountant. 

**CT-GAN** : To be added 

## Dataset description :

**Cervical Cancer** : This dataset focuses on the prediction of indicators/diagnosis of cervical cancer. The features cover demographic information, habits, and historic medical records.
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29#

**Epileptic Seizure** : This dataset is a pre-processed and re-structured/reshaped version of a very commonly used dataset featuring epileptic seizure detection.
https://www.kaggle.com/harunshimanto/epileptic-seizure-recognition

**NHANES Diabetes** : National Health and Nutrition Examination Survey (NHANES) questionnaire is used to predict the onset of type II diabetes.
https://github.com/semerj/NHANES-diabetes/tree/master/data

**Adult Census** : The dataset comprises of census attributes like age, gender, native country etc and the goal is to predict whether a person earns more than $ 50k a year or not.
https://archive.ics.uci.edu/ml/datasets/adult

**Give Me Some Credit** : Historical data are provided on 250,000 borrowers and task is to help in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
https://www.kaggle.com/c/GiveMeSomeCredit/data

**Home Credit Default Risk** : Home Credit makes use of a variety of alternative data including telco and transactional information along with the client's past financial record to predict their clients' repayment abilities.
https://www.kaggle.com/c/home-credit-default-risk/data

**Adult Categorical** : This dataset is the same as the Adult Census dataset, but the feature values for continuous attributes are put in buckets. We evaluate Private-PGM's performance on this dataset.
https://github.com/ryan112358/private-pgm/tree/master/data

The datasets can be downloaded to the /data folder by using the ```download_datasets.sh``` and can be preprocessed using the scripts in the /preprocess folder. Preprocessing is data set specific and mostly involves dealing with missing values, normalization, encoding
of attribute values, splitting data into train and test etc.

Example :   
   ```sh download_datasets.sh cervical```  
            ```python preprocessing/preprocess_cervical.py```

## Downstream classifiers :
Classifiers used are 
- Logistic Regression (LogisticRegression)
- Random Forests (RandomForestClassifier) 
- Gaussian Naive Bayes (GaussianNB)
- Bernoulli Naive Bayes (BernoulliNB)
- Linear Support Vector Machine (svm)
- Decision Tree (DecisionTree) 
- Linear Discriminant Analysis Classifier (LinearDiscriminantAnalysis)
- Adaptive Boosting (AdaBoost) (AdaBoostClassifier)
- Bootstrap Aggregating (Bagging) (BaggingClassifier)
- Gradient Boosting Machine (GBM) (GradientBoostingClassifier)
- Multi-layer Perceptron (MLPClassifier)
- XgBoost (XGBoostRegressor)
  
with default settings from sklearn.


## Data Format :
The data needs to be in csv format and has to be partitioned as train and test before feeding it to the models. The generative models are learned using the training data. The downstream classifiers are either trained using
the real train data or synthetic data generated by the models. The classifiers are evaluated on the left out test data.

Currently only two attribute types are supported : 

1. All attributes are continuous : supported models are **pate-gan**, **dp-wgan**


In case the data has both kinds of attributes, it needs to be pre-processed (discretization for continuous values/ encoding for categorical attrbiutes) to use one of the models.
Missing values are not supported and needs to replaced appropriately by the user before usage.

**NOTE** : Some imputation methods compute statistics using other data samples to fill missing values. Care needs to be taken to make the computed statistics differentially private and the cost must be added to the generative modeling privacy cost to compute the total privacy cost.

The first line of the csv data file is assumed to contain the column names and the target column (labels) needs to be specified using the `--target-variable` flag when running the evaluation script as shown below.


## How to:

```python evaluate.py --target-variable=<> --train-data-path=<> --test-data-path=<> <model_name>  --enable-privacy --target-epsilon=5 --target-delta=1e-5```

Model names can be **real-data**, **pate-gan**, **dp-wgan**, or **ct-gan**.

### Example:
After preprocessing Cervical data using the preprocess_cervical.py, we can train a differentially private wasserstein GAN on it and evaluate the quality of the synthetic dataset using the below script :

```python evaluate.py --target-variable='Biopsy' --train-data-path=./data/cervical_processed_train.csv --test-data-path=./data/cervical_processed_test.csv --normalize-data dp-wgan  --enable-privacy --sigma=0.8 --target-epsilon=8```

### Example Output:

```
Evaluate downstream classifiers on test data:
------------------------------------------------------------
LogisticRegression             auc 0.2405	 auprc 0.0437
------------------------------------------------------------
RandomForestClassifier         auc 0.4831	 auprc 0.0633
------------------------------------------------------------
GaussianNB                     auc 0.3411	 auprc 0.0716
------------------------------------------------------------
BernoulliNB                    auc   0.5	 auprc 0.064
------------------------------------------------------------
DecisionTreeClassifier         auc 0.5816	 auprc 0.1251
------------------------------------------------------------
LinearDiscriminantAnalysis     auc 0.2044	 auprc 0.0419
------------------------------------------------------------
AdaBoostClassifier             auc 0.4591	 auprc 0.1166
------------------------------------------------------------
BaggingClassifier              auc 0.3735	 auprc 0.053
------------------------------------------------------------
GradientBoostingClassifier     auc 0.5085	 auprc 0.1614
------------------------------------------------------------
MLPClassifier                  auc 0.9113	 auprc 0.4919
------------------------------------------------------------
SVC                            auc   0.5	 auprc 0.064
------------------------------------------------------------
GradientBoostingRegressor      auc 0.5195	 auprc 0.1016
------------------------------------------------------------
Average:                       auc 0.4685	 auprc 0.1165
```

Synthetic data can be saved in the /data folder using the flag ```--save-synthetic```. For example,
```python python evaluate.py --target-variable='status'  --train-data-path=./data/processed_train.csv --test-data-path=./data/processed_test.csv --normalize-data dp-wgan --enable-privacy --sigma=0.8 --target-epsilon=8 --save-synthetic --output-data-path=./data ```

## Some useful user args:

### General args:

```--downstream-task : ``` **classification** or **regression**

```--normalize-data : ``` Apply sigmoid function to each value in the data

```--categorical : ``` If all attrbiutes of the data are categorical

```--target-variable : ``` Attribute name denoting the target

### Privacy args:

```--enable-privacy : ``` Enables private data generation. Non private mode can only be used for DP-WGAN and IMLE.

```--target-epsilon : ``` epsilon parameter of differential privacy

```--target-delta : ``` delta parameter of differential privacy

For more details refer to https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/dwork.pdf

### Noisy gradient descent args:

```--sigma : ``` Gaussian noise variance multiplier. A larger sigma will make the model train for longer epochs for the same privacy budget

```--clip-coeff : ``` The coefficient to clip the gradients to before adding noise for private SGD training

```--micro-batch-size : ``` Parameter to tradeoff speed vs efficiency. Gradients are averaged for a microbatch and then clipped before adding noise

### Model specific args:

#### PATE-GAN:

```--lap-scale : ``` Inverse laplace noise scale multiplier. A larger lap_scale will reduce the noise that is added per iteration of training

```--num-teachers : ``` Number of teacher disciminators

```--teacher-iters : ``` Teacher iterations during training per generator iteration

```--student-iters : ``` Student iterations during training per generator iteration

```--num-moments : ``` Number of higher moments to use for epsilon calculation

#### DP-WGAN:

```--clamp-lower : ``` Lower clamp parameter for the weights of the NN in wasserstein GAN

```--clamp-upper : ``` Upper clamp parameter for the weights of the NN in wasserstein GAN

#### CT-GAN:
TODOs













