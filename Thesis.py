# Imports some of the modules that are required to run the rest of the scripts
import pandas as pd 
import numpy as np 
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
 
 
# Load in data for determining precision and recall for models on OS fingerprinting traffic 

df = pd.read_csv('/home/john/Thesis/Bot-IoT/OS_Fingerprinting.csv') 

# Or... 
# Load in data for determining precision and recall for models on Service Scan traffic 

df = pd.read_csv('/home/john/Thesis/Bot-IoT/Service_Scan.csv') 

# Or... 

# Load in data for determining precision and recall for models on OS fingerprinting traffic 

df = pd.read_csv('/home/john/Thesis/Bot-IoT/Keylogging.csv') 

# Or... 
# Load in data for determining precision and recall for models on Service Scan traffic 
df = pd.read_csv('/home/john/Thesis/Bot-IoT/Data_Exfil.csv') 
 
# The service scan data has a persistent column resulting from a joining function when 
# building the data set 
del df['Unnamed: 0'] 
 
# The service scan file also has some totally random values that don't convert to float...Needs to be cleaned 
df = df[df['state'] != "PAR"] 
 
 
# Drop empty columns and target column from keylogging netflows data 
df_without_blanks = df.drop(['smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'attack'], axis=1) 
 
 
# Update keylogging to only keep the features to be used for traffic classification 
training_features = df_without_blanks.drop(['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 
                                            'sport', 'daddr', 'dport', 'state', 'ltime'], axis=1) 
 
# Import packages for creating test and train data 
from sklearn.model_selection import train_test_split 
 
y = df.pop('attack') # define the target variable (dependent variable) as y 
 
# Create variables for training and testing data 
train, test, train_labels, test_labels = train_test_split(training_features, y, test_size=0.2) 

 

 

# Translates the categories from the 'state' column into numerical values for further processing for the model 
# Then translates the previously encoded numerical values into binary values so that model can process values from the 'state' column 
# Then normalizes all numerical float values to a scale between 0 and 1 to optimize predictive performance of the model 
train_num = train.drop('state', axis=1) 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Normalizer 
num_pipeline = Pipeline([ 
    ('norm_scaler', Normalizer()), 
]) 
from sklearn.compose import ColumnTransformer 
num_attribs = list(train_num) 
cat_attribs = ['state'] 
full_pipeline = ColumnTransformer([ 
    ('num', num_pipeline, num_attribs), 
    ('cat', OneHotEncoder(), cat_attribs), 
]) 
train_prepared = full_pipeline.fit_transform(train) 
test_num = test.drop('state', axis=1) 
num_attribs = list(test_num) 
test_prepared = full_pipeline.fit_transform(test) 

 
# The correlation value of each feature to the target variable, then the features are compared against 
# another to determine whether any features within the feature set are redundant. Highly correlated features 
# tend to degrade the performance of an ML algorithm 
corr_matrix = train_num.corr() 
corr_matrix['attack'].sort_values(ascending=False) 
from pandas.plotting import scatter_matrix 
attributes = ['attack', 'rate', 'drate', 'stddev', 'srate', 'min', 'max', 'mean', 'dbytes', 'dpkts', 'dur', 'bytes', 'sbytes', 'pkts', 'spkts', 'sum'] 
axes = scatter_matrix( 
    train_num[attributes], 
    figsize  = [15, 15], 
    marker   = ".", 
    s        = 5, 
    ) 
# Create the Random Forest model with 100 trees 
model = RandomForestClassifier(n_estimators=100, criterion='gini', 
                               max_depth=None, min_samples_split=2, 
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                               max_features='auto', max_leaf_nodes=None, 
                               min_impurity_decrease=0.0, min_impurity_split=None, 
                               bootstrap=True, oob_score=False, n_jobs=-1, 
                               random_state=None, verbose=0, warm_start=False, 
                               class_weight=None, ccp_alpha=0.0, max_samples=None) 
 

# Or... 
# Create the k Nearest Neighbor model 
model = KNeighborsClassifier(n_neighbors=3, weights='uniform', 
                             algorithm='auto',leaf_size=30,p=2, 
                             metric_params=None,n_jobs=None) 
 

# Or... 
# Create the Support Vector Classifier model 
model = SVC(C=1, kernel='poly', degree=3) 
 
# Fits the model to the training data and times it 
import time 
 
start = time.process_time() 
model.fit(train, train_labels) 
print(time.process_time() - start) 
 
 
# Training predictions (to demonstrate overfitting) 
train_rf_predictions = model.predict(train) 
train_rf_probs = model.predict_proba(train)[:,1] 
 
# Testing predictions (to determine performance) 
rf_predictions = model.predict(test) 
rf_probs = model.predict_proba(test)[:,1] 
 
# Import packages for generating reports 
from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve 
 
# Plot formatting 
plt.style.use('fivethirtyeight') 
plt.rcParams['font.size'] = 18 
 
# Defines the function for model evaluation 
def evaluate_model(predictions, probs, train_predictions, train_probs): 
    """Compare machine learning model to baseline performance. 
    Computes statistics and shows ROC curve.""" 
 
    baseline = {} 
 
    baseline['recall'] = recall_score(test_labels, 
                                      [1 for _ in range(len(test_labels))]) 
    baseline['precision'] = precision_score(test_labels, 
                                            [1 for _ in range(len(test_labels))]) 
    baseline['roc'] = 0.5 
 
    results = {} 
 
    results['recall'] = recall_score(test_labels, predictions) 
    results['precision'] = precision_score(test_labels, predictions) 
    results['roc'] = roc_auc_score(test_labels, probs) 
 
    train_results = {} 
    train_results['recall'] = recall_score(train_labels, train_predictions) 
    train_results['precision'] = precision_score(train_labels, train_predictions) 
    train_results['roc'] = roc_auc_score(train_labels, train_probs) 
 
    for metric in ['recall', 'precision', 'roc']: 
        print( 
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}') 
 
    # Calculate false positive rates and true positive rates 
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))]) 
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs) 
 
    plt.figure(figsize=(8, 6)) 
    plt.rcParams['font.size'] = 16 
 
    # Plot both curves 
    plt.plot(base_fpr, base_tpr, 'b', label='baseline') 
    plt.plot(model_fpr, model_tpr, 'r', label='model') 
    plt.legend(); 
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves'); 
    plt.show(); 
 
# Calls the function to evaluate the model 
evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs) 
plt.savefig('roc_auc_curve.png') 
 
from sklearn.metrics import confusion_matrix 
import itertools 
 
# Defines function that plots the confusion matrix 
def plot_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion matrix', 
                          cmap=plt.cm.Oranges): 
    """ 
    This function prints and plots the confusion matrix. 
    Normalization can be applied by setting `normalize=True`. 
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
    """ 
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix") 
    else: 
        print('Confusion matrix, without normalization') 
 
    print(cm) 
 
    # Plot the confusion matrix 
    plt.figure(figsize=(10, 10)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    plt.title(title, size=24) 
    plt.colorbar(aspect=4) 
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45, size=14) 
    plt.yticks(tick_marks, classes, size=14) 
 
    fmt = '.2f' if normalize else 'd' 
    thresh = cm.max() / 2. 
 
    # Labeling the plot 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20, 
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black") 
 
    plt.grid(None) 
    plt.tight_layout() 
    plt.ylabel('True label', size=18) 
    plt.xlabel('Predicted label', size=18) 
 
 
# Calls the function to plot the confusion matrix 
cm = confusion_matrix(test_labels, rf_predictions) 
plot_confusion_matrix(cm, classes=['Normal', 'Suspect'], 
                      title='Traffic Classification Matrix') 
 
# Saves the confusion matrix to a file 
plt.savefig('cm.png') 
 
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report 
 
# Create variables for determining importance of features 
Y_Train = train_labels 
# X_Train = StandardScaler().fit_transform(train) 
X_Train = train 
 
Y_Test = test_labels 
# X_Test = StandardScaler().fit_transform(test) 
X_Test = test 
 
 
# Determine feature importance in a Random Forest Classifier 
trainedforest = RandomForestClassifier(n_estimators=100).fit(X_Train,Y_Train) 
predictionforest = trainedforest.predict(X_Test) 
print(classification_report(Y_Test,predictionmodel)) 
 
feat_importances = pd.Series(trainedforest.feature_importances_, index= train.columns) 
plt.show(feat_importances.nlargest(16).plot(kind='barh')) 
 
 

 
