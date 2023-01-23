import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
from sklearn import datasets, metrics, model_selection, svm
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve,roc_auc_score, roc_curve, accuracy_score
from statsmodels.graphics.mosaicplot import mosaic


def plot_confusion_matrix(conf_m, classifier):
	fig, ax = plt.subplots(figsize=(8, 8))
	fig.suptitle('Confusion matrix for ' + classifier)
	ax.imshow(conf_m)
	ax.grid(False)
	ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
	ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
	ax.set_ylim(1.5, -0.5)
	for i in range(2):
	    for j in range(2):
	        ax.text(j, i, conf_m[i, j], ha='center', va='center', color='red')


def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# ------------------------------------------ get data from apify and prepare them ------------------------------------------
x = pd.read_csv('FinalDataset.csv')
column_usernames = ['username']
usernames = x.loc[:,column_usernames]
x = x.drop(columns=['username'])

# ------------------------------------------ plot section ------------------------------------------
sns.heatmap(x.isnull()).set(title='None values')
plt.show()

sns.scatterplot(data=x, x="follower_num", y="following_num", hue="label").set(title='Followers and Followings of accounts')
plt.show()

fig, axs = plt.subplots(ncols=3)
fig.suptitle('Charasteristics of usernames')
sns.boxplot(y='username_len', x='label', data=x, ax=axs[0])
sns.boxplot(y='fullname_len', x='label', data=x, ax=axs[1])
sns.boxplot(y='Digits_in_username', x='label', data=x, ax=axs[2])
plt.show()

fig2, axs2 = plt.subplots(ncols=3)
mosaic(x, ['is_private', 'label'], title='Private vs fake accounts', ax=axs2[0], labelizer=None, statistic=False, axes_label=True, label_rotation=0.0)
mosaic(x, ['is_verified', 'label'], title='Verified vs fake accounts', ax=axs2[1], labelizer=None, statistic=False, axes_label=True, label_rotation=0.0)
mosaic(x, ['is_business_account', 'label'], title='Business vs fake accounts', ax=axs2[2], labelizer=None, statistic=False, axes_label=True, label_rotation=0.0)
plt.show()

# ------------------------------------------ training ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x.drop('label',axis=1), 
                                    x['label'], test_size=0.25, 
                                    random_state=42)

logmodel = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
logmodel.fit(X_train, y_train)
filename = "logistic_regression_model.joblib"
joblib.dump(logmodel, filename)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
filename = "random_forest_model.joblib"
joblib.dump(logmodel, filename)

# ------------------------------------------ evaluation logistic regression ------------------------------------------
p_pred = logmodel.predict_proba(X_test)
y_pred = logmodel.predict(X_test)
score = logmodel.score(X_test, y_test)
conf_m = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('Score of the logistic regression: ', score)
print(report)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False, title='Confusion Matrix for Logistic Regression')
plt.show()

metrics.plot_roc_curve(logmodel, X_test, y_test)
plt.show()

# ------------------------------------------ evaluation random forest ------------------------------------------
y_pred_forest = rf.predict(X_test)
score_forest = rf.score(X_test, y_test)
conf_m_forest = confusion_matrix(y_test, y_pred_forest)
forest_accuracy = accuracy_score(y_test, y_pred_forest)
report_forest = classification_report(y_test, y_pred_forest)

print('Score of the random forest', score_forest)
print(report_forest)

# graph of the confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred_forest, normalize=False, title='Confusion Matrix for Random Forest')
plt.show()

# plot roc curve and auc
metrics.plot_roc_curve(rf, X_test, y_test)
plt.show()

# print predictions plot
scatterplot_dataset = X_test
scatterplot_dataset['predictions'] = y_pred_forest
sns.scatterplot(data=scatterplot_dataset, x="follower_num", y="following_num", hue="predictions").set(title='Predictions')
plt.show()

