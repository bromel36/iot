# %%
# %matplotlib inline
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.stats import uniform as sp_randFloat
from sklearn import svm
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
from tabulate import tabulate
import numpy as np
import pandas as pd
import sklearn
import warnings

warnings.filterwarnings('ignore')
import os

# %%
from scipy.stats import randint as sp_randInt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy import sparse

# %%
import json

with open('../GA_output_ET.json', 'r') as fp:
    feature_list = json.load(fp)

# %%
feature_list

# %%
file_list = {'SYN': ['./csvs/dos-synflooding-1-dec.pcap_Flow.csv',
                     './INPUT/VAL/VAL-SYN.csv'],
             'HTTP': ['./csvs/mirai-httpflooding-4-dec.pcap_Flow.csv',
                      './INPUT/VAL/VAL-HTTP.csv'],
             'ACK': ['./csvs/mirai-ackflooding-4-dec.pcap_Flow.csv',
                     './INPUT/VAL/VAL-ACK.csv'],
             'UDP': ['./csvs/mirai-udpflooding-4-dec.pcap_Flow.csv',
                     './INPUT/VAL/VAL-UDP.csv'],
             'ARP': ['./csvs/mitm-arpspoofing-6-dec.pcap_Flow.csv',
                     './INPUT/VAL/VAL-ARP.csv'],
             'SP': ['./csvs/scan-hostport-3-dec.pcap_Flow.csv', './INPUT/VAL/VAL-SP.csv'],
             'BF': ['./csvs/mirai-hostbruteforce-5-dec.pcap_Flow.csv',
                    './INPUT/VAL/VAL-BF.csv']}


# %%
def run_random_search(model, params, x_train, y_train):
    # grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid = RandomizedSearchCV(model, param_grid, cv=ps, scoring='f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_, 8), grid.best_estimator_)


# %%
def find_the_way(path, file_format, con=""):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                if con in file:
                    files_add.append(os.path.join(r, file))

    return files_add


# %% [markdown]
# # RandomizedSearchCV RF

# %%
lines = [['bootst', 'criter', 'max_depth', 'max_features', "min_samp_split", "n_estimators", "F1", "Std", "Time", "No",
          "Attack"]]

for j in file_list:
    print(j)

    df = pd.read_csv(file_list[j][0], usecols=feature_list[j])
    X_train = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_train = df['Label'].cat.codes

    df = pd.read_csv(file_list[j][1], usecols=feature_list[j])
    X_test = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_test = df['Label'].cat.codes

    X = np.concatenate([X_train, X_test])
    test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
    y = np.concatenate([y_train, y_test])
    ps = PredefinedSplit(test_fold)

    print('%-35s %-20s %-8s %-8s' % ("HYPERPARAMETERS", "F1 Score", "Time", "No"))

    # use a full grid over all parameters
    param_grid = {"max_depth": np.linspace(1, 32, 32, endpoint=True).astype(int),
                  "n_estimators": sp_randint(1, 200),
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    second = time()
    f1 = []
    clf = RandomForestClassifier()
    for ii in range(1):
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
    f1 = sum(f1) / len(f1)
    # if f1>0.76:
    print('%-35s %-20s %-8s %-8s' % ("default", f1, round(time() - second, 3), ii))

    ######################################################################################################################
    for i in tqdm(range(10)):
        second = time()
        a, b, clf = run_random_search(RandomForestClassifier(), param_grid, X, y)
        f1 = []
        for ii in range(5):
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
        f1_result = sum(f1) / len(f1)
        f1 = np.array(f1)
        stndtd = f1.std()
        temp = list(a.values())
        # print('%-90s %-20s %-8s %-8s' % (a,f1_result,round(time()-second,3),i))
        temp = temp + [f1_result, stndtd, round(time() - second, 3), i, j]
        lines.append(temp)

        # if f1>0.76:
results = pd.DataFrame(lines[1:], columns=lines[0])
results.to_csv("RF_HPO.csv", index=False)

final_parametres = [
    ['bootst', 'criter', 'max_depth', 'max_features', "min_samp_split", "n_estimators", "F1", "Std", "Time", "No",
     "Attack"]]

for i in results["Attack"].unique():
    df = results[results["Attack"] == i]
    m = df["F1"].max()
    df = df[df["F1"] == m]
    m = df["max_depth"].min()
    df = df[df["max_depth"] == m]
    final_parametres.append(list(df.values)[0])
results = pd.DataFrame(final_parametres[1:], columns=final_parametres[0])
print(tabulate(results, headers=list(results.columns)))

# %%
# NB
from sklearn.naive_bayes import GaussianNB

# %%
lines = [['var_smoothing', "F1", "Std", "Time", "No", "Attack"]]
for j in file_list:
    print(j)

    df = pd.read_csv(file_list[j][0], usecols=feature_list[j])
    X_train = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_train = df['Label'].cat.codes

    df = pd.read_csv(file_list[j][1], usecols=feature_list[j])
    X_test = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_test = df['Label'].cat.codes

    X = np.concatenate([X_train, X_test])
    test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
    y = np.concatenate([y_train, y_test])
    ps = PredefinedSplit(test_fold)

    second = time()

    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=100),
    }

    second = time()
    f1 = []
    clf = GaussianNB()
    for ii in range(1):
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
    f1 = sum(f1) / len(f1)
    # if f1>0.76:
    print('%-35s %-20s %-8s %-8s' % ("default", f1, round(time() - second, 3), ii))
    ######################################################################################################################
    for i in tqdm(range(10)):
        second = time()
        a, b, clf = run_random_search(GaussianNB(), param_grid, X, y)
        f1 = []
        for ii in range(5):
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
        f1_result = sum(f1) / len(f1)
        f1 = np.array(f1)
        stndtd = f1.std()
        temp = list(a.values())
        # print('%-90s %-20s %-8s %-8s' % (a,f1_result,round(time()-second,3),i))
        temp = temp + [f1_result, stndtd, round(time() - second, 3), i, j]
        lines.append(temp)

results = pd.DataFrame(lines[1:], columns=lines[0])
results.to_csv("NB_HPO.csv", index=False)
print(tabulate(results, headers=list(results.columns)))

# %%

final_parametres = [["var_smoothing", "F1", "Std", "Time", "No", "Attack"]]
for i in results["Attack"].unique():
    df = results[results["Attack"] == i]
    m = df["F1"].max()
    df = df[df["F1"] == m]
    final_parametres.append(list(df.values)[0])

results = pd.DataFrame(final_parametres[1:], columns=final_parametres[0])
print(tabulate(results, headers=list(results.columns)))

# # RandomizedSearchCV  SVM

# %%
lines = [['gamma', 'C', "F1", "Std", "Time", "No", "Attack"]]

for j in file_list:
    print(j)

    df = pd.read_csv(file_list[j][0], usecols=feature_list[j])
    X_train = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_train = df['Label'].cat.codes

    df = pd.read_csv(file_list[j][1], usecols=feature_list[j])
    X_test = df.iloc[:, 0:-1]
    df['Label'] = df['Label'].astype('category')
    y_test = df['Label'].cat.codes

    X = np.concatenate([X_train, X_test])
    test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
    y = np.concatenate([y_train, y_test])
    ps = PredefinedSplit(test_fold)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}

    second = time()
    f1 = []
    clf = svm.SVC()
    for ii in range(1):
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
    f1 = sum(f1) / len(f1)
    # if f1>0.76:
    print('%-35s %-20s %-8s %-8s' % ("default", f1, round(time() - second, 3), ii))

    ######################################################################################################################
    for i in tqdm(range(1)):
        second = time()
        a, b, clf = run_random_search(svm.SVC(), param_grid, X, y)
        f1 = []
        for ii in range(1):
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            f1.append(sklearn.metrics.f1_score(y_test, predict, average="macro"))
        f1_result = sum(f1) / len(f1)
        f1 = np.array(f1)
        stndtd = f1.std()
        temp = list(a.values())
        print('%-90s %-20s %-8s %-8s' % (a, f1_result, round(time() - second, 3), i))
        temp = temp + [f1_result, stndtd, round(time() - second, 3), i, j]
        lines.append(temp)

        # if f1>0.76:

results = pd.DataFrame(lines[1:], columns=lines[0])
results.to_csv("svm_HPO.csv", index=False)

print(tabulate(results, headers=list(results.columns)))
