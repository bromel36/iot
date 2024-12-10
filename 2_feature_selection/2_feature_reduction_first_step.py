import warnings

warnings.filterwarnings("ignore")

# %%
import os
import time
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import random
from tabulate import tabulate
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.tree import ExtraTreeClassifier

evaluate = {'Acc': "Accuracy", 'b_Acc': "Balanced Accuracy", 'F1': "F1 Score", 'kap': "Kappa", 'ROC': "Roc"}


# %%
def folder(f_name):  # this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print("The folder could not be created!")


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


# files_add=find_the_way("./INPUT/SM",".csv")
# files_add

# %%
ml_list = {"ET": ExtraTreeClassifier()}  # ,"SVC":SVC()}}


# %%
def target_name(name):
    df = pd.read_csv(name, usecols=["Label"])
    target_names = sorted(list(df["Label"].unique()))
    return target_names


# %%
folder("./results/compare/SS/")
folder("./results/compare/CV/")
folder("./results/compare/DD/")
folder("./pdfs")


def score(train_time, test_time, predict, y_test, class_based_results, repeat, cv, dname, ml, sw):
    train_time = train_time[0]
    test_time = test_time[0]
    rc = sklearn.metrics.recall_score(y_test, predict, average="macro")
    pr = sklearn.metrics.precision_score(y_test, predict, average="macro")
    f_1 = sklearn.metrics.f1_score(y_test, predict, average="macro")
    accuracy = sklearn.metrics.accuracy_score(y_test, predict)
    accuracy_b = sklearn.metrics.balanced_accuracy_score(y_test, predict)
    kappa = sklearn.metrics.cohen_kappa_score(y_test, predict, labels=None, weights=None, sample_weight=None)
    try:
        roc = sklearn.metrics.roc_auc_score(y_test, predict)
    except:
        roc = 0
    report = sklearn.metrics.classification_report(y_test, predict, target_names=target_names, output_dict=True)
    cr = pd.DataFrame(report).transpose()
    line = [sw, dname, repeat, cv, ml, accuracy, accuracy_b, pr, rc, f_1, kappa, roc, train_time, test_time]

    if class_based_results.empty:
        class_based_results = cr
    else:
        class_based_results = class_based_results.add(cr, fill_value=0)
    return class_based_results, line


# %%
def ML(loop1, loop2, output_csv, cols, dname, sw):
    fold = 5
    repetition = 25
    for ii in ml_list:
        class_based_results = pd.DataFrame()  # "" #pd.DataFrame(0, index=np.arange((len(target_names)+3)), columns=["f1-score","precision","recall","support"])
        cm = pd.DataFrame()
        cv = 0
        lines = [["Attack", "Feature", "T", "CV", "ML", "Acc", "b_Acc", "Prec", "Rec", "F1", "kap", "ROC", "tra-T",
                  "test-T"]]
        for i in range(repetition):
            df = pd.read_csv(loop1, usecols=cols)  # ,header=None )
            df = df.fillna(0)
            X_train = df[df.columns[0:-1]]
            X_train = np.array(X_train)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y_train = df[df.columns[-1]].cat.codes

            df = pd.read_csv(loop2, usecols=cols)  # ,header=None )
            df = df.fillna(0)
            X_test = df[df.columns[0:-1]]
            X_test = np.array(X_test)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y_test = df[df.columns[-1]].cat.codes

            # dname=loop1  [6:-13]
            results_y = []

            results_y.append(y_test)

            precision = []
            recall = []
            f1 = []
            accuracy = []
            train_time = []
            test_time = []
            total_time = []
            kappa = []
            accuracy_b = []

            # machine learning algorithm is applied in this section
            clf = ml_list[ii]  # choose algorithm from ml_list dictionary
            second = time.time()
            clf.fit(X_train, y_train)
            train_time.append(float((time.time() - second)))
            second = time.time()
            predict = clf.predict(X_test)
            test_time.append(float((time.time() - second)))

            altime = 0
            class_based_results, line = score(train_time, test_time, predict, y_test, class_based_results, cv, i, dname,
                                              ii, sw)
            lines.append(line)

        results = pd.DataFrame(lines[1:], columns=lines[0])
        output_csv = output_csv.replace("./", "")
        output_csv = output_csv.replace("/", "-")
        results.to_csv(output_csv.replace("ML", ii), index=False)
        results = results.mean()
        results = results.round(3)
        # print (tabulate(results, headers=list(results.columns)))
        # print()
        return list(results.values)


# %%
def ML_CV(loop1, loop2, output_csv, cols, dname, sw):
    fold = 3
    repetition = 5
    for ii in ml_list:
        class_based_results = pd.DataFrame()  # "" #pd.DataFrame(0, index=np.arange((len(target_names)+3)), columns=["f1-score","precision","recall","support"])
        cm = pd.DataFrame()
        cv = 0
        lines = [["Attack", "Feature", "T", "CV", "ML", "Acc", "b_Acc", "Prec", "Rec", "F1", "kap", "ROC", "tra-T",
                  "test-T"]]
        for i in range(repetition):

            rnd = random()

            kfold = sklearn.model_selection.KFold(n_splits=fold, shuffle=True, random_state=int(rnd * 100))
            cv = 0
            df = pd.read_csv(loop1, usecols=cols)  # ,header=None )
            ##df = df.reset_index(drop=True)
            # missing_cols = [col for col in cols if col not in df.columns]
            # if missing_cols:
            #     raise ValueError(f"Các cột sau không tồn tại trong dữ liệu: {missing_cols}")
            df = df.fillna(0)
            df = df.sample(frac=1).reset_index(drop=True)
            # del df["MAC"] # if dataset has MAC colomn please uncomment this line
            X = df[df.columns[0:-1]]
            X = np.array(X)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y = df[df.columns[-1]].cat.codes
            X.shape
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # dname=loop1  [6:-13]
                results_y = []
                cv += 1
                results_y.append(y_test)

                precision = []
                recall = []
                f1 = []
                accuracy = []
                train_time = []
                test_time = []
                total_time = []
                kappa = []
                accuracy_b = []

                # machine learning algorithm is applied in this section
                clf = ml_list[ii]  # choose algorithm from ml_list dictionary
                second = time.time()
                clf.fit(X_train, y_train)
                train_time.append(float((time.time() - second)))
                second = time.time()
                predict = clf.predict(X_test)
                test_time.append(float((time.time() - second)))

                altime = 0
                class_based_results, line = score(train_time, test_time, predict, y_test, class_based_results, cv, i,
                                                  dname, ii, sw)
                lines.append(line)

        results = pd.DataFrame(lines[1:], columns=lines[0])
        output_csv = output_csv.replace("./", "")
        output_csv = output_csv.replace("/", "-")
        results.to_csv(output_csv.replace("ML", ii), index=False)

        numeric_results = results.select_dtypes(include=[np.number])  # Chỉ lấy các cột số
        results = numeric_results.mean()

        # results = results.mean()
        results = results.round(3)
        results
        # print (tabulate(results, headers=list(results.columns)))
        # print()
        return list(results.values)


# %%
features = ['Src Port', 'Dst Port', 'Protocol',
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
            'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
            'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
            'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
            'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
            'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
            'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
            'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
            'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
            'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
            'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
            'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
            'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
            'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
            'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
            'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
            'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

# %%
len(features)

# %%


file_list = {'./csv_flow/dos-synflooding-1-dec.pcap_Flow.csv': "SYN",
             './csv_flow/mirai-httpflooding-4-dec.pcap_Flow.csv': "HTTP",
             './csv_flow/mirai-ackflooding-4-dec.pcap_Flow.csv': "ACK",
             './csv_flow/mirai-udpflooding-4-dec.pcap_Flow.csv': "UDP",
             './csv_flow/mitm-arpspoofing-6-dec.pcap_Flow.csv': "ARP",
             './csv_flow/scan-hostport-3-dec.pcap_Flow.csv': "SP",
             './csv_flow/mirai-hostbruteforce-5-dec.pcap_Flow.csv': "BF"}

for train in (file_list):
    print(train)
    df = pd.read_csv(train, usecols=["Label"])
    print(df.groupby("Label").size())

# %%


# %%
for train in (file_list):
    lines = [["Attack", "Feature", "ML", "Acc", "b_Acc", "Prec", "Rec", "F1", "kap", "ROC", "tra-T", "test-T"]]
    for dname in tqdm(features):
        try:
            target_names = ["attack", "benign"]
            feature = [dname, 'Label']
            output_csv = f"./results/compare/CV/ET_{file_list[train]}_{dname}.csv"
            # print(f"{list[train]} Dataset - Feature {number+1}/{len(features)}")
            print(output_csv)
            temp = ML_CV(train, "", output_csv, feature, dname, file_list[train])
            print(output_csv)
            temp = temp[2:]
            temp = [file_list[train], dname, "ET"] + temp
            lines.append(temp)

        except Exception as e:
            print("#" * 110)
            print(f"ERROR ABOUT {list[train]} Dataset - Feature {dname}")
            print("#" * 110 + "\n\n")
            # print("#"*110)
            # print(f"LỖI VỚI DỮ LIỆU: {loop1} - TÍNH NĂNG: {dname}")
            print(f"Chi tiết lỗi: {str(e)}")
            # print("#"*110+"\n")

    results = pd.DataFrame(lines[1:], columns=lines[0])
    print(tabulate(results, headers=list(results.columns)))
