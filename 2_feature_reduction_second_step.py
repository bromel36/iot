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
import json
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
def target_name(name):
    df = pd.read_csv(name, usecols=["Label"])
    target_names = sorted(list(df["Label"].unique()))
    return target_names


results = [['Attack', 'Feature', "Folder", 'T', 'CV', 'Acc', 'b_Acc', 'Prec', 'Rec',
            'F1', 'kap', 'ROC', 'tra-T', 'test-T']]

for f in ["CV", "SS", "DD"]:
    files_add = find_the_way(f".\\results\\compare\\{f}", ".csv")
    for i in tqdm(files_add):
        df = pd.read_csv(i)
        temp = df.values

        numeric_results = df.select_dtypes(include=[np.number])  # Chỉ lấy các cột số
        df = numeric_results.mean()

        df = list(df.values)
        temp = list(temp[0][:2])
        temp.append(f)
        temp = temp + df
        results.append(temp)
results = pd.DataFrame(results[1:], columns=results[0])

# %%###########################################
print(results.columns)

method = {"CV": "Cross-validation", "SS": "2 Diffirent Sessions", "DD": "2 Diffirent Dataset"}
import matplotlib.pylab as pylab

sns.set_style("whitegrid")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (35, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'x-large'}
for i in results["Attack"].unique():
    clean_i = i.replace("/", "_")
    graph_name = f"./pdfs/Comparison_f1_{clean_i}.pdf"
    plt.margins(x=0)

    df = results[results["Attack"] == i]
    for ii in results["Folder"].unique():
        sf = df[df["Folder"] == ii]
        my_xticks = sf["Feature"]  # list(iso.index)
        pylab.rcParams.update(params)
        # plt.figure(figsize=(10,10))
        # plt.plot(my_xticks,iso['Acc'], linestyle='--', marker='.', color='b',label= "Separate Train & Test acc")
        # plt.plot(my_xticks,cv['Acc'], linestyle='--', marker='.', color='r',label= "10-Fold CV acc")
        plt.plot(my_xticks, sf['F1'], linestyle='-', marker='o', label=method[ii])
    # plt.plot(my_xticks,iso[' F1-score'], linestyle='-', marker='o', color='m',label= "Diffirent Dataset Isolated")
    # plt.plot(my_xticks,cv[' F1-score'], linestyle='-', marker='o', color='b',label= "5-Fold CV")
    # plt.axhline(0.492885, color='r',label= "Primary feature list")
    # plt.axhline(0.443367, color='r',label= "Primary feature list")
    plt.title(f"Comparison of isolated and cross-validated data result for {i} attack ")
    plt.legend(numpoints=1)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=90)
    # plt.ylim([0.69, 0.71])
    plt.savefig(graph_name, bbox_inches='tight', format="pdf")  # , dpi=400)
    # plt.show()
    print(graph_name)
###############################################
method = {"CV": "Cross-validation", "SS": "2 Diffirent Sessions", "DD": "2 Diffirent Dataset"}
import matplotlib.pylab as pylab

sns.set_style("whitegrid")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (35, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'x-large'}
for i in results["Attack"].unique():
    graph_name = f"./pdfs/Comparison_kappa_{i}.pdf"
    plt.margins(x=0)
    temp = []
    df = results[results["Attack"] == i]
    for ii in results["Folder"].unique():
        sf = df[df["Folder"] == ii]
        my_xticks = sf["Feature"]  # list(iso.index)
        pylab.rcParams.update(params)
        temp.append(sf['kap'].values)
        # plt.figure(figsize=(10,10))
        # plt.plot(my_xticks,iso['Acc'], linestyle='--', marker='.', color='b',label= "Separate Train & Test acc")
        # plt.plot(my_xticks,cv['Acc'], linestyle='--', marker='.', color='r',label= "10-Fold CV acc")
        plt.plot(my_xticks, sf['kap'], linestyle='-', marker='o', label=method[ii])
    # plt.plot(my_xticks,iso[' F1-score'], linestyle='-', marker='o', color='m',label= "Diffirent Dataset Isolated")
    # plt.plot(my_xticks,cv[' F1-score'], linestyle='-', marker='o', color='b',label= "5-Fold CV")
    # plt.axhline(0.492885, color='r',label= "Primary feature list")
    # plt.axhline(0.443367, color='r',label= "Primary feature list")
    plt.title(f"Comparison of isolated and cross-validated data result for {i} attack ")
    plt.legend(numpoints=1)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel("Kappa")
    plt.xticks(rotation=90)
    # plt.ylim([0.69, 0.71])
    plt.savefig(graph_name, bbox_inches='tight', format="pdf")  # , dpi=400)
    # plt.show()
    print(graph_name)

IDF = ['DNS_id',
       'TCP_seq', 'Ether_dst', 'ICMP_chksum',
       'ICMP_id', "IP_len",
       "pck_size",
       'ICMP_seq',
       'ICMP_unused',
       'ID',
       'IP_chksum',
       'IP_dst',
       "random",
       'IP_id',
       'IP_src', 'TCP_ack', 'TCP_chksum', 'sport_bare', 'dport_bare',
       'dport23', 'sport23', 'dport',
       'sport']

GA_input = {}
method = {"CV": "Cross-validation", "SS": "2 Diffirent Sessions", "DD": "2 Diffirent Dataset"}
import matplotlib.pylab as pylab

print("============================================generate ga section====================================")

for attack in results["Attack"].unique():
    print(f"____________________________________{attack}________________________________________________________")
    # plt.margins(x=0)
    temp = []
    df = results[results["Attack"] == attack]
    for ii in results["Folder"].unique():
        sf = df[df["Folder"] == ii]
        my_xticks = sf["Feature"]  # list(iso.index)
        temp.append(sf['kap'].values)
    itself = []
    same = []
    diff = []
    my_xticks = my_xticks.values
    flag = 1
    for j in range(len(temp[0])):
        if temp[0][j] > 0:
            itself.append(my_xticks[j])
        if temp[1][j] > 0:
            same.append(my_xticks[j])
        if temp[2][j] > 0:
            diff.append(my_xticks[j])
    itself.append("Label")
    same.append("Label")
    diff.append("Label")

    for j in [itself, same]:
        print(len(j))
        print(f"{j}\n\n")

    merged = itself + same
    final_list = []
    for d in diff:
        if d in merged:
            final_list.append(d)

    c1 = Counter(final_list)
    c2 = Counter(IDF)
    final_list = list((c1 - c2).elements())
    print(len(final_list))
    print(f"{final_list}\n\n")
    main = itself + same + diff
    GA_input[attack] = final_list
    main = set(main)
    mainlist = []
    for i in main:
        temp = [i, int(i in itself), int(i in same), int(i in diff)]
        mainlist.append(temp)
    data = pd.DataFrame(mainlist,
                        columns=["Feature", "Cross-validation", "Diffirent Sessions", "Diffirent Datasets"]).set_index(
        'Feature')
    clean_i = attack.replace("/", "_")
    graph_name = f"./pdfs/kappa_{clean_i}_Voting2.PDF"
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    plt.rcParams.update(params)
    import matplotlib.pylab as pylab

    # pylab.rcParams.update(params)

    data.plot.bar(stacked=True, figsize=(28, 5))
    plt.xlabel('Features')
    plt.ylabel('Votes')
    plt.savefig(graph_name, bbox_inches='tight', format="pdf")  # , dpi=400)
    # plt.show()
    print(graph_name)
# with open('GA_input.json', 'w') as fp:
#     json.dump(GA_input, fp)
#
# đoạn này là trắng lấy GA của ngta luôn, vì của mình cái dataset vs dataset bị thiếu nên không chạy so sánh được
GA_input = {'ACK': ['ACK Flag Cnt',
                    'Bwd IAT Std',
                    'Bwd Pkt Len Max',
                    'Bwd Pkt Len Mean',
                    'Bwd Pkt Len Std',
                    'Bwd Seg Size Avg',
                    'Flow Byts/s',
                    'Flow IAT Std',
                    'Fwd Act Data Pkts',
                    'Fwd IAT Max',
                    'Fwd IAT Mean',
                    'Fwd IAT Min',
                    'Fwd IAT Std',
                    'Fwd IAT Tot',
                    'Fwd Pkt Len Max',
                    'Fwd Pkt Len Mean',
                    'Fwd Pkt Len Std',
                    'Fwd Seg Size Avg',
                    'Pkt Len Max',
                    'Pkt Len Mean',
                    'Pkt Len Std',
                    'Pkt Len Var',
                    'Pkt Size Avg',
                    'Src Port',
                    'Subflow Bwd Byts',
                    'Subflow Fwd Byts',
                    'SYN Flag Cnt',
                    'TotLen Bwd Pkts',
                    'TotLen Fwd Pkts',
                    'Label'],
            'ARP': ['ACK Flag Cnt',
                    'Active Mean',
                    'Active Min',
                    'Active Std',
                    'Bwd Header Len',
                    'Bwd IAT Max',
                    'Bwd IAT Mean',
                    'Bwd IAT Min',
                    'Bwd IAT Tot',
                    'Bwd Pkt Len Max',
                    'Bwd Pkt Len Mean',
                    'Bwd Pkt Len Min',
                    'Bwd Pkt Len Std',
                    'Bwd Pkts/s',
                    'Bwd PSH Flags',
                    'Bwd Seg Size Avg',
                    'Down/Up Ratio',
                    'Dst Port',
                    'Flow Byts/s',
                    'Flow Duration',
                    'Flow IAT Max',
                    'Flow IAT Mean',
                    'Flow IAT Min',
                    'Flow IAT Std',
                    'Flow Pkts/s',
                    'Fwd Act Data Pkts',
                    'Fwd Header Len',
                    'Fwd IAT Max',
                    'Fwd IAT Mean',
                    'Fwd IAT Min',
                    'Fwd IAT Std',
                    'Fwd IAT Tot',
                    'Fwd Pkt Len Max',
                    'Fwd Pkt Len Mean',
                    'Fwd Pkt Len Min',
                    'Fwd Pkt Len Std',
                    'Fwd Pkts/s',
                    'Fwd Seg Size Avg',
                    'Idle Max',
                    'Idle Mean',
                    'Idle Min',
                    'Idle Std',
                    'Init Bwd Win Byts',
                    'Pkt Len Max',
                    'Pkt Len Mean',
                    'Pkt Len Min',
                    'Pkt Len Std',
                    'Pkt Len Var',
                    'Pkt Size Avg',
                    'Protocol',
                    'PSH Flag Cnt',
                    'Src Port',
                    'Subflow Bwd Byts',
                    'Subflow Bwd Pkts',
                    'Subflow Fwd Byts',
                    'Subflow Fwd Pkts',
                    'Tot Bwd Pkts',
                    'Tot Fwd Pkts',
                    'TotLen Bwd Pkts',
                    'TotLen Fwd Pkts',
                    'Label'],
            'BF': ['Bwd IAT Min',
                   'Bwd Pkts/s',
                   'Flow IAT Max',
                   'Flow IAT Std',
                   'Fwd IAT Max',
                   'Fwd IAT Min',
                   'Fwd IAT Std',
                   'Fwd IAT Tot',
                   'Fwd Pkts/s',
                   'Init Bwd Win Byts',
                   'Src Port',
                   'Subflow Bwd Pkts',
                   'Subflow Fwd Byts',
                   'Tot Bwd Pkts',
                   'TotLen Fwd Pkts',
                   'Label'],
            'HTTP': ['Bwd Pkts/s',
                     'Dst Port',
                     'FIN Flag Cnt',
                     'Flow Duration',
                     'Flow IAT Max',
                     'Flow IAT Mean',
                     'Flow Pkts/s',
                     'Fwd IAT Max',
                     'Fwd IAT Mean',
                     'Fwd IAT Min',
                     'Fwd IAT Tot',
                     'Fwd Pkts/s',
                     'Subflow Fwd Pkts',
                     'SYN Flag Cnt',
                     'Tot Fwd Pkts',
                     'Label'],
            'SP': ['Active Max',
                   'Active Mean',
                   'Active Min',
                   'Active Std',
                   'Bwd Pkt Len Max',
                   'Bwd Pkt Len Mean',
                   'Bwd Pkt Len Min',
                   'Bwd Seg Size Avg',
                   'Down/Up Ratio',
                   'Flow Byts/s',
                   'Flow IAT Std',
                   'Fwd Act Data Pkts',
                   'Fwd Header Len',
                   'Fwd IAT Max',
                   'Fwd IAT Mean',
                   'Fwd IAT Min',
                   'Fwd IAT Std',
                   'Fwd IAT Tot',
                   'Fwd Pkt Len Max',
                   'Fwd Pkt Len Mean',
                   'Fwd Pkt Len Min',
                   'Fwd Pkt Len Std',
                   'Fwd Pkts/s',
                   'Fwd Seg Size Avg',
                   'Idle Max',
                   'Idle Mean',
                   'Idle Min',
                   'Idle Std',
                   'Init Bwd Win Byts',
                   'Pkt Len Max',
                   'Pkt Len Mean',
                   'Pkt Len Min',
                   'Pkt Len Std',
                   'Pkt Len Var',
                   'Pkt Size Avg',
                   'Protocol',
                   'Subflow Bwd Byts',
                   'Subflow Bwd Pkts',
                   'Subflow Fwd Byts',
                   'Subflow Fwd Pkts',
                   'SYN Flag Cnt',
                   'Tot Bwd Pkts',
                   'Tot Fwd Pkts',
                   'TotLen Bwd Pkts',
                   'TotLen Fwd Pkts',
                   'Label'],
            'SYN': ['ACK Flag Cnt',
                    'Active Max',
                    'Active Mean',
                    'Active Min',
                    'Active Std',
                    'Bwd IAT Max',
                    'Bwd IAT Mean',
                    'Bwd IAT Min',
                    'Bwd IAT Std',
                    'Bwd IAT Tot',
                    'Bwd Pkt Len Max',
                    'Bwd Pkt Len Mean',
                    'Bwd Pkt Len Min',
                    'Bwd Pkt Len Std',
                    'Bwd Pkts/s',
                    'Bwd PSH Flags',
                    'Bwd Seg Size Avg',
                    'FIN Flag Cnt',
                    'Flow Byts/s',
                    'Flow Duration',
                    'Flow IAT Max',
                    'Flow IAT Mean',
                    'Flow IAT Min',
                    'Flow IAT Std',
                    'Flow Pkts/s',
                    'Fwd Act Data Pkts',
                    'Fwd Header Len',
                    'Fwd IAT Max',
                    'Fwd IAT Mean',
                    'Fwd IAT Min',
                    'Fwd IAT Std',
                    'Fwd IAT Tot',
                    'Fwd Pkt Len Max',
                    'Fwd Pkt Len Mean',
                    'Fwd Pkt Len Min',
                    'Fwd Pkt Len Std',
                    'Fwd Pkts/s',
                    'Fwd Seg Size Avg',
                    'Idle Max',
                    'Idle Mean',
                    'Idle Min',
                    'Idle Std',
                    'Pkt Len Max',
                    'Pkt Len Mean',
                    'Pkt Len Min',
                    'Pkt Len Std',
                    'Pkt Len Var',
                    'Pkt Size Avg',
                    'Protocol',
                    'PSH Flag Cnt',
                    'Src Port',
                    'Subflow Bwd Byts',
                    'Subflow Bwd Pkts',
                    'Subflow Fwd Byts',
                    'Subflow Fwd Pkts',
                    'SYN Flag Cnt',
                    'Tot Bwd Pkts',
                    'Tot Fwd Pkts',
                    'TotLen Bwd Pkts',
                    'TotLen Fwd Pkts',
                    'Label'],
            'UDP': ['Bwd Pkt Len Max', 'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                    'Fwd IAT Std', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Pkt Len Max', 'Pkt Size Avg', 'Src Port',
                    'Subflow Bwd Byts', 'Subflow Fwd Byts', 'TotLen Bwd Pkts', 'TotLen Fwd Pkts', 'Label']}

# import json
with open('GA_input.json', 'w') as fp:
    json.dump(GA_input, fp)

print("============================================read ga section====================================")
method = {"CV": "Cross-validation", "SS": "2 Diffirent Sessions", "DD": "2 Diffirent Dataset"}
import matplotlib.pylab as pylab

for attack in results["Attack"].unique():
    print(f"____________________________________{attack}________________________________________________________")
    # plt.margins(x=0)
    temp = []
    df = results[results["Attack"] == attack]
    for ii in results["Folder"].unique():
        sf = df[df["Folder"] == ii]
        my_xticks = sf["Feature"]  # list(iso.index)
        temp.append(sf['kap'].values)
    itself = []
    same = []
    diff = []
    my_xticks = my_xticks.values
    flag = 1
    for j in range(len(temp[0])):
        try:
            if temp[0][j] > 0:
                itself.append(my_xticks[j])
            if temp[1][j] > 0:
                same.append(my_xticks[j])
            if temp[2][j] > 0:
                diff.append(my_xticks[j])
        except:
            pass
    itself.append("Label")
    same.append("Label")
    diff.append("Label")

    for j in [itself, same]:
        print(len(j))
        print(f"{j}\n\n")

    c1 = Counter(diff)
    c2 = Counter(IDF)
    diff = list((c1 - c2).elements())
    print(len(diff))
    print(f"{diff}\n\n")
    main = itself + same + diff

    main = set(main)
    if len(diff) > 1:
        GA_input[attack] = diff
    else:
        GA_input[attack] = list(main)
    mainlist = []
    for i in main:
        temp = [i, int(i in itself), int(i in same), int(i in diff)]
        mainlist.append(temp)

    data = pd.DataFrame(mainlist, columns=["Feature", "itself", "same", "diffirent"]).set_index('Feature')
    clean_attack = attack.replace("/", "_")
    graph_name = f"./pdfs/kappa_{clean_attack}_Voting2.PDF"
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    plt.rcParams.update(params)
    import matplotlib.pylab as pylab

    # pylab.rcParams.update(params)

    data.plot.bar(stacked=True, figsize=(28, 5))
    plt.xlabel('Features')
    plt.ylabel('Votes')
    plt.savefig(graph_name, bbox_inches='tight', format="pdf")  # , dpi=400)

    # plt.show()
with open('GA_input.json', 'w') as fp:
    json.dump(GA_input, fp)
