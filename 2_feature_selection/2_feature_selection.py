# feature selection

# %%
import warnings

warnings.filterwarnings("ignore")

# %%
import os
import time
import sklearn
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.tree import ExtraTreeClassifier

evaluate = {'Acc': "Accuracy", 'b_Acc': "Balanced Accuracy", 'F1': "F1 Score", 'kap': "Kappa", 'ROC': "Roc"}

# %%
import time

# sleep for 3 seconds
print('Sleep time: ', str(3600), 'seconds')
# time.sleep(3600)
print('Woke up after: ', str(3), 'seconds')


# %%
def GA(train, test, cols, gen_number=25, outputcsv="GA_output.csv"):
    # defining various steps required for the genetic algorithm
    # GA adapted from https://datascienceplus.com/genetic-algorithm-in-machine-learning-using-python/
    def initilization_of_population(size, n_feat):
        population = []
        for i in range(size):
            chromosome = np.ones(n_feat, dtype=np.bool_)
            chromosome[:int(0.3 * n_feat)] = False
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def fitness_score(population):
        scores = []
        for chromosome in population:
            logmodel.fit(X_train.iloc[:, chromosome], y_train)
            predictions = logmodel.predict(X_test.iloc[:, chromosome])
            scores.append(sklearn.metrics.f1_score(y_test, predictions, average="macro"))
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        return list(scores[inds][::-1]), list(population[inds, :][::-1])

    def selection(pop_after_fit, n_parents):
        population_nextgen = []
        for i in range(n_parents):
            population_nextgen.append(pop_after_fit[i])
        return population_nextgen

    def crossover(pop_after_sel):
        population_nextgen = pop_after_sel
        for i in range(len(pop_after_sel)):
            child = pop_after_sel[i]
            child[3:7] = pop_after_sel[(i + 1) % len(pop_after_sel)][3:7]
            population_nextgen.append(child)
        return population_nextgen

    def mutation(pop_after_cross, mutation_rate):
        population_nextgen = []
        for i in range(0, len(pop_after_cross)):
            chromosome = pop_after_cross[i]
            for j in range(len(chromosome)):
                if random.random() < mutation_rate:
                    chromosome[j] = not chromosome[j]
            population_nextgen.append(chromosome)
        # print(population_nextgen)
        return population_nextgen

    def generations(size, n_feat, n_parents, mutation_rate, n_gen, X_train,
                    X_test, y_train, y_test):

        best_chromo = []
        best_score = []
        population_nextgen = initilization_of_population(size, n_feat)
        for i in tqdm(range(n_gen)):
            second = time.time()
            scores, pop_after_fit = fitness_score(population_nextgen)
            # print(scores[:2])
            zaman = time.time() - second

            ths.write(f"{np.mean(scores)},{np.mean(scores)},{zaman}\n")

            pop_after_sel = selection(pop_after_fit, n_parents)
            pop_after_cross = crossover(pop_after_sel)
            population_nextgen = mutation(pop_after_cross, mutation_rate)
            best_chromo.append(pop_after_fit[0])
            best_score.append(scores[0])
        return best_chromo, best_score

    df = pd.read_csv(train, usecols=cols)  # ,header=None )
    df = df.fillna(0)
    # df = df.sample(n = 10000)
    X_train = df[df.columns[0:-1]]
    # X_train=np.array(X_train)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y_train = df[df.columns[-1]].cat.codes
    df = pd.read_csv(test, usecols=cols)  # ,header=None )
    df = df.fillna(0)
    # df = df.sample(n = 10000)
    X_test = df[df.columns[0:-1]]
    # X_test=np.array(X_test)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y_test = df[df.columns[-1]].cat.codes

    ths = open(f"{outputcsv}", "w")
    ths.write("MEAN,STD,TIME\n")
    logmodel = ExtraTreeClassifier()
    # print ('%-30s %-30s %-30s' % ("MEAN","STD","TIME"))
    chromo, score = generations(size=200, n_feat=X_train.shape[1], n_parents=120, mutation_rate=0.005,
                                n_gen=gen_number, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    # logmodel.fit(X_train.iloc[:,chromo[-1]],y_train)
    # predictions = logmodel.predict(X_test.iloc[:,chromo[-1]])
    # print("F1 Score score after genetic algorithm is= "+str(sklearn.metrics.f1_score(y_test,predictions,average= "macro")))
    ths.close()
    sonuç = []
    for k, j in enumerate(chromo):
        temp = X_train.iloc[:, j]
        temp = list(temp.columns)
        temp.append("Label")
        sonuç.append(temp)

    np.save(outputcsv.replace("csv", "npy"), np.array(sonuç, dtype=object))
    gf = pd.read_csv(outputcsv)
    gf = gf["MEAN"].values
    gf = np.argmax(gf)
    return sonuç[gf], gf


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


# %%
ml_list = {"ET": ExtraTreeClassifier()}  # ,"SVC":SVC()}}


# %%
def target_name(name):
    df = pd.read_csv(name, usecols=["Label"])
    target_names = sorted(list(df["Label"].unique()))
    return target_names


# %%
folder("/content/drive/MyDrive/iot/results")
folder("/content/drive/MyDrive/iot/results/beforeGA/")
folder("/content/drive/MyDrive/iot/results/afterGA/")
folder("/content/drive/MyDrive/iot/pdfs")


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
    line = [dname, sw, repeat, cv, ml, accuracy, accuracy_b, pr, rc, f_1, kappa, roc, train_time, test_time]

    if class_based_results.empty:
        class_based_results = cr
    else:
        class_based_results = class_based_results.add(cr, fill_value=0)
    return class_based_results, line


# %%
def ML_CV(loop1, loop2, output_csv, cols, dname, sw):
    fold = 5
    repetition = 10

    for ii in ml_list:
        class_based_results = pd.DataFrame()  # "" #pd.DataFrame(0, index=np.arange((len(target_names)+3)), columns=["f1-score","precision","recall","support"])
        cm = pd.DataFrame()
        cv = 0
        lines = [
            ["Dataset", "SW", "T", "CV", "ML", "Acc", "b_Acc", "Prec", "Rec", "F1", "kap", "ROC", "tra-T", "test-T"]]
        for i in range(repetition):

            # rnd = random()

            # kfold = sklearn.model_selection.KFold(n_splits=fold, shuffle=True, random_state=int(rnd*100))
            cv = 0
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
            class_based_results, line = score(train_time, test_time, predict, y_test, class_based_results, cv, i, dname,
                                              ii, sw)
            lines.append(line)
            df_cm = pd.DataFrame(sklearn.metrics.confusion_matrix(y_test, predict))
            if cm.empty:
                cm = df_cm
            else:
                cm = cm.add(df_cm, fill_value=0)

        results = pd.DataFrame(lines[1:], columns=lines[0])
        results.to_csv(output_csv.replace("ML", ii), index=False)
        results = results.round(3)
        print(tabulate(results, headers=list(results.columns)))
        print()

        class_based_results = class_based_results / repetition
        print(tabulate(class_based_results, headers=list(class_based_results.columns)))
        class_based_results.to_csv(output_csv.replace(".csv", "class_based_results.csv"))
        if True:
            cm = cm // repetition
            graph_name = output_csv[:-4] + "_confusion matrix.pdf"
            plt.figure(figsize=(5, 3.5))
            sns.heatmap(cm, xticklabels=target_names, yticklabels=target_names, annot=True, fmt='g')
            plt.savefig(graph_name, bbox_inches='tight')  # , dpi=400)
            plt.show()
            # print(cm)
            print("\n\n\n")

        # %%


# %%
# find_the_way("./csvs","csv",con="")

# %%
file_list = {'SYN': ['/content/drive/MyDrive/iot/csv_flow/dos-synflooding-1-dec.pcap_Flow.csv',
                     '/content/drive/MyDrive/iot/csv_flow/dos-synflooding-6-dec.pcap_Flow.csv'],

             'HTTP': ['/content/drive/MyDrive/iot/csv_flow/mirai-httpflooding-4-dec.pcap_Flow.csv',
                      '/content/drive/MyDrive/iot/csv_flow/mirai-httpflooding-1-dec.pcap_Flow.csv'],

             'ACK': ['/content/drive/MyDrive/iot/csv_flow/mirai-ackflooding-4-dec.pcap_Flow.csv',
                     '/content/drive/MyDrive/iot/csv_flow/mirai-ackflooding-1-dec.pcap_Flow.csv'],

             'UDP': ['/content/drive/MyDrive/iot/csv_flow/mirai-udpflooding-4-dec.pcap_Flow.csv',
                     '/content/drive/MyDrive/iot/csv_flow/mirai-udpflooding-1-dec.pcap_Flow.csv'],

             'ARP': ['/content/drive/MyDrive/iot/csv_flow/mitm-arpspoofing-6-dec.pcap_Flow.csv',
                     '/content/drive/MyDrive/iot/csv_flow/mitm-arpspoofing-1-dec.pcap_Flow.csv'],

             'SP': ['/content/drive/MyDrive/iot/csv_flow/scan-hostport-3-dec.pcap_Flow.csv',
                    '/content/drive/MyDrive/iot/csv_flow/scan-hostport-1-dec.pcap_Flow.csv'],

             'BF': ['/content/drive/MyDrive/iot/csv_flow/mirai-hostbruteforce-5-dec.pcap_Flow.csv',
                    '/content/drive/MyDrive/iot/csv_flow/mirai-hostbruteforce-1-dec.pcap_Flow.csv']}

# %%
import json

with open('/content/drive/MyDrive/iot/GA_input.json', 'r') as fp:
    feature_list = json.load(fp)

# %%
for file in file_list:
    print(file)
    features = feature_list[file]
    train = file_list[file][0]
    test = file_list[file][1]
    # feature,_=GA(train,test,features,gen_number=25,outputcsv=f"{file}_ET_chosed_GA_output.csv")
    feature = features
    print(feature)
    # GA_output[file]=feature
    output_csv = f"/content/drive/MyDrive/iot/results/beforeGA/{file}_chosed_output_ML_.csv"
    target_names = ["Benign", file]
    ML_CV(train, test, output_csv, feature, file, 5)

# %%
GA_output = {}

# %%
for file in file_list:
    print(file)
    features = feature_list[file]
    train = file_list[file][0]
    test = file_list[file][1]
    feature, _ = GA(train, test, features, gen_number=25,
                    outputcsv=f"/content/drive/MyDrive/iot/results/afterGA/{file}_ET_chosed_GA_output.csv")

    print(feature)
    GA_output[file] = feature
    output_csv = f"/content/drive/MyDrive/iot/results/afterGA/{file}_chosed_output_ML_.csv"
    target_names = ["Benign", file]
    ML_CV(train, test, output_csv, feature, file, 5)

# %%
with open('/content/drive/MyDrive/iot/GA_output_ET.json', 'w') as fp:
    json.dump(GA_output, fp)

# %%
# !shutdown /s /t 360

# %%
GA_output

# %%
a = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Min',
     'Fwd Pkt Len Mean', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
     'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std',
     'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Bwd Pkts/s',
     'Pkt Len Min', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'ACK Flag Cnt',
     'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Bwd Seg Size Avg',
     'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Bwd Byts/b Avg', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
     'Init Bwd Win Byts', 'Active Mean', 'Active Std', 'Active Max', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
     'Label']

# %%



