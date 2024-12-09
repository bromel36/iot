# %%
import os
import pandas as pd


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
def folder(f_name):  # this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print("The folder could not be created!")


# %%
folder("./pcaps/csvs")
folder("./pcaps/labelled")

# %%
files_add = find_the_way('../pcaps', '.pcap')
uzun = len(files_add)
uzun

# %%
df = pd.read_csv("../dataset_description.csv")
df

df = df.dropna(how='all')
df["File Name"] = df["File Name"].fillna(method="ffill")

# %%
files = {}
for i in df["File Name"].values:
    files[i] = files.get(i, 0) + 1
files

# %%
normal_rules = {}
for i in files:
    if (files[i]) > 1:
        print(i)
        temp = df[df["File Name"] == i].values
        if len(temp) < 2:  # Kiểm tra nếu không đủ dữ liệu
            print(f"Skipping {i}: Not enough rows in dataset_description.csvs")
            continue
        rule = f"!({temp[0][4]}) && !({temp[1][4]})"
        normal_rules[i] = rule

# %%
normal_rules

# %%
df.columns

# %%
name = df['File Name']
rule = df['Rule']
cat = df['Category']
subcat = df['Sub-category']

# %%
subcat

# %%
for i in range(len(name)):
    output = str(name[i])[:-5] + "_" + str(subcat[i]) + ".csv"
    output = output.replace(" ", "-")
    print(output)
    if files[name[i]] == 1:
        add = str(cat[i]) + "@" + str(subcat[i]) + "_"
        add = add.replace(" ", "_")
        command = 'tshark -Y \"' + str(rule[i]) + "\" -r ./pcaps/" + str(
            name[i]) + f" -T fields -e frame.number > ./pcaps/attack_" + output
        os.system(command)
        command = 'tshark -Y \"!(' + str(rule[i]) + ")\" -r ./pcaps/" + str(
            name[i]) + f" -T fields -e frame.number > ./pcaps/normal_" + output
        os.system(command)
        files_add = find_the_way('../', '.pcap')
        if uzun == len(files_add):
            print(command, "\n")
        uzun = len(files_add)
    else:
        add = str(cat[i]) + "@" + str(subcat[i]) + "_"
        add = add.replace(" ", "_")
        command = 'tshark -Y \"' + str(rule[i]) + "\" -r ./pcaps/" + str(
            name[i]) + f" -T fields -e frame.number > ./pcaps/attack_" + output
        os.system(command)
        command = 'tshark -Y \"' + str(normal_rules[name[i]]) + "\" -r ./pcaps/" + str(
            name[i]) + f" -T fields -e frame.number > ./pcaps/normal_" + output
        os.system(command)
        files_add = find_the_way('../', '.pcap')
        if uzun == len(files_add):
            print(command, "\n")
        uzun = len(files_add)

    # %%

# %%
name_list = find_the_way('../pcaps/', '.csv')
for i in name_list:
    os.rename(i, i.replace("./pcaps", './pcaps/csvs'))

# %%


# %%


# %%
"""
# MERGE CSVS
"""


# %%
def merger(name_list, name):
    flag = 1
    df = pd.read_csv(name_list[0])
    col_names = list(df.columns)
    empty = pd.DataFrame(columns=col_names)
    empty.to_csv(name, mode="w", index=False)  # ,header=False)

    for i in name_list:
        if "normal" in i:
            if flag:
                df = pd.read_csv(i)
                df.to_csv(name, mode="a", index=False, header=False)
                df = pd.read_csv(name)
                flag = 0
        else:
            df = pd.read_csv(i)
            df.to_csv(name, mode="a", index=False, header=False)

    df = pd.read_csv(name)
    df = df.sort_values('PacketNumber')
    df.to_csv(name, index=False)


# %%
name_list = find_the_way('../pcaps/csvs/', '.csv')
for i in name_list:
    if "normal" in i:
        label = "Benign"
    else:
        temp = i[8:-4]
        label = temp.split("_")[2]

    df = pd.read_csv(i, names=["PacketNumber"], header=None)
    df["Label"] = len(df) * [label]
    df.to_csv(i, index=False)

# %%
temp

# %%
name_list = find_the_way('../pcaps', '.csv')
for i in name_list:
    temp = i[8:-4]
    temp = temp.split("_")
    print(temp[1])
    name = f"./pcaps/labelled/{temp[1]}.csv"
    sub_list = find_the_way('../pcaps/csvs/', temp[1])
    merger(sub_list, name)

# %%
