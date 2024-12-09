# %%
import numpy as np
import pandas as pd
import os

# %%
"""
### Discovering csv_flow extension files under "csvs_label_maker" folder.
"""


# %%
def find_the_way(path, file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))
    return files_add


# name_list = find_the_way('./csvs_label_maker', '.csv_flow')
# name_list

"""
# SYN FLOOD
"""
"""
### IoT-NID 1 and 2
"""

# %%
file_list = ['./csv_flow/dos-synflooding-1-dec.pcap_Flow.csv_flow',
             './csv_flow/dos-synflooding-2-dec.pcap_Flow.csv_flow']

for file in file_list:
    df = pd.read_csv(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[0][0:4] == "222.":
            if i[1] == '192.168.0.13':
                if i[2] == 554:
                    if i[3] == 6:
                        label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################

    df['Label'] = label
    df.to_csv(file, index=False)

"""
### IoT-NID 3
"""
file_list = ['./csv_flow/dos-synflooding-3-dec.pcap_Flow.csv_flow']

for file in file_list:
    df = pd.read_csv(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[0][0:4] == "111.":
            if i[1] == '192.168.0.13':
                if i[2] == 554:
                    if i[3] == 6:
                        label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################

    df['Label'] = label
    df.to_csv(file, index=False)

"""
### IoT-NID 4,5, and 6
"""

file_list = ['./csv_flow/dos-synflooding-4-dec.pcap_Flow.csv_flow',
             './csv_flow/dos-synflooding-5-dec.pcap_Flow.csv_flow',
             './csv_flow/dos-synflooding-6-dec.pcap_Flow.csv_flow']

for file in file_list:
    df = pd.read_csv(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[0][0:4] == "111.":
            if i[1] == '192.168.0.24':
                if i[2] == 19604:
                    if i[3] == 6:
                        label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################

    df['Label'] = label
    df.to_csv(file, index=False)

"""
# HTTP FLOOD
"""

name_list = find_the_way('../csv_flow', 'http')
name_list

# %%
for file in name_list:
    df = pd.read_csv(file)
    print(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[1] == '210.89.164.90':
            label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################
    df['Label'] = label
    df.to_csv(file, index=False)


"""
# ACK FLOOD
"""

name_list = [
    './csv_flow/mirai-ackflooding-1-dec.pcap_Flow.csv_flow',
    './csv_flow/mirai-ackflooding-2-dec.pcap_Flow.csv_flow',
    './csv_flow/mirai-ackflooding-3-dec.pcap_Flow.csv_flow',
    './csv_flow/mirai-ackflooding-4-dec.pcap_Flow.csv_flow']

# %%
for file in name_list:
    df = pd.read_csv(file)
    print(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[1] == '210.89.164.90':
            label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################
    df['Label'] = label
    df.to_csv(file, index=False)

# %%
"""
# UDP FLOOD
"""

# %%
name_list = find_the_way('../csv_flow', 'udp')
name_list

# %%
for file in name_list:
    df = pd.read_csv(file)
    print(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[1] == '210.89.164.90':
            label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################
    df['Label'] = label
    df.to_csv(file, index=False)

"""
# MitM ARP
"""

# %%
name_list = find_the_way('../csv_flow', 'MitM-ARP')
name_list

# %%
for file in name_list:
    df = pd.read_csv(file)
    df["ID1"] = df["IP_src"] + "-" + df["IP_dst"] + "-" + df["sport"].apply(str) + "-" + df["dport"].apply(str) + "-" + \
                df["IP_proto"].apply(str)
    IDS = {}
    for j in df["ID1"].unique():
        filtered = df[df["ID1"] == j]
        temp = dict(filtered.groupby("Label").size())
        if len(temp) == 1:
            IDS[j] = temp

    flow = pd.read_csv(f'./mitm-arpspoofing-{file[-5]}-dec.pcap_Flow.csv_flow')
    new_label = []
    for j in flow["Flow ID"]:
        try:
            new_label.append(list(IDS[j].keys())[0])
        except:
            new_label.append("No Label")
    flow["Label"] = new_label
    flow = flow[flow["Label"] != "No Label"]
    print(file, flow.groupby("Label").size(), "\n", "=" * 100)
    flow.to_csv(f'./csvs_label_maker/mitm-arpspoofing-{file[-5]}-dec.pcap_Flow.csv_flow', index=False)


# %%
name_list = ['./csv_flow/scan-hostport-1-dec.pcap_Flow.csv_flow',
             './csv_flow/scan-hostport-2-dec.pcap_Flow.csv_flow',
             './csv_flow/scan-hostport-3-dec.pcap_Flow.csv_flow']
# RULE ip.src == 192.168.0.15 and ip.dst == 192.168.0.13

# %%
for file in name_list:
    df = pd.read_csv(file)
    print(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if (i[0] == '192.168.0.15') and (i[1] == '192.168.0.13'):
            label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################
    df['Label'] = label
    df.to_csv(file, index=False)
    print(df.groupby("Label").size())

# %%
name_list = ['./csv_flow/scan-hostport-4-dec.pcap_Flow.csv_flow',
             './csv_flow/scan-hostport-5-dec.pcap_Flow.csv_flow',
             './csv_flow/scan-hostport-6-dec.pcap_Flow.csv_flow']

# RULE  ip.src == 192.168.0.15 and ip.dst == 192.168.0.24

# %%
for file in name_list:
    df = pd.read_csv(file)
    print(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if (i[0] == '192.168.0.15') and (i[1] == '192.168.0.24'):
            label.append(1)
        else:
            label.append(0)
    # RULES  ##################################################
    df['Label'] = label
    df.to_csv(file, index=False)
    print(df.groupby("Label").size())
#


"""
# Brute ForCE
"""

# %%
# file_list = find_the_way('./csvs_label_maker', 'brute')
file_list = ['./csv_flow/mirai-hostbruteforce-1-dec.pcap_Flow.csv_flow',
             './csv_flow/mirai-hostbruteforce-3-dec.pcap_Flow.csv_flow',
             './csv_flow/mirai-hostbruteforce-5-dec.pcap_Flow.csv_flow']

# %%
for file in file_list:
    df = pd.read_csv(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[0] == "192.168.0.13":
            if i[3] == 6:
                label.append(1)
            else:
                label.append(0)

        else:
            label.append(0)
    # RULES  ##################################################

    df['Label'] = label
    df.to_csv(file, index=False)
    print(df.groupby("Label").size())

# %%
file_list = [
    './csv_flow/mirai-hostbruteforce-2-dec.pcap_Flow.csv_flow',

    './csv_flow/mirai-hostbruteforce-4-dec.pcap_Flow.csv_flow']

# %%
for file in file_list:
    df = pd.read_csv(file)
    labeller = df[['Src IP', 'Dst IP', 'Dst Port', 'Protocol']]

    # RULES  ##################################################
    label = []
    for i in labeller.values:
        if i[0] == "192.168.0.24":
            if i[3] == 6:
                label.append(1)
            else:
                label.append(0)

        else:
            label.append(0)
    # RULES  ##################################################

    df['Label'] = label
    df.to_csv(file, index=False)
    print(df.groupby("Label").size())
