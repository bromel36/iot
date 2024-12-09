# %%
import os
import pandas as pd

# %%
def folder(f_name): #this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")

# %%
folder("pcaps")

# %%
def find_the_way(path,file_format,con=""):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                if con in file:
                    files_add.append(os.path.join(r, file))  
            
    return files_add

# %%
files_add=find_the_way('../', '.pcap')
files_add

# %%
for i in files_add:
    temp=i[2:]
    temp=temp.replace(" ","_")
    temp=temp.replace("\\","-")
    temp=f"./pcaps/{temp}"
    os.rename(i,temp)
    print(temp)

# %%
files_add=find_the_way('../', '.pcap')
uzun=len(files_add)
uzun

# %%
df = pd.read_csv("../dataset_description.csv")
df

# %%
df.columns

# %%
name=df['File Name']
rule=df['Rule']
cat=df['Category']
subcat=df['Sub-category']

# %%
name

# %%
for i in range(len(name)):
    add=str(cat[i])+"@"+str(subcat[i])+"_"
    add=add.replace(" ","_")
    command='tshark -Y \"'+str(rule[i])+"\" -r ./pcaps/"+str(name[i])+" -T fields -e frame.number > ./pcaps/attack_"+str(name[i])[:-4]+"csv_flow"
    os.system(command)
    command='tshark -Y \"!('+str(rule[i])+")\" -r ./pcaps/"+str(name[i])+" -T fields -e frame.number > ./pcaps/normal_"+str(name[i])[:-4]+"csv_flow"
    os.system(command)
    files_add=find_the_way('../', '.pcap')
    if uzun==len(files_add):    
        print(command,"\n")
    uzun=len(files_add) 
 
    

# %%
"""
# MERGE CSVS
"""

# %%
name_list=find_the_way('../pcaps', '.csv_flow')

# %%
label_files=[]
for i in name_list:
    temp=i[15:]
    if temp not in label_files:
        label_files.append(temp)
    

# %%
for i in label_files:

    name=find_the_way('../pcaps', i)
    print(name)

    dfA = pd.read_csv(name[0], header=None)
    dfA=dfA.rename(columns={0: "PacketNumber"})
    dfA["Label"]=[1] * len(dfA)
    
    
    dfN = pd.read_csv(name[1], header=None)
    dfN=dfN.rename(columns={0: "PacketNumber"})    
    dfN["Label"]=[0] * len(dfN) 

    df = pd.concat([dfA,dfN])
    df=df.sort_values('PacketNumber')
    df.to_csv(f"./pcaps/{i}",  index=False)
    print(i,df.groupby("Label").size(),"\n\n\n")
    os.remove(f"./pcaps\\normal_{i}")
    os.remove(f"./pcaps\\attack_{i}")

# %%
#!shutdown /s /t 36

# %%
