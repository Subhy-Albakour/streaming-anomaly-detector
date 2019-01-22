import pandas as pd

data_file="data/mulcross.csv"
df = pd.read_csv(data_file, comment='#')
y=df['Target'].values
anoms=(y=="'Anomaly'")
normal=(y=="'Normal'")
y[anoms]=1
y[normal]=0
X = df.drop(["Target"], axis=1)