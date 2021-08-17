import pandas as pd
import matplotlib.pyplot as plt

#表の作成
data = {"月日":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
      "販売数":[
701105,
701102,
701101,
701080,
701080,
701087,
701074,
701050,
701009,
701045,
701020,
701001,
700981,
700979,
700964,
700978,
700960,
700977,
700954,
701060,
 ]}

df = pd.DataFrame(data)

# 移動平均の計算

df["3日間移動平均"]=df["販売数"].rolling(3).mean().round(1)
df["5日間移動平均"]=df["販売数"].rolling(5).mean().round(1)
df["7日間移動平均"]=df["販売数"].rolling(7).mean().round(1)

plt.plot(df["月日"], df["販売数"], label="origin")
plt.plot(df["月日"], df["3日間移動平均"], "k--", label="SMA(3)")
# plt.plot(df["月日"], df["5日間移動平均"], "r--", label="SMA(5)")
# plt.plot(df["月日"], df["7日間移動平均"], "g--", label="SMA(7)")
plt.xticks(rotation=90)
plt.xlabel("i")
plt.ylabel("z")
plt.legend()
plt.ylim(700950, 701110)

plt.show()