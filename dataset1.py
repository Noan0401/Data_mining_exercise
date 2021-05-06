#!/usr/bin/env python
# coding: utf-8

# In[47]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#真の関数として y = sin(pi * x * 0.8) の結果を返す関数を定義せよ。関数名は true_function とする。
def true_function(x):
    y = np.sin(np.pi * x * 0.8)
    return y


# In[25]:


#doctestにより、x=0のときy=0であることをテストせよ。(あとでします....)
true_function(0)


# In[46]:



def Plot_1_1():
    x = np.linspace(-1, 1)
    plt.plot(x, np.sin(np.pi * x * 0.8), color='b', ls='-', label='true_function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphs')
    plt.show()
    
Plot_1_1()


# ##演習1.2：観測点と真値の準備
# 
# 課題内容
# *観測点（x座標）を乱数により設定しよう。ここでは、定義域-1 <= x <= 1の範囲内でシード値（デフォルト値0）を指定した上で、ランダムにn個（デフォルト値20）の観測点を用意せよ。
# 
# *次に、各観測点に対応する真の値を、真の関数により求めよ。
# 
# *上記2手順により、観測点とそれに対応した真値が揃ったはずだ。このサンプル集合を pandas.DataFrame型（20行2列）として設定せよ。列の順番は-問わないが、各々「観測点」「真値」として列名を設定せよ。
# 
# *演習1.1の線グラフ上に、サンプル集合をプロットせよ。プロットは見やすいようにサイズ調整すること。グラフは ex1.2.png として保存すること。

# In[27]:


random_array = np.random.uniform(-1, 1, 20)
print(random_array)


# In[28]:


true_value = []
df_array = []

for i in random_array:
    true_value.append(true_function(i))
    df_array.append([i,true_function(i)])
    
print(true_value)
print(df_array)


# In[48]:


def Make_rendum_Observation_point():
    random_array = np.random.uniform(-1, 1, 20)
    true_value = []
    df_array = []

    for i in random_array:
        true_value.append(true_function(i))
        df_array.append([i,true_function(i)])
    return df_array


# In[54]:


import pandas as pd

df = pd.DataFrame(df_array, columns= ["Observation point", "true value"])

print(df)
print(type(df))

test_x = df["Observation point"]
print(test_x)


# In[60]:



def Make_DataFrame(array):
    df = pd.DataFrame(df_array, columns= ["Observation point", "true value"])
    return df

df_test = Make_DataFrame(Make_rendum_Observation_point())
df_test


# In[56]:


X = random_array
Y = true_value


def Plot_1_2(X,Y):
    x = np.linspace(-1, 1)
    plt.plot(x, np.sin(np.pi * x * 0.8), color='b', ls='-', label='true_function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphs')
    plt.scatter(X, Y, label='data set')
    plt.savefig("ex1.2.png")
    plt.show()

Plot_1_2(X,Y)
Plot_1_2(test_x,Y)


# 演習1.3：ノイズを付与した観測値の準備
# 背景
# 真の関数は本来は知ることができない。用意できるデータセットには、reporting bias, selection biasなど様々なバイアスが混入したサンプルが含まれる。このような「バイアスが混入したサンプル」を作るため、真の関数にノイズを付与してみよう。
# 課題内容
# サンプルに対応する真の値は true_function により得られる。その値に付与するノイズを正規分布（ガウス分布）により用意しよう。20個のサンプルに対して、平均値を0.0、分散を2.0としたときの正規分布に従うノイズ（ホワイトノイズ）を付与せよ。付与した値を、演習1.2で用意したDataFrameにおける新たな列として挿入せよ。列名を「観測値」とする。
# 演習1.2のグラフ内に、観測値をプロットせよ。グラフは ex1.3.png として保存すること。

# In[37]:


from scipy.stats import norm

White_noise= norm.pdf(true_value,0.0,2.0)


# In[38]:


df["White_noise"] = White_noise
print(df)


# In[62]:



def Make_white_noise(true_value, df):
    White_noise= norm.pdf(true_value,0.0,2.0)
    df["White_noise"] = White_noise
    return df

new_df = Make_white_noise(true_value,df_test)
new_df


# In[65]:



def Plot_1_3(X,Y,df):
   x = np.linspace(-1, 1)
   plt.plot(x, np.sin(np.pi * x * 0.8), color='b', ls='-', label='true_function')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('Graphs')
   plt.scatter(X, Y, label='data set')
   plt.plot(df["true value"],df["White_noise"],color='r')
   plt.savefig("ex1.3.png")
   plt.show()
   
Plot_1_3(X,Y,new_df)


# 演習1.4：データセットのファイル出力
# 課題内容
# DataFrameに保存しているデータをTSV形式でファイル出力せよ。

# In[66]:


def Output_tsv(df):
    df.to_csv('output.tsv', sep='\t', index=True)


# 演習1.5：データセットのファイル読み込み
# 課題内容
# TSVファイルで保存されているデータを、DataFrame型で読み込め。

# In[67]:


def Input_tsv(df):
    df_2 = pd.read_csv('output.tsv', sep='\t', index_col=0)
    return df_2


# 演習1.6：演習1.1〜演習1.5をモジュールとして整理
# 課題内容
# 演習1.1〜演習1.5をそれぞれ関数として定義し、他のファイルからimportで利用できるように整理せよ。ファイル名を dataset1.py とする。

# In[ ]:





# In[ ]:




