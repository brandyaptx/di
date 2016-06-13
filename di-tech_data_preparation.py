
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# ### 0. 测试上传数据
# - 文件名: test_upload.csv
# - 假设结果正态分布，用random数据。

# In[137]:

test = pd.read_csv('season_1/test_0.csv',header=None);test.describe()


# In[19]:

test.describe()


# In[23]:

import random
test_list = []
for i in range(len(test)):
    test_list.append(abs(random.gauss(0,20)))
random_array = np.array(test_list)


# In[35]:

type(random_array[1])


# In[25]:

random_forcast = pd.DataFrame(random_array)


# In[31]:

test_result = pd.concat([test[0],test[1],random_forcast[0]],axis=1);test_result.columns=[0,1,2]


# In[33]:

test_result.to_csv('test_upload.csv',header=False,index=False)


# ## 数据获取和清理

# In[2]:

import os

print os.getcwd()


# In[7]:

def output(day):
    order = pd.read_csv('season_2/training_data/order_data/order_data_{}'.format(day),delimiter='\t',header=None)
    order.columns = ["order_id", "driver_id","passenger_id","start_district_hash", "dest_district_hash", "Price","Time"]
    order["time"] = pd.to_datetime(order.Time,unit='s')
    order["time_id"] = order.time.apply(lambda x: x.hour*6 + int(x.minute/10)+1)
    # 地点信息
    district_data = pd.read_csv('season_2/training_data/cluster_map/cluster_map', delimiter='\t',header=None)
    district_data.columns = ["start_district_hash","district_id"]
    # 合并地区信息
    order_m = pd.merge(order,district_data,on="start_district_hash")
    # 简化表格，并计算gap
    std_df = order_m[["driver_id","time_id","district_id"]]
    # 整理 driver_id 信息，将 非NaN 都变为0，将NaN变为1
    mask = std_df.isnull()
    # convert the NaN data (gap) to 1 
    std_df['gap'] = np.where(mask.driver_id,1,0)
    # rid the hash data
    std_df =std_df.drop("driver_id",axis=1)
    # 
    std_output = std_df.groupby(["district_id","time_id"]).sum()
    
    std_output.to_csv('order_data_{}_gap.csv'.format(day),index=True)
    


# In[8]:

for i in range(1,22):
    if i < 10:
        output("2016-01-0{}".format(i))
    else:
        output("2016-01-{}".format(i))


# In[51]:

df_traffic = pd.read_csv('season_2/training_data/traffic_data/traffic_data_2016-01-01',delimiter='\t',header=None)
df_traffic.columns = ["district_hash", "1","2","3", "4","Time"]
df_traffic["time"] = pd.to_datetime(df_traffic.Time,unit='s')
df_traffic["time_id"] = df_traffic.time.apply(lambda x: x.hour*6 + int(x.minute/10)+1)
district_data = pd.read_csv('season_2/training_data/cluster_map/cluster_map', delimiter='\t',header=None)
district_data.columns = ["district_hash","district_id"]
# 合并地区信息
district_data_m = pd.merge(df_traffic,district_data,on="district_hash")



# In[91]:

# 尝试读取
df_test_output = pd.read_csv('./order_data_2016-01-{}_gap.csv'.format("01"))
df_test_output

district_data_y = pd.merge(district_data_m,df_test_output,on=["district_id","time_id"])
district_data_y = district_data_y.sort(["district_id","time_id"],ascending=[1,1])

district_data_y.head()


# In[93]:

district_data_x = pd.merge(district_data_y,df_poi,left_on="district_hash",right_on=0)
district_data_x = district_data_x.drop(["district_hash",0],axis=1)


# ### 尝试订单信息获得
# - 具体步骤分解

# In[142]:

order_1 = pd.read_csv('season_1/training_data/order_data/order_data_2016-01-01', delimiter='\t',header=None)
order_1.columns = ["order_id", "driver_id","passenger_id","start_district_hash", "dest_district_hash", "Price","Time"]


# In[124]:

order_1.info()
# 可见 总体订单gap 为 (501287- 325577) / 501287 = 0.3505


# ### 订单信息数据整理
# - 时间切片，获得 time_id 对应 144 个时区
# - 地点切片，获得 district_id 对应的 66 个地区
# 

# In[125]:

# 如何把时间做切片？
order_1["time"] = pd.to_datetime(order_1.Time,unit='s');order_1.head()


# In[126]:

order_1["time_id"] = order_1.time.apply(lambda x: x.hour*6 + int(x.minute/10)+1)


# In[128]:

# 地点信息
district_data = pd.read_csv('season_1/training_data/cluster_map/cluster_map', delimiter='\t',header=None)
district_data.columns = ["start_district_hash","district_id"];district_data.head()


# In[129]:

# 合并地理信息入数据
order_1_m = pd.merge(order_1,district_data,on="start_district_hash");order_1_m.head()


# In[130]:

# 简化表格，并计算gap
std_df = order_1_m[["driver_id","time_id","district_id"]]
# 整理 driver_id 信息，将 非NaN 都变为0，将NaN变为1
mask = std_df.isnull();mask
# convert the NaN data (gap) to 1 
std_df['gap'] = np.where(mask.driver_id,1,0);std_df.head()


# In[131]:

std_df =std_df.drop("driver_id",axis=1);std_df.head()


# In[132]:

group0 = std_df.groupby(["district_id","time_id"])


# In[133]:

std_output = group0.sum() 


# In[76]:

# 获得order_1 的预测值了~
std_output.to_csv('order_1_gap.csv',index=True)


# ### 区域的地域属性数据获取

# ### 基准预测
