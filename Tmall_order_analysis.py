#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


# In[2]:


fileName = r'D:\Python_data\tmall_order_report.csv'
df = pd.read_csv(fileName,engine='python',encoding='utf')


# In[5]:


df.head(10)


# In[17]:


df.tail(10)


# In[93]:


#查看数据类型
df.info()


# In[13]:


df.columns
#发现有空格


# In[3]:


colNameDict = {'收货地址 ':'收货地址','订单付款时间 ':'订单付款时间'}
df = df.rename(columns = colNameDict)


# In[28]:


#检查是否有重复值
df.duplicated().sum()


# In[29]:


#检查是否有缺失值
df.isnull().sum()


# In[4]:


#创建转化率字典
dict_convs = dict()


# In[5]:


#订单总笔数加入字典
key = '总订单数'
dict_convs[key] = len(df)
len(df)


# In[6]:


#付款订单数加入字典
key = '付款订单数'
dict_convs[key] = len(df[df['订单付款时间'].notnull()])


# In[7]:


#到款订单数加入字典
key = '到款订单数'
df_paid = df[df['买家实际支付金额']>0]
dict_convs[key] = len(df_paid)
dict_convs['到款订单数']


# In[8]:


#全额到款订单数加入字典
key = '全额到款订单数'
dict_convs[key] = len(df_paid[df_paid['退款金额']==0])
dict_convs['全额到款订单数']


# In[9]:


#呈现表格
df_convs = pd.Series(dict_convs,name = '订单数').to_frame()
df_convs


# In[10]:


df_convs.loc['总订单数','订单数']


# In[11]:


#计算总体转化率
name = '总体转化率'
total_convs = df_convs['订单数']/df_convs.loc['总订单数','订单数']*100
df_convs[name] = total_convs.apply(lambda x:round(x,0))
df_convs


# In[12]:


#计算单一环节转化率
name = '单一环节转化率'
df_single_convs = df_convs['订单数'].shift()
df_convs[name] = df_single_convs.fillna(df_convs.loc['总订单数','订单数'])
#df_convs[name]
df_convs[name] = (df_convs['订单数']/df_convs[name]*100).apply(lambda x : round(x,0))

df_convs[name]


# In[13]:


from pyecharts.charts import Funnel
from pyecharts import options as opts 


# In[14]:


zipped = zip(df_convs.index,df_convs[name])

name = '总体转化率'
funnel = Funnel().add(
                    series_name = name,
                    data_pair = list(zipped),
                    label_opts = opts.LabelOpts(position = 'inside')

)
funnel.set_series_opts(tooltip_opts = opts.TooltipOpts(formatter = '{b}<br/>{a}:{c}%'))
funnel.set_global_opts( title_opts = opts.TitleOpts(title = name) )
funnel.render_notebook()


# In[15]:


zipped = zip(df_convs.index,df_convs[name])

funnel = Funnel().add(
                    series_name=name,
                    data_pair=list(zipped),
                    label_opts=opts.LabelOpts(position = 'inside')

)
funnel.set_series_opts(tooltip_opts = opts.TooltipOpts(formatter = '{b}<br/>{a}:{c}%'))
funnel.set_global_opts( title_opts = opts.TitleOpts(title = name) )
funnel.render_notebook()


# In[16]:


#收货地址统计
plt.figure(figsize=(16,8))
plt.xticks(rotation=90)

df_destine = df.groupby('收货地址')['订单创建时间'].count().sort_values(ascending=False).to_frame()
df_destine['count'] = df_destine['订单创建时间']
sns.barplot(x = '收货地址',y='count',data = df_destine.reset_index())


# In[17]:


df.head(10)


# In[18]:


#整体订单趋势
df_trans = df
df_trans['订单创建时间'] = df['订单创建时间'].astype('datetime64')
df_trans = df_trans.set_index('订单创建时间')
df_trans.head(10)


# In[19]:


plt.figure(figsize=(10,6))
plt.xticks(rotation=90)
se_trans_month = df_trans.resample('D').count()

#resample时间要注意，时间得作为index
#思路就是，以一个一个新的时间段为index，描述各个column的特征
#以‘订单编号’为例，此图标明了在以天为单位的时间轴下，订单的增长趋势
se_trans_month = se_trans_month.reset_index()
sns.lineplot(x='订单创建时间',y='订单编号',data=se_trans_month)
#se_trans_month.plot()


# In[7]:


from pyecharts.charts import Line


# In[8]:


name = '订单数'

(
    Line()
    .add_xaxis(xaxis_data = se_trans_month['订单创建时间'])
    .add_yaxis(
        y_axis = se_trans_month['订单编号'],
        series_name = ''
    )

    
)


# In[5]:


#订单平均价格
df[df['买家实际支付金额']!=0].mean()


# In[83]:


#销量区域分布-地理图
#定义一个处理省份名称的函数
def removeRegion(region):
    resultList = []
    for value in region:
        if value.endswith('自治区'):
            if value == '内蒙古自治区':
                value = value[:3]
                resultList.append(value)
            else:
                value = value[:2]
                resultList.append(value)
        else:
            resultList.append(value)
    return resultList

df['收货地址'] = removeRegion(df['收货地址'])
df.head(20)


# In[5]:


#筛选出有效订单表
dfValid = df[df['买家实际支付金额']!=0]
dfValid


# In[6]:


#生成订单数量表基于省份
dfRegion = dfValid.groupby('收货地址')['买家实际支付金额'].count()
dfRegion = dfRegion
dfRegion


# In[69]:


#删除‘省’
dfRegion.index = dfRegion.index.str.replace('省','')


# In[8]:


import pyecharts.options as opts
from pyecharts.charts import Map


# In[9]:



def xxx(value):
    locationList = []
    for v in value:
        locationList.append(v)
    return locationList


# In[38]:


zzz = [list(i) for i in zip(dfRegion.index,dfRegion)]
zzz


# In[11]:


zipped = zip(xxx(dfRegion.index),xxx(dfRegion))
list(zipped)


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


#展示地理分布图
#zipped = zip(dfRegion.index,dfRegion['买家实际支付金额'])

name = '订单数'

(
    Map()
    .add(
        series_name = name,
        data_pair = zzz)
    .set_global_opts(
        title_opts = opts.TitleOpts(title = '各省份订单数量（分段型）'),
        visualmap_opts = opts.VisualMapOpts(max_=3000, is_piecewise=True),
    )
    .render_notebook()
)


# In[14]:


from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.charts import Map3D
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode
from pyecharts.datasets import register_url
from pyecharts.charts import HeatMap


# In[15]:


fileName = r'D:\Python_data\China_states_coordinates.csv'
dfP = pd.read_csv(fileName,engine='python',encoding='utf')


# In[16]:


dfP


# In[17]:


import pypinyin


# In[43]:


def pinyin(word):
 x = ''
 for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
     x += ''.join(i)
 return x


# In[54]:





# In[77]:


#把省份全部转化为拼音
#自定义一个func
def trans(value):
    for v in value:
            v = pinyin(v)
    return v


# In[82]:





# In[28]:


#3D地图试验 
c = (
    Map3D()
    .add_schema(
    maptype = "china",
    is_show_ground = True)
    
            
    .add(
    series_name = '随便',
    data_pair = example_data,
    type_= ChartType.BAR3D,
    maptype = "china",
    is_map_symbol_show = True

        
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
    .render_notebook()
 )


# In[29]:


c


# In[ ]:




