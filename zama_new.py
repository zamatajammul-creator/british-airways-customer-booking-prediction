#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
df=pd.read_csv("Downloads/customer_booking.csv",encoding="ISO-8859-1")
df.head()


# In[8]:


df.info()


# In[10]:


df["flight_day"].unique()


# In[36]:


df.describe()


# In[ ]:




