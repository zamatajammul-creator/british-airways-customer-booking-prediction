#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

booking = pd.read_csv('C:/Users/HP/Documents/british airways job simulation/customer_booking.csv', encoding='ISO-8859-1')

print(booking.head())


print(booking.isnull().sum())


label_cols = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']

le = LabelEncoder()
for col in label_cols:
    booking[col] = le.fit_transform(booking[col])


X = booking.drop('booking_complete', axis=1)
y = booking['booking_complete']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


print(classification_report(y_test, y_pred))


print(confusion_matrix(y_test, y_pred))


cv_scores = cross_val_score(model, X, y, cv=5)
print('Cross Validation Scores:', cv_scores)
print('Average CV Score:', cv_scores.mean())


plt.show()


# In[ ]:




