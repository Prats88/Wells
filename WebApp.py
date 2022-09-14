# import os
from typing import Text
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
import traceback
# from filecmp import clear_cache
from fileinput import filename
# from turtle import clearscreen
from flask import Flask, render_template, request, send_file
# from tensorflow.python.tools import module_util as _module_util

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/About/')
def about():
    return render_template("about.html")


@app.route('/wells')
def wells():
    return render_template("wells.html")

@app.route('/success-table2', methods=['POST'])
def success_table2():
    global filename
    if request.method == "POST":
        file = request.files['file']
        try:
            df = pd.read_csv(file)
            df.to_csv("results.csv", index=None)
            # X = df1.iloc[:, 3:-1].values
            # y = df1.iloc[:, -1].values

            # print(X)
            # print(y)
            return render_template("wells.html", text=df.to_html())
        except:
            traceback.print_exc()

@app.route('/success-table', methods=['POST'])
def success_table():
    global filename
    if request.method == "POST":
        file = request.files['file']
        try:
            df = pd.read_csv(file)
            df.to_csv("WellsUploaded.csv", index=None)
            # X = df1.iloc[:, 3:-1].values
            # y = df1.iloc[:, -1].values

            # print(X)
            # print(y)
            return render_template("wells.html", text=df.to_html())
        except:
            traceback.print_exc()

df = pd.read_csv("WellsUploaded.csv")
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values            
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,6] = le.fit_transform(X[:,6])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

loss_dataframe = pd.DataFrame(ann.history.history)

loss_dataframe.plot()

y_pred = ann.predict(X_test)
y_pred = (y_pred >0.5)
result = print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#confusion matrix
pred_df = pd.DataFrame(y_test,columns=['Test true Y'])
ann.evaluate(X_test,y_test,verbose=0)
ann.evaluate(X_train,y_train,verbose=0)
test_prediction = ann.predict(X_test)
test_predictions = pd.Series(test_prediction.reshape(13,))
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df = pd.DataFrame(y_test,columns=['Test true Y'])
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test true Y','Model predictions']
# sns.scatterplot(data=pred_df)

pred_df.to_csv("results.csv", index=None)



if __name__ == "__main__":
         app.run(debug=True)
 