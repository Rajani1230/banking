# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:17:32 2019

@author: yasin
"""

from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Tk()
root.title('Prediction Bank loan')
root.geometry('850x650')
root.configure(background="purple2")

var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="purple2")
var.set('Prediction Bank loan')
label.grid(row=0,columnspan=6)

data = ""
data1 = ""
#data cleansing
def train_file():
     root1=Tk()
     root1.title("login page")
     root1.geometry('600x500')
     root1.configure(background="purple2")
     def login():
         user = E.get()
         password = E1.get()
         admin_login(user,password)
     L  = Label(root1, text = "Username",bd=8,background="purple2",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 0,column=0)
     E  = Entry(root1)
     E.grid(row = 0, column = 1)
     L1 = Label(root1, text = "Password",bd=8,background="purple2",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 1,column=0)
     E1 = Entry(root1,show="*")
     E1.grid(row = 1, column = 1)
     B1 = Button(root1,text="Login",width=4,height=1,command=login,bd=8,background="purple2")
     B1.grid(row = 2, column = 1)
     #root1.mainloop()

def admin_login(user,password):
     #print(user,password)
     if user == "admin" and password == "admin":
         root3 = Tk()
         root3.title('choose file')
         root3.geometry('600x300')
         root3.configure(background="purple2")
         E2=Button(root3,text="Browse file",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="purple2",command=OpenFile_train)
         E2.place(x=200,y=100)
         
         root3.mainloop()  
     else:
         root3 = Tk()
         root3.title('ERROR')
         L2 = Label(root3, text = "user name and password is wrong",font=('arial',16,'bold'),fg='red').grid(row = 2)
         root3.mainloop()
              
def OpenFile_train():
    global data,data1
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("CSV File", "*.csv"),("All Files","*.*")),
                           title = "Choose a file.")
    try:
        with open(name,'r') as UseFile:
            data = pd.read_csv("Bank loan data.csv")
            data = data.drop(['Loan_ID','CoapplicantIncome'],axis=1)
            data = data.fillna(data.mean())
            data = data.dropna()
            data.rename(columns = {'Self_Employed':'SelfEmployed','Loan_Amount_Term':'LoanAmountTerm','Credit_History':'CreditHistory','Property_Area':'PropertyArea','Loan_Status':'LoanStatus'},inplace = True)
            data1 = data
            train()
    except FileNotFoundError:
         print("No file exists") 





def train():
    global data,data1,le
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    Gender_tr=le.fit_transform(data1.Gender) 
    #print("gender ",data1.Gender)
    #print(Gender_tr)
    PropertyArea_tr=le.fit_transform(data1.PropertyArea)  
    #print("PropertyArea",data1.PropertyArea)
    #print(PropertyArea_tr)
    Married_tr=le.fit_transform(data1.Married)
    #print("Married",data1.Married)
    #print("Married",Married_tr)
    Education_tr=le.fit_transform(data1.Education) 
    #print("Education",data1.Education)
    #print("Education",Education_tr)
    SelfEmployed_tr=le.fit_transform(data1.SelfEmployed) 
    #print("SelfEmployed",data1.SelfEmployed)
    #print(SelfEmployed_tr)
    LoanStatus_tr=le.fit_transform(data1.LoanStatus)  
    #print("LoanStatus",data1.LoanStatus)
    #print(LoanStatus_tr)
    Dependents_tr=le.fit_transform(data1.Dependents)  
    #print("Dependents",data1.Dependents)
    #print("Dependents",Dependents_tr)
    
    testing = data1
    
    testing['Gender'] = Gender_tr
    testing['PropertyArea'] = PropertyArea_tr
    testing['Married'] = Married_tr
    testing['Education'] = Education_tr
    testing['SelfEmployed'] = SelfEmployed_tr
    testing['LoanStatus'] = LoanStatus_tr
    testing['Dependents'] = Dependents_tr

    X=testing.values[:,:10]
    Y=testing.values[:,-1]
    global X_train,X_test,y_train,y_test,logreg
    from sklearn.linear_model  import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
    
    from sklearn.linear_model import LogisticRegression
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    #global X_train,X_test,y_train,y_test,logreg 
    y_pred = logreg.predict(X_test)
    
    from sklearn import metrics
    from PIL import ImageTk,Image
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)



    #%matplotlib inline
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('graph1.png',dpi=199)
    plt.show()
    
    image = Image.open("graph1.png")
    image = image.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)  
    panel1 = Label(root, image=img)
    panel1.image = img
    panel1.grid(row=2,column=2)
    
    data_out = "Accuracy: {} \nPrecision: {}\nRecall: {}".format(metrics.accuracy_score(y_test, y_pred),metrics.accuracy_score(y_test, y_pred),metrics.recall_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    
    labelText = StringVar()
    labelText.set(data_out)
    output = Label(root, textvariable=labelText,width=45, height=6,bg="purple2")
    output.grid(row=3,column=4)

    
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig('graph2.png',dpi=199)
    plt.show()

    image = Image.open("graph2.png")
    image = image.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)  
    panel1 = Label(root, image=img)
    panel1.image = img
    panel1.grid(row=2,column=5)

    print(X_test[0])
    
def predict():
    from tkinter import ttk
    root10 = Tk()
    root10.title('Predict Defaultor or Non-Defalutor')
    root10.geometry('850x650')
    root10.configure(background="Purple3")
    
    """var = StringVar()
    label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="Powderblue")
    var.set("Predict STRESS DETECTOR")
    label.grid(row=0,columnspan=6)
    """
    label_1 = ttk.Label(root10, text ='Loan_ID',font=("Helvetica", 16),background="Purple3")
    label_1.grid(row=0,column=0)
    
    Entry_1 = Entry(root10)
    Entry_1.grid(row=0,column=1)
    
    label_2 = ttk.Label(root10, text = 'Gender MALE:1 Female:0',font=("Helvetica", 16),background="Purple3")
    label_2.grid(row=1,column=0)
    
    Entry_2 = Entry(root10)
    Entry_2.grid(row=1,column=1)
    
    label_3 = ttk.Label(root10, text = 'Married NO:0 YES:1',font=("Helvetica", 16,),background="Purple3")
    label_3.grid(row=2,column=0)
    
    Entry_3 = Entry(root10)
    Entry_3.grid(row=2,column=1)
    
    label_4 = ttk.Label(root10, text = 'NO of Dependents (0-3)' ,font=("Helvetica", 16),background="Purple3")
    label_4.grid(row=3,column=0)
    
    Entry_4 = Entry(root10)
    Entry_4.grid(row=3,column=1)
    
    label_5 = ttk.Label(root10, text = 'Education Graduate:0 Non-Graduate : 1',font=("Helvetica", 16),background="Purple3")
    label_5.grid(row=4,column=0)
    
    Entry_5 = Entry(root10)
    Entry_5.grid(row=4,column=1)
    
    label_6 = ttk.Label(root10, text = 'Self Employed NO:0 yes:1',font=("Helvetica", 16),background="Purple3")
    label_6.grid(row=5,column=0)
    
    Entry_6 = Entry(root10)
    Entry_6.grid(row=5,column=1)
    
    label_7 = ttk.Label(root10, text = 'Applicant Income',font=("Helvetica", 16),background="Purple3")
    label_7.grid(row=6,column=0)
    
    Entry_7 = Entry(root10)
    Entry_7.grid(row=6,column=1)
    
    label_8 = ttk.Label(root10, text = 'Coapplicant Income',font=("Helvetica", 16),background="Purple3")
    label_8.grid(row=7,column=0)
    
    Entry_8 = Entry(root10)
    Entry_8.grid(row=7,column=1)
    
    
    label_9 = ttk.Label(root10, text = 'Loan Amount',font=("Helvetica", 16),background="Purple3")
    label_9.grid(row=8,column=0)
    
    Entry_9 = Entry(root10)
    Entry_9.grid(row=8,column=1)
    
    label_10 = ttk.Label(root10, text = 'Loan Amount Term',font=("Helvetica", 16),background="Purple3")
    label_10.grid(row=9,column=0)
    
    Entry_10 = Entry(root10)
    Entry_10.grid(row=9,column=1)
    
    label_11 = ttk.Label(root10, text = 'Credit History yes:1 No:0 ',font=("Helvetica", 16),background="Purple3")
    label_11.grid(row=10,column=0)
    
    Entry_11 = Entry(root10)
    Entry_11.grid(row=10,column=1)
    
    label_11 = ttk.Label(root10, text = 'Property_Area urban:0 semiurban:1 rural:2',font=("Helvetica", 16),background="Purple3")
    label_11.grid(row=11,column=0)
    
    Entry_11 = Entry(root10)
    Entry_11.grid(row=11,column=1)
    
    label_12 = ttk.Label(root10, text ='Loan status Yes:1 No:0',font=("Helvetica", 16),background="Purple3")
    label_12.grid(row=12,column=0)
    
    Entry_12 = Entry(root10)
    Entry_12.grid(row=12,column=1)
    
    global labelText

    def predout_svn():
        global labelText,logreg,le,X_test
        data = (float(Entry_2.get()),float(Entry_3.get()),float(Entry_4.get()),float(Entry_5.get()),float(Entry_6.get()),float(Entry_7.get()),float(Entry_8.get()),float(Entry_9.get()),float(Entry_10.get()),float(Entry_11.get()))
        
        list_1 =[data]
        print(X_test[0])
        print(type(Entry_2.get()))
        print(list_1)
        out_pred = logreg.predict(list_1)
        
        if out_pred == 1:
            output.delete(0, END)
            output.insert(0,"Defalutor")
        else :
            output.delete(0, END)
            output.insert(0,"NON-Deflautor")
        labelText = StringVar()
        labelText.set(data)
        
        
    label_27 = Button(root10, text = 'predict',font=("Helvetica", 16),background="Purple3",command = predout_svn)
    label_27.grid(row=13,column=0)
    

    output = Entry(root10)
    output.grid(row=13,column=1)
    
   
B = Button(root, text = "Train",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="purple2",command = train_file)
B.grid(row=1,column=0)

labelText = StringVar()
labelText.set("")
output = Label(root, textvariable=labelText,width=45, height=6,bg="purple2")
output.grid(row=3,column=4)

B1 = Button(root, text = "Predict",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="purple2",command = predict)
B1.grid(row=1,column=5)

root.mainloop()