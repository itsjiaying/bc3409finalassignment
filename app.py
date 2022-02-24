#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import pickle

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        
        modelLR = pickle.load(open("logistic_regression.sav", 'rb'))
        predLR = modelLR.predict([[float(income), float(age), float(loan)]])
        sLR = "Predicted default on credit card (Logistic Regression model): " + str(predLR)
        
        modelDT = pickle.load(open("decision_tree.sav", 'rb'))
        predDT = modelDT.predict([[float(income), float(age), float(loan)]])
        sDT = "Predicted default on credit card (Decision Tree model): " + str(predDT)
    
        modelRF = pickle.load(open("random_forest.sav", 'rb'))
        predRF = modelRF.predict([[float(income), float(age), float(loan)]])
        sRF = "Predicted default on credit card (Random Forest model): " + str(predRF)
    
        modelXG = pickle.load(open("xgboost.sav", 'rb'))
        predXG = modelXG.predict([[float(income), float(age), float(loan)]])
        sXG = "Predicted default on credit card (XGBoost model): " + str(predXG)
    
        modelMLP = pickle.load(open("mlp.sav", 'rb'))
        predMLP = modelMLP.predict([[float(income), float(age), float(loan)]])
        sMLP = "Predicted default on credit card (MLPClassifier model): " + str(predMLP)
        
        return(render_template("index.html", result1 = sLR, result2 = sDT, result3 = sRF, result4 = sXG, result5 = sMLP))
    
    else: 
        s = "."
        return(render_template("index.html", result1 = s, result2 = s, result3 = s, result4 = s, result5 = s))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




