#!/usr/bin/env python
# coding: utf-8

# ## Importando pacotes

# In[3]:


pip install --upgrade scikit-learn


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


from sklearn.feature_selection import RFE


# In[4]:


data = pd.read_csv('base_modelo.csv')


# ## Explorando o Dataset

# In[5]:


## overview no dataset
data.head()


# In[6]:


data.rename(columns = {'y': 'is_churn'}, inplace = True)


# In[8]:


#total de nulos
col = data.columns
total_null=data.isnull().sum()
total_null


# In[9]:


## percent of null
data.isna().sum()/data.shape[0]*100


# In[10]:


##substituindo nulos por zero
data.fillna(0, inplace = True)


# In[11]:


data.info()


# In[12]:


## ajustando formatação dados das colunas
data['id'] = data['id'].astype(str)
data['safra'] = data['safra'].astype(str)


# In[13]:


## verificando quantidade de nulos
nulos = data.isnull()
nulos.sum()


# In[14]:


## métricas descritivas das variáveis
data.describe().round(2)


# In[15]:


# tamanho dataset
data.shape


# In[16]:


## contando variável dependente
data['is_churn'].value_counts()
sns.countplot(data['is_churn'])


# In[17]:


## correlação variáveis
correlation = data.corr().round(2)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
correlation


# In[19]:


##Visualizar em pares
df = pd.DataFrame(data)
s = correlation.unstack()
s = correlation.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
so


# In[20]:


## Separando base de dados (sem nulos)
y = data['is_churn']
x = data.drop(columns = ['id', 'safra', 'is_churn'])


# ## Modelo 1 - Usando base (com nulos zerados)

# In[21]:


## cutoff padrão do scikit learn = 0.5
SEED = 80
train_x, test_x, train_y, test_y = train_test_split (x, y, test_size = 0.3, random_state = SEED)


# In[22]:


model = LogisticRegression(max_iter = 100)
model.fit(train_x, train_y)


# In[23]:


## accurance test - model (1) with nulls
predicts = model.predict(test_x)
accuracy_1 = accuracy_score(test_y, predicts)
round(accuracy_1, 3) * 100
## cut off 0.5


# In[24]:


## confusion matrix
cm = confusion_matrix(test_y, predicts, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[25]:


## metrics about the confusion matrix
print(classification_report(test_y, predicts))


# In[26]:


## specificity e sensitivy on roc curve
prob = model.predict_proba(test_x)[:, 1]
tfp, tvp, limit = roc_curve(test_y, prob)
print('roc_auc', roc_auc_score(test_y, prob))


# In[27]:


plt.subplots(1, figsize = (5,5))
plt.title('ROC curve')
plt.plot(tfp, tvp)
plt.xlabel('Specifity')
plt.ylabel('Sensibility')
plt.plot([0,1], ls = "--", c = 'black')
plt.plot([0,0], [1,0], ls = "--", c = 'red'), plt.plot([1,1], ls="--", c = 'red')
plt.show()


# ## Modelo 2 - Usando base com cut off menor

# In[28]:


##MODELO COM CUT OFF DIFERENTE
threshold = 0.3
pred_2 = pd.Series(prob).map(lambda x: 1 if x > threshold else 0)
print(classification_report(test_y, pred_2))


# In[29]:


accuracy_2 = accuracy_score(test_y, pred_2)
round(accuracy_2, 3) * 100
## cut off 0.3


# ## Modelo 3 - Fazendo stepwise

# In[30]:


glm = sm.GLM(test_y, test_x, family=sm.families.Binomial())
res = glm.fit()
print(res.summary())


# In[31]:


## Tirando variáveis com p>0.05 (visando entrender possível melhora do modelo)
y = data['is_churn']
x = data.drop(columns = ['id', 'safra', 'is_churn', 'VAR_15', 'VAR_21', 'VAR_22', 'VAR_34', 'VAR_42', 'VAR_45', 'VAR_45', 'VAR_48', 'VAR_52', 'VAR_54', 'VAR_56', 'VAR_61', 'VAR_62', 'VAR_64', 'VAR_65', 'VAR_69', 'VAR_71', 'VAR_73', 'VAR_77'])


# In[32]:


## cutoff padrão do scikit learn = 0.5
SEED = 80
train_x, test_x, train_y, test_y = train_test_split (x, y, test_size = 0.3, random_state = SEED)


# In[33]:


model = LogisticRegression(max_iter = 100)
model.fit(train_x, train_y)


# In[34]:


## accurance test - model (1) with nulls
predicts = model.predict(test_x)
accuracy_1 = accuracy_score(test_y, predicts)
round(accuracy_1, 3) * 100
## cut off 0.5


# In[35]:


## confusion matrix
cm = confusion_matrix(test_y, predicts, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[36]:


## metrics about the confusion matrix
print(classification_report(test_y, predicts))


# In[37]:


## specificity e sensitivy on roc curve
prob = model.predict_proba(test_x)[:, 1]
tfp, tvp, limit = roc_curve(test_y, prob)
print('roc_auc', roc_auc_score(test_y, prob))


# In[38]:


plt.subplots(1, figsize = (5,5))
plt.title('ROC curve')
plt.plot(tfp, tvp)
plt.xlabel('Specifity')
plt.ylabel('Sensibility')
plt.plot([0,1], ls = "--", c = 'black')
plt.plot([0,0], [1,0], ls = "--", c = 'red'), plt.plot([1,1], ls="--", c = 'red')
plt.show()


# In[ ]:


## Conclusão: Aumentou a performance em 0.03 pontos e o número de verdadeiros positivos também teve um crescimento, o que é improtante para o nosso modelo de crédito.


# ## Modelo 4 - Variáveis normalizadas + stepwise

# In[39]:


colunas = data.columns
scaler_minMax = MinMaxScaler()
data_normalize = pd.DataFrame(scaler_minMax.fit_transform(data),columns = colunas)
data_normalize.head()


# In[40]:


y = data_normalize['is_churn']
x = data_normalize.drop(columns = ['id', 'safra', 'is_churn', 'VAR_15', 'VAR_21', 'VAR_22', 'VAR_34', 'VAR_42', 'VAR_45', 'VAR_45', 'VAR_48', 'VAR_52', 'VAR_54', 'VAR_56', 'VAR_61', 'VAR_62', 'VAR_64', 'VAR_65', 'VAR_69', 'VAR_71', 'VAR_73', 'VAR_77'])


# In[41]:


## cutoff padrão do scikit learn = 0.5
SEED = 80
train_x, test_x, train_y, test_y = train_test_split (x, y, test_size = 0.3, random_state = SEED)


# In[42]:


model = LogisticRegression(max_iter = 100)
model.fit(train_x, train_y)


# In[43]:


## accurance test - model (1) with nulls
predicts = model.predict(test_x)
accuracy_1 = accuracy_score(test_y, predicts)
round(accuracy_1, 3) * 100
## cut off 0.5


# In[44]:


## confusion matrix
cm = confusion_matrix(test_y, predicts, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[45]:


## metrics about the confusion matrix
print(classification_report(test_y, predicts))


# In[46]:


plt.subplots(1, figsize = (5,5))
plt.title('ROC curve')
plt.plot(tfp, tvp)
plt.xlabel('Specifity')
plt.ylabel('Sensibility')
plt.plot([0,1], ls = "--", c = 'black')
plt.plot([0,0], [1,0], ls = "--", c = 'red'), plt.plot([1,1], ls="--", c = 'red')
plt.show()


# In[ ]:


##MODELO APRESENTOU A MELHOR PERFORMANCE, VISTO A NORMALIZAÇÃO E O STEPWISE JUNTOS


# ## Modelo 5 - stepwise + rfe (feature selection)

# In[47]:


colunas = data.columns
scaler_minMax = MinMaxScaler()
data_normalize = pd.DataFrame(scaler_minMax.fit_transform(data),columns = colunas)
data_normalize.head()


# In[48]:


y_2 = data['is_churn']
x_2 = data.drop(columns = ['id', 'safra', 'is_churn'])


# In[49]:


rfe = RFE(estimator=model, n_features_to_select=30, step=1)
fit = rfe.fit(x_2, y_2)


# In[50]:


rfe_support = fit.support_
rfe_feature = x_2.loc[:, rfe_support].columns.tolist()
features = rfe_feature
features


# In[51]:


## Ajustando base de x e y
y = data_normalize['is_churn']
x = data_normalize.drop(columns = ['id', 'safra', 'is_churn', 'VAR_1','VAR_5','VAR_9','VAR_11','VAR_18','VAR_19','VAR_20','VAR_22','VAR_25','VAR_27','VAR_28','VAR_29','VAR_31','VAR_33','VAR_36','VAR_37','VAR_40','VAR_41','VAR_42','VAR_43','VAR_44','VAR_46','VAR_48','VAR_49','VAR_55','VAR_57','VAR_61','VAR_64','VAR_74','VAR_78'])


# In[52]:


## cutoff padrão do scikit learn = 0.5
SEED = 80
train_x, test_x, train_y, test_y = train_test_split (x_2, y_2, test_size = 0.3, random_state = SEED)


# In[53]:


model = LogisticRegression(max_iter = 100)
model.fit(train_x, train_y)


# In[54]:


## accurance test - model (1) with nulls
predicts = model.predict(test_x)
accuracy_1 = accuracy_score(test_y, predicts)
round(accuracy_1, 3) * 100
## cut off 0.5


# In[55]:


## confusion matrix
cm = confusion_matrix(test_y, predicts, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# In[56]:


## metrics about the confusion matrix
print(classification_report(test_y, predicts))


# # # Modelo 6 - dropando os nulos

# In[57]:


## NÃO É POSSÍVEL, DEVIDO A QUANTIDADE DE NUOS NA BASE

