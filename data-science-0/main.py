#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[40]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[41]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# Análise questão 01

# In[42]:


black_friday.head()


# In[43]:


# print('Resposta questão 01: ', q1())


# Análise questão 02

# In[44]:


black_friday.info()


# In[45]:


black_friday.Gender.unique()


# In[46]:


black_friday.Age.unique()


# In[47]:


# print('Resposta questão 02: ', q2())


# Análise questão 03
# 

# In[48]:


# print('Resposta questão 03: ', q3())


# Análise questão 04
# 

# In[49]:


# print('Resposta questão 04: ', q4())


# Análise questão 05
# 

# In[50]:


# print('Resposta questão 05: ', q5())


# Análise questão 06
# 

# In[51]:


# print('Resposta questão 06: ', q6())


# Análise questão 07
# 

# In[52]:


# print('Resposta questão 07: ', q7())


# Análise questão 08
# 

# In[53]:


# print('Resposta questão 08: ', q8())


# Análise questão 09
# 

# In[54]:


# print('Resposta questão 09: ', q9())


# Análise questão 10
# 

# In[55]:


#Verifica a quantidade de valores nulos
black_friday.Product_Category_2.isnull().sum()


# In[56]:


#Verifica a quantidade de valores nulos
black_friday.Product_Category_3.isnull().sum()


# In[57]:


# print('Resposta questão 10: ', q10())


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[58]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    # pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[59]:


def q2():
    # Retorne aqui o resultado da questão 2.
    gender_age_filter = black_friday[(black_friday.Gender.values == "F") & (black_friday.Age.values == "26-35")]
    return gender_age_filter.shape[0]
    # pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[60]:


def q3():
    # Retorne aqui o resultado da questão 3.
    unique_user_id = black_friday.User_ID.nunique()
    return unique_user_id
    # pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[61]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()
    # pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[62]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]
    # pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[63]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return max(black_friday.isnull().sum())
    # pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[64]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(black_friday.Product_Category_3.mode())
    # pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[65]:


def q8():
    # Retorne aqui o resultado da questão 8.
    scale = MinMaxScaler()
    purchase_normal = scale.fit_transform(black_friday.Purchase.dropna().values.reshape(-1,1))
    return float(purchase_normal.mean())
    # pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[66]:


def q9():
    # Retorne aqui o resultado da questão 9.
    scale = StandardScaler()
    purchase_std = scale.fit_transform(black_friday.Purchase.dropna().values.reshape(-1,1))
    purchase_std_filter_count = ((purchase_std >= -1) & (purchase_std <= 1)).sum()
    return int(purchase_std_filter_count)
    # pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[67]:


def q10():
    # Retorne aqui o resultado da questão 10.
    #Se a quantidade de valores nulos em Product_Category_2 for igual a quantidade de valores nulos em (Product_Category_2 e Product_Category_3) a afirmação é verdadeira
    PC2_null_count = black_friday.Product_Category_2.isnull().sum()
    PC2_and_PC3_null_count = (black_friday.Product_Category_2.isnull() & black_friday.Product_Category_3.isnull()).sum()
    return bool(PC2_null_count == PC2_and_PC3_null_count)
    # pass

