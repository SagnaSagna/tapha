#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pyspark
pyspark.__version__


# In[70]:


import findspark
findspark.init()
findspark.find()


# In[71]:


# Import SparkSession
from pyspark.sql import SparkSession

# Create SparkSession qui permet impoter de donnés dans le memoire
spark = SparkSession.builder.appName("Adult_Data").getOrCreate() 
spark


# In[72]:


# lire les donnéés
authors = spark.read.csv('ratings.csv', sep=',', inferSchema=True, header=True)


# In[73]:


# afficher les données

authors.show()


# In[74]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


# In[75]:


df=authors.rdd.map(lambda x: [x[3],x[1]*20/100])


# In[76]:


print(df)


# In[ ]:


# Construisez le modèle de recommandation en utilisant ALS sur les données d'entraînement 
# Notez que nous définissons la stratégie de démarrage à froid sur 'drop' pour nous assurer que nous n'obtenons pas de métriques d'évaluation NaN 

als  =  ALS ( maxIter = 5 ,  regParam = 0.01 ,  userCol = "userId" ,  itemCol = "movieId" ,  ratingCol = "rating" , 
          coldStartStrategy = "drop" ) 
model  =  als . adapté (training ) 


# In[ ]:


# Évaluer le modèle en calculant la RMSE sur les données de test
prédictions  =  modèle . transform ( test ) 
evaluator  =  RegressionEvaluator ( metricName = "rmse" ,  labelCol = "rating" , 
                                predictionCol = "prediction" ) 
rmse  =  evaluator . évaluer ( prédictions ) 
imprimer ( "Erreur quadratique moyenne = "  +  str ( rmse ))


# In[ ]:


#Trouvez un exemple de code complet dans "examples/src/main/python/ml/als_example.py" dans le dépôt Spark.
#Si la matrice d'évaluation est dérivée d'une autre source d'informations (c'est-à-dire qu'elle est déduite d'autres signaux), vous pouvez définir implicitPrefssur Truepour obtenir de meilleurs résultats :

als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating")

