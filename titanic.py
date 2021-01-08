import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

path = '/home/flavios/Desktop/Projetos/Titanic-Kaggle/titanic/'
#Read dos dados em dataframes PANDA e copiando em outras variaveis
test = pd.read_csv('/home/flavios/Desktop/Projetos/Titanic-Kaggle/titanic/test.csv')
train = pd.read_csv('/home/flavios/Desktop/Projetos/Titanic-Kaggle/titanic/train.csv')
testX = test.copy()
trainX = train.copy()

#set para visualizar todas as colunas
pd.set_option('display.max_columns', None)
print("Head dados de treino:\n", trainX.head())
print("Head dados de teste:\n", testX.head())

#Drop de colunas irrelevantes.
trainX.drop(columns=['PassengerId', 'Embarked', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
testX.drop(columns=['PassengerId', 'Embarked', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Removendo dados Null do Dataset.
listNull = ['Age', 'Fare']
for i in listNull:
    trainX[i].fillna(value=train[i].median(), inplace=True)
    testX[i].fillna(value=train[i].median(), inplace=True)
#removendo dados com outliers
#cols = ['Age', 'SibSp', 'Parch', 'Fare']
#trainX[cols]= trainX[cols].clip(lower= trainX[cols].quantile(0.15), upper= trainX[cols].quantile(0.85), axis=1)
trainX.drop(columns=['Parch'], axis=1, inplace=True)

#testX[cols]= testX[cols].clip(lower= testX[cols].quantile(0.15), upper= testX[cols].quantile(0.85), axis=1)
testX.drop(columns=['Parch'], axis=1, inplace=True)

#Drop em dados desnecessários.
trainY = trainX['Survived']
trainX.drop(columns=['Survived'], axis=1, inplace=True)
trainX = pd.get_dummies(trainX, columns=['Pclass', 'Sex'], drop_first=True)
testX = pd.get_dummies(testX, columns=['Pclass', 'Sex'], drop_first=True)

#Normalizando os dados.
ss = StandardScaler()
featurestoSS = ['Age', 'SibSp', 'Fare']
trainX[featurestoSS] = ss.fit_transform(trainX[featurestoSS])
testX[featurestoSS] = ss.fit_transform(testX[featurestoSS])

#Regressão Logística
clf = LogisticRegression ()
clf.fit(trainX, trainY.ravel())
prediction = clf.predict(testX)
print('Previsão do modelo:', round(clf.score(trainX, trainY)*100,2),'%')
submission = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived': prediction})
print ('Submission head:\n', submission.head())
print("Head dados de treino:\n", trainX.head())
print("Head dados de teste:\n", testX.head())
#pd.DataFrame.to_csv(submission,path_or_buf='/home/flavios/Desktop/Projetos/Titanic-Kaggle/titanic/titanicSub.csv', index=False)
#modelo com Keras

trainX = np.array(trainX)
testX = np.array(testX)
initializer = tf.keras.initializers.random_normal(stddev=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(7, input_dim=(trainX.shape[1]), kernel_initializer= initializer, activation = 'relu'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(14, kernel_initializer= initializer, activation = 'sigmoid'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(125, kernel_initializer= initializer, activation = 'sigmoid'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(512, kernel_initializer= initializer, activation = 'sigmoid'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(1, kernel_initializer= initializer, activation = 'relu'))
model.summary()

#Compilação do modelo
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.fit(trainX,trainY, epochs=500)
predictTF = model.predict(testX)
#print(predictTF,'\n')
labelPredict = (predictTF > 0.50).astype(np.int)
#print(labelPredict)
labels = []
for x in labelPredict:
    labels = np.append(labels, x).astype(int)

print(labels)
submission2 = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived': labels})
pd.DataFrame.to_csv(submission2,path_or_buf=(path+'titanicDP.csv'), index=False)

submission['predict-Match?'] = np.where(submission['Survived'] == submission2['Survived'], 'True', 'False')
print(submission)