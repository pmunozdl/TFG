import numpy as np
import streamlit as st
#escalas
from sklearn.preprocessing import StandardScaler
#algoritmos 
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import keras
from keras.src.models.sequential import Sequential
from keras.src.layers import Dense
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing #suavizado exponencial triple, que tiene en cuenta la estacionalidad
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)

#leemos el fichero original


df = pd.read_excel("../dataset/ventasEcommerce.xlsx")
df["Order_Date"] = (pd.to_datetime(df["Order_Date"]))
df["Order_Date"] = df["Order_Date"].dt.normalize() 
df2 = df[["Order_Date","Total"]]
df_resultadoVent = df2.resample("ME",on="Order_Date").sum()
df_resultadoUni = df2.resample("ME",on="Order_Date").count()
#predecimos las ventas de 2019 usando Suavizado exponencial
modeloSE2019 = ExponentialSmoothing(df_resultadoVent[12:48],seasonal_periods=12, trend='add', seasonal='add')
ajusteSE2019 = modeloSE2019.fit()
y_predSE2019 = ajusteSE2019.forecast(steps = len(df_resultadoVent[36:48])) #predicción ingresos 2019
dfy_predSE2019 = pd.DataFrame(y_predSE2019, index= y_predSE2019.index.values,columns=["Total"])
#una vez tenemos las ventas de 2019, predecimos el número de unidades vendidas de ese año
X_train = df_resultadoUni[12:48] # tres primeros años de unidades vendidas (2016 a 2018)
Y_train = df_resultadoVent[12:48] # tres primeros años de ventas (2016 a 2018)
Y_test = dfy_predSE2019 #predicciones ventas 2019
scalerVent = StandardScaler()
scalerUni = StandardScaler()
X_trainEsc = scalerUni.fit_transform(X_train)
Y_trainEsc = scalerVent.fit_transform(Y_train)
Y_testEsc = scalerVent.transform(Y_test)
regr = linear_model.LinearRegression()
regr.fit(Y_trainEsc,X_trainEsc)
X_predRLEsc2019 = regr.predict(Y_testEsc) #prediccion de unidades vendidas 2019
X_predRL2019 = np.round(scalerUni.inverse_transform(X_predRLEsc2019)) 
X_predRL2019 = pd.DataFrame(X_predRL2019,index = y_predSE2019.index, columns=["Total"])#predicciones unidades vendidas 2019
# una vez tenemos las unidades vendidas de 2019, calculamos las unidades vendidas desglosadas por categoría. Primero transformamos los datos
df3 = df[["Order_Date","Total","Category"]].copy()
df3['Technology'] = (df3['Category'] == 'Technology').astype(int)
df3['Office Supplies'] = (df3['Category'] == 'Office Supplies').astype(int)
df3['Furniture'] = (df3['Category'] == 'Furniture').astype(int)
df3['Order_Date'] = pd.to_datetime(df3['Order_Date'])
df_resultadoVent2 = df_resultadoVent #ventas 2015-2018
df_resultadoUni2 = df3.groupby('Category').resample('M', on='Order_Date').size()
df_resultadoUni2 = df_resultadoUni2.unstack(level=0)
df_resultadoUni2['Total'] = df3.resample('M', on='Order_Date')['Total'].count()
df_resultadoUni2 = pd.DataFrame(df_resultadoUni2.values, columns = df_resultadoUni2.columns, index = df_resultadoUni2.index.values) #unidades vendidas por categoría 2015-2018
#calculamos el porcentaje de unidades vendidas por categoría
PorcentajeTecnología = round(df_resultadoUni2["Technology"] / df_resultadoUni2["Total"],3)
PorcentajeOfficeSupplies = round(df_resultadoUni2["Office Supplies"] / df_resultadoUni2["Total"],3)
PorcentajeFurniture = round(df_resultadoUni2["Furniture"] / df_resultadoUni2["Total"],3)
array_concatenado = np.column_stack((PorcentajeTecnología,PorcentajeOfficeSupplies,PorcentajeFurniture))
pdArray_concatenado = pd.DataFrame(array_concatenado,index=df_resultadoUni2.index.values,columns=["Technology","Office Supplies","Furniture"])
#preparamos los datos
pdArray_concatenadoTrain = pdArray_concatenado[12:48] #porcentaje categorías 2016-2018
df_resultadoVent2 = df_resultadoVent2[12:48] #ingresos mensuales 2016-2018
#predecimos los porcentajes de unidades vendidas por categoría, 
scalerUniCat = StandardScaler()
UniCatEsc = scalerUniCat.fit_transform(pdArray_concatenadoTrain)
regr = linear_model.LinearRegression()
regr.fit(X_trainEsc,UniCatEsc) 
X_predRLCatEsc = regr.predict(X_predRLEsc2019) #porcentajes por categoría 2019
#una vez tenemos los porcentajes, y las ventas totales, ponderamos para saber el valor exacto
X_predRLCat = scalerUniCat.inverse_transform(X_predRLCatEsc)
X_predRLCat = pd.DataFrame(X_predRLCat,index = y_predSE2019.index,columns = pdArray_concatenado.columns)
X_predRLCat = X_predRLCat.round(3)
prediccionUnidades2019Cat = np.round(X_predRL2019.values * X_predRLCat.values)
prediccionUnidades2019Cat = pd.DataFrame(prediccionUnidades2019Cat,index = y_predSE2019.index, columns = pdArray_concatenado.columns)
#Una vez tenemos las unidades vendidas por categoría de 2019, calculamos las ventas por categoría de 2019
#primero, preparamos los datos
df_resultadoVent2 = df3.groupby('Category').resample("ME",on="Order_Date").sum() #ingresos de entrenamiento
df_resultadoVent2 = df_resultadoVent2[["Total"]]
df_resultadoVent2 = df_resultadoVent2.unstack(level=0)
df_resultadoVent2.columns = df_resultadoVent2.columns.droplevel(0)
df_resultadoVent2 = df_resultadoVent2[["Technology","Office Supplies","Furniture"]]
df_resultadoVent2["Total"] = df_resultadoVent
#calculamos porcentajes
PorcentajeTecnología2 = round(df_resultadoVent2["Technology"] / df_resultadoVent2["Total"],4)
PorcentajeOfficeSupplies2 = round(df_resultadoVent2["Office Supplies"] / df_resultadoVent2["Total"],4)
PorcentajeFurniture2 = round(df_resultadoVent2["Furniture"] / df_resultadoVent2["Total"],4)
#calcular a partir del porcentaje y luego ponderar
array_concatenado2 = np.column_stack((PorcentajeTecnología2,PorcentajeOfficeSupplies2,PorcentajeFurniture2))
pdArray_concatenado2 = pd.DataFrame(array_concatenado2,index=df_resultadoVent2.index.values,columns=["Technology","Office Supplies","Furniture"])
#preparamos los datos. Necesitamos Ventas porcentaje entrenamiento y unidades vendidas todo
pdArray_concatenado2Train = pdArray_concatenado2[12:48] #porcentajes ventas 2016-2018
UniCatTrain = df_resultadoUni2[["Technology","Office Supplies","Furniture"]][12:48] #unidades vendidas por categoría 2016 a 2018
UniCatTest = prediccionUnidades2019Cat #unidades vendidas por categoría 2019 (predicción anterior)
scalerVentCat = StandardScaler() #escalador ventas por categoría porcentaje
scalerUniCat2 = StandardScaler() #escalador unidades vendidas por categoría, esta vez como valor entero y no porcentaje
scalerVentCat2 = StandardScaler() #escalador ventas por categoría, esta vez como valor entero y no como porcentaje
#escalamos los datos
UniCatTrainEsc = scalerUniCat2.fit_transform(UniCatTrain)
UniCatTestEsc =  scalerUniCat2.transform(UniCatTest)
pdArray_concatenado2TrainEsc = scalerVentCat.fit_transform(pdArray_concatenado2Train)
#predecimos los ingresos por categoría de 2019
keras.utils.set_random_seed(1)
def prediccionVentasCat():
    modelRNVentCat2019 = Sequential()
    modelRNVentCat2019.add(Dense(64, input_shape=(3,), activation='relu'))  # Capa oculta con 64 neuronas, tres entradas y función de activación ReLU
    modelRNVentCat2019.add(Dense(3))  # Capa de salida con una neurona (predicción de ingresos)
    modelRNVentCat2019.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
    modelRNVentCat2019.fit(UniCatTrainEsc, pdArray_concatenado2TrainEsc, epochs=100, batch_size=32, verbose = 0)
    predVentCat2019RNEsc = modelRNVentCat2019.predict(UniCatTestEsc,verbose=0)
    return predVentCat2019RNEsc
predVentCat2019RN = scalerVentCat.inverse_transform(prediccionVentasCat())
predVentCat2019RN = pd.DataFrame(predVentCat2019RN,columns=X_predRLCat.columns, index = X_predRLCat.index.values)
predVentCat2019RN = predVentCat2019RN
#una vez tenemos los porcentajes predichos, los multiplicamos por los ingresos de 2019 para desglosar los ingresos por categoría
predVentCat2019 = np.round(dfy_predSE2019.values * predVentCat2019RN.values)
predVentCat2019 = pd.DataFrame(predVentCat2019,columns = prediccionUnidades2019Cat.columns, index = prediccionUnidades2019Cat.index.values)
predVentCat2019["Total"] = dfy_predSE2019
prediccionUnidades2019Cat["Total"] = X_predRL2019
#una vez tenemos las ventas por categoría, calculamos los ingresos por categoría

#obtener ingresos totales 5 años, ventas totales 5 años, ventas categoría 5 años, e ingresos por categoría 5 años
#ventas por categoría 5 años+
def VentasCategoria():
        VentCatFinal = pd.concat([df_resultadoVent2, predVentCat2019], axis=0)
        return VentCatFinal
#Unidades vendidas por categoría 5 años
def UnidadesCategoria():
    UniCatFinal = pd.concat([df_resultadoUni2.astype(int), prediccionUnidades2019Cat.astype(int)], axis=0)
    return UniCatFinal

def leerExcel ():
      return df

def Pedidos ():
    ventas_por_pedido = df.groupby('Order_ID')['Total'].sum().reset_index()
    primer_nombre_cliente = df.groupby('Order_ID')['Customer_ID'].first().reset_index()
    pedidos = pd.merge(ventas_por_pedido, primer_nombre_cliente, on='Order_ID')
    return pedidos
#Ventas totales durante los 5 años
#VentTotFinal = pd.concat([df_resultadoVent, dfy_predSE2019])
#Unidades vendidas totales 
#UniTotFinal = pd.concat([df_resultadoUni.astype(int), X_predRL2019.astype(int)])