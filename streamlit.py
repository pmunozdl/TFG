import streamlit as st
import numpy as np
#escalas
from sklearn.preprocessing import StandardScaler
#algoritmos 
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

#from tensorflow import keras
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Sequential

from keras.src.models.sequential import Sequential
from keras.src.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing #suavizado exponencial triple, que tiene en cuenta la estacionalidad
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)

#leemos el fichero original
#df = pd.read_excel("/Users/pablomunozdelorenzo/Desktop/ventasEcommerce.xlsx")
df = pd.read_excel("dataset/ventasEcommerce.xlsx")
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
PorcentajeTecnología = df_resultadoUni2["Technology"] / df_resultadoUni2["Total"]
PorcentajeOfficeSupplies = df_resultadoUni2["Office Supplies"] / df_resultadoUni2["Total"]
PorcentajeFurniture = df_resultadoUni2["Furniture"] / df_resultadoUni2["Total"]
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
PorcentajeTecnología2 = df_resultadoVent2["Technology"] / df_resultadoVent2["Total"]
PorcentajeOfficeSupplies2 = df_resultadoVent2["Office Supplies"] / df_resultadoVent2["Total"]
PorcentajeFurniture2 = df_resultadoVent2["Furniture"] / df_resultadoVent2["Total"]
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
@st.cache_data()
def prediccionVentCat2019():
    modelRNVentCat2019 = Sequential()
    modelRNVentCat2019.add(Dense(64, input_shape=(3,), activation='relu'))  # Capa oculta con 64 neuronas, tres entradas y función de activación ReLU
    modelRNVentCat2019.add(Dense(3))  # Capa de salida con una neurona (predicción de ingresos)
    modelRNVentCat2019.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
    modelRNVentCat2019.fit(UniCatTrainEsc, pdArray_concatenado2TrainEsc, epochs=100, batch_size=32)
    predVentCat2019RNEsc = modelRNVentCat2019.predict(UniCatTestEsc)
    return predVentCat2019RNEsc
predVentCat2019RN = scalerVentCat.inverse_transform(prediccionVentCat2019())
predVentCat2019RN = pd.DataFrame(predVentCat2019RN,columns=X_predRLCat.columns, index = X_predRLCat.index.values)
#una vez tenemos los porcentajes predichos, los multiplicamos por los ingresos de 2019 para desglosar los ingresos por categoría
predVentCat2019 = np.round(dfy_predSE2019.values * predVentCat2019RN.values)
predVentCat2019 = pd.DataFrame(predVentCat2019,columns = prediccionUnidades2019Cat.columns, index = prediccionUnidades2019Cat.index.values)
predVentCat2019["Total"] = dfy_predSE2019
prediccionUnidades2019Cat["Total"] = X_predRL2019
#predecimos los ingresos por categoría de 2019

#obtener ingresos totales 5 años, ventas totales 5 años, ventas categoría 5 años, e ingresos por categoría 5 años
#ventas por categoría 5 años
VentCatFinal = pd.concat([df_resultadoVent2, predVentCat2019], axis=0)
#Unidades vendidas por categoría 5 años
UniCatFinal = pd.concat([df_resultadoUni2.astype(int), prediccionUnidades2019Cat.astype(int)], axis=0)
#Ventas totales durante los 5 alos
VentTotFinal = pd.concat([df_resultadoVent, dfy_predSE2019])
#Unidades vendidas totales 
UniTotFinal = pd.concat([df_resultadoUni.astype(int), X_predRL2019.astype(int)])

# Título de la aplicación
st.title("Clasificación de ventas según criterio ABC")
st.markdown("""
El sistema de clasificación ABC permite segmentar y organizar los productos de un almacén en base a su importancia.
Pretende priorizar las mercancías de un almacén más importantes 

- **Categoría A**: Los productos pertenecientes a la categoría A son los más importantes para la empresa. Son solo en torno a un 20% del inventario pero suponen la mayoría del movimiento habitual de almacén, con mayor rotación y también los que aportan en torno al 80% de los ingresos 
- **Categoría B**: Los productos pertenecientes a la categoría B tienen una importancia y rotación moderada para la empresa. Generalmente suponen en torno al 30% del total de productos del almacén, y por norma, no suelen generar más del 20% de los ingresos de la empresa.
- **Categoría C**: Los productos de la categoría C serán los más numerosos, pero también las que menos ingresos aportan a la empresa. Pueden suponer más del 50% de las referencias de productos pero en términos de ingresos no alcanzar ni el 5% del total.
""")


# Mostrar los DataFrames
colMes1, colMes2 = st.columns(2)
meses_disponibles = ['Todos'] + list(np.unique(UniCatFinal.index.month))
meses_disponiblesI = ['Todos'] + list(np.unique(VentCatFinal.index.month))
# Mostrar los gráficos en las columnas
with colMes1:
    st.write("Unidades Vendidas")
    mes_seleccionado = st.selectbox("Mes filtro Unidades Vendidas", meses_disponibles)
with colMes2:
    st.write("Ventas")
    mes_seleccionadoI = st.selectbox("Mes filtro Ventas", meses_disponiblesI)
#para filtras por meses las unidades vendidas
if mes_seleccionado != 'Todos':
    df_filtered = UniCatFinal[UniCatFinal.index.month == mes_seleccionado]
else:
    df_filtered = UniCatFinal
#para filtrar por meses los ventas
if mes_seleccionadoI != 'Todos':
    df_filteredI = VentCatFinal[VentCatFinal.index.month == mes_seleccionadoI]
else:
    df_filteredI = VentCatFinal

#a continuación, filtramos por el año
años_disponibles = ['Todos'] + list(np.unique(df_filtered.index.year))
años_disponiblesI = ['Todos'] + list(np.unique(df_filteredI.index.year))

with colMes1: #reutilizamos las columnas anteriores
    año_seleccionado = st.selectbox("Año filtro Unidad vendida", años_disponibles)

with colMes2: #reutilizamos las columnas anteriores
    año_seleccionadoI = st.selectbox("Año filtro Venta", años_disponiblesI)
#filtrar por años las unidades vendidas
if año_seleccionado != 'Todos':
    df_filtered2 = df_filtered[df_filtered.index.year == año_seleccionado]
else:
    df_filtered2 = df_filtered
#para filtrar por meses los ventas
if año_seleccionadoI != 'Todos':
    df_filteredI2 = df_filteredI[df_filteredI.index.year == año_seleccionadoI]
else:
    df_filteredI2 = df_filteredI

# Filtro por valor mínimo de unidades vendidas
min_sales = st.slider("Filtrar por unidades vendidas mínimas", min_value=(df_filtered2["Total"].min()+1), max_value=(df_filtered2["Total"].max()-1), value=df_filtered2["Total"].min())
df_filtered3 = df_filtered2[df_filtered2["Total"] >= min_sales].dropna()
# Filtro por valor máximo de unidades vendidas
max_sales = st.slider("Filtrar por unidades vendidas máximas", min_value=(df_filtered3["Total"].min()+1), max_value=(df_filtered3["Total"].max()-1), value=df_filtered3["Total"].max())
df_filtered4 = df_filtered3[df_filtered3["Total"] <= max_sales].dropna()
# Filtro por valor mínimo de ventas
min_amount = st.slider("Filtrar por importe de ventas mínimo", min_value=(df_filteredI2["Total"].min()+1), max_value=(df_filteredI2["Total"].max()-1), value=df_filteredI2["Total"].min())
df_filteredI3 = df_filteredI2[df_filteredI2["Total"] >= min_amount].dropna()
# Filtro por valor máximo de ingresos
max_amount = st.slider("Filtrar por importe de ventas máximo", min_value=(df_filteredI3["Total"].min()+1), max_value=(df_filteredI3["Total"].max()-1), value=df_filteredI3["Total"].max())
df_filteredI4 = df_filteredI3[df_filteredI3["Total"] <= max_amount].dropna()

# Aplicar el filtro

st.header("Unidades vendidas por categoría")
st.dataframe(df_filtered4[["Technology","Office Supplies","Furniture"]])


st.header("Ventas por categoría")
st.dataframe(df_filteredI4[["Technology","Office Supplies","Furniture"]])
# Mostrar el DataFrame resultante


##repetimos el código para los filtros por ingresos

# Aplicar el filtro

# Mostrar el DataFrame resultante


# Grafico de barras
st.bar_chart(df_filtered4[["Technology","Office Supplies","Furniture"]])
st.bar_chart(df_filteredI4[["Technology","Office Supplies","Furniture"]])

#gráfico circular
colors = ['lightblue', 'green', 'red']
mediaUnidadesCat = df_filtered4[["Technology","Office Supplies","Furniture"]].mean().to_numpy()
fig1, ax = plt.subplots()
ax.pie(mediaUnidadesCat, labels=["Technology","Office Supplies","Furniture"], colors = colors,autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Para asegurar que el gráfico sea un círculo

#segunda figura
mediaVentCat = df_filteredI4[["Technology","Office Supplies","Furniture"]].mean().to_numpy()
fig2, ax2 = plt.subplots()
ax2.pie(mediaVentCat, labels=["Technology","Office Supplies","Furniture"], colors = colors,
      autopct='%1.1f%%', startangle=140)

ax2.axis('equal')  # Para asegurar que el gráfico sea un círculo
# Mostrar el gráfico en Streamlit
col1, col2 = st.columns(2)

# Mostrar los gráficos en las columnas
with col1:
    st.write("Unidades vendidas Por Categoría")
    st.pyplot(fig1)

with col2:
    st.write("Ventas Por Categoría")
    st.pyplot(fig2)

###añadir gráfico de columnas de los 5 años. 
st.write("Unidades vendidas totales")
st.line_chart(df_filtered2[["Total"]])
st.write("Ventas totales")
st.line_chart(df_filteredI2[["Total"]])
