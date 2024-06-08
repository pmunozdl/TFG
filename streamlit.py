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

#leemos el fichero original
#df = pd.read_excel("/Users/pablomunozdelorenzo/Desktop/ventasEcommerce.xlsx")
df = pd.read_excel("dataset/ventasEcommerce.xlsx")
df["Order_Date"] = (pd.to_datetime(df["Order_Date"]))
df["Order_Date"] = df["Order_Date"].dt.normalize() 
df2 = df[["Order_Date","Total"]]
df_resultadoIng = df2.resample("ME",on="Order_Date").sum()
df_resultadoVent = df2.resample("ME",on="Order_Date").count()
#predecimos los ingresos de 2019 usando Suavizado exponencial
modeloSE2019 = ExponentialSmoothing(df_resultadoIng[12:48],seasonal_periods=12, trend='add', seasonal='add')
ajusteSE2019 = modeloSE2019.fit()
y_predSE2019 = ajusteSE2019.forecast(steps = len(df_resultadoIng[36:48])) #predicción ingresos 2019
dfy_predSE2019 = pd.DataFrame(y_predSE2019,columns = ["Total"])
#una vez tenemos los ingresos de 2019, predecimos el número de ventas de ese año
X_train = df_resultadoVent[12:48] # tres primeros años de ventas (2016 a 2018)
Y_train = df_resultadoIng[12:48] # tres primeros años de ingresos (2016 a 2018)
Y_test = dfy_predSE2019 #predicciones 2019
scalerIng = StandardScaler()
scalerVent = StandardScaler()
X_trainEsc = scalerVent.fit_transform(X_train)
Y_trainEsc = scalerIng.fit_transform(Y_train)
Y_testEsc = scalerIng.transform(Y_test)
regr = linear_model.LinearRegression()
regr.fit(Y_trainEsc,X_trainEsc)
X_predRLEsc2019 = regr.predict(Y_testEsc)
X_predRL2019 = np.round(scalerVent.inverse_transform(X_predRLEsc2019)) 
X_predRL2019 = pd.DataFrame(X_predRL2019,index = y_predSE2019.index, columns=["Total"])#predicciones ventas 2019
# una vez tenemos las ventas de 2019, calculamos por categoría. Primero transformamos los datos
df3 = df[["Order_Date","Total","Category"]]
df3['Technology'] = (df3['Category'] == 'Technology').astype(int)
df3['Office Supplies'] = (df3['Category'] == 'Office Supplies').astype(int)
df3['Furniture'] = (df3['Category'] == 'Furniture').astype(int)
df_resultadoIng2 = df3.resample("ME",on="Order_Date").sum() #ingresos de entrenamiento
df_resultadoVent2 = df3.resample("ME",on="Order_Date").count() #ventas de entrenamiento
df_resultadoVent2 = df_resultadoIng2[["Technology","Office Supplies","Furniture"]]
df_resultadoVent2["Ventas Totales"] = df_resultadoVent2["Technology"] + df_resultadoVent2["Office Supplies"] + df_resultadoVent2["Furniture"]
#calculamos el porcentaje de ventas 
PorcentajeTecnología = df_resultadoVent2["Technology"] / df_resultadoVent2["Ventas Totales"]
PorcentajeOfficeSupplies = df_resultadoVent2["Office Supplies"] / df_resultadoVent2["Ventas Totales"]
PorcentajeFurniture = df_resultadoVent2["Furniture"] / df_resultadoVent2["Ventas Totales"]
array_concatenado = np.column_stack((PorcentajeTecnología,PorcentajeOfficeSupplies,PorcentajeFurniture))
pdArray_concatenado = pd.DataFrame(array_concatenado,index=df_resultadoVent2.index.values,columns=["Technology","Office Supplies","Furniture"])
#una vez tenemos las ventas de 2019, calculamos las ventas de cada categoría
pdArray_concatenadoTrain = pdArray_concatenado[12:48] #porcentaje categorías 2016-2018
df_resultadoIng2 = df_resultadoIng[12:48] #ingresos mensuales 2016-2018
#predecimos las ventas por categoría, 
scalerVentasCat = StandardScaler()
VentasCatEsc = scalerVentasCat.fit_transform(pdArray_concatenadoTrain)
regr = linear_model.LinearRegression()
regr.fit(X_trainEsc,VentasCatEsc)
X_predRLCatEsc = regr.predict(X_predRLEsc2019) #porcentajes por categoría 2019
#una vez tenemos los porcentajes, y las ventas totales, ponderamos para saber el valor exacto
X_predRLCat = scalerVentasCat.inverse_transform(X_predRLCatEsc)
X_predRLCat = pd.DataFrame(X_predRLCat,index = y_predSE2019.index,columns = pdArray_concatenado.columns)
prediccionVentas2019Cat = np.round(X_predRL2019.values * X_predRLCat.values)
prediccionVentas2019Cat = pd.DataFrame(prediccionVentas2019Cat,index = y_predSE2019.index, columns = pdArray_concatenado.columns)
#Una vez tenemos las ventas por categoría de 2019, calculamos los ingresos de 2019
df_resultadoIng2 = df3.groupby('Category').resample("ME",on="Order_Date").sum() #ingresos de entrenamiento
df_resultadoIng2 = df_resultadoIng2[["Total"]]
df_resultadoIng2 = df_resultadoIng2.unstack(level=0)
df_resultadoIng2.columns = df_resultadoIng2.columns.droplevel(0)
df_resultadoIng2 = df_resultadoIng2[["Technology","Office Supplies","Furniture"]]
scalerIngCat = StandardScaler()
scalerVentasCat2 = StandardScaler() #ventas sin porcentaje
VenCatTrain = df_resultadoVent2[["Technology","Office Supplies","Furniture"]][12:48] #ventas 2016 a 2018
VenCatTest = prediccionVentas2019Cat #ventas categoría 2019
IngCatTrain = df_resultadoIng2[12:48] #ingresos de 2016 a 2018
#escalamos los datos
VenCatTrainEsc = scalerVentasCat2.fit_transform(VenCatTrain)
VenCatTestEsc =  scalerVentasCat2.transform(VenCatTest)
IngCatTrainEsc = scalerIngCat.fit_transform(IngCatTrain)
#predecimos los ingresos por categoría de 2019
@st.cache_data()
def prediccionIng2019():
    modelRNIngCat2019 = Sequential()
    modelRNIngCat2019.add(Dense(64, input_dim=3, activation='relu'))  # Capa oculta con 64 neuronas y función de activación ReLU
    modelRNIngCat2019.add(Dense(3))  # Capa de salida con una neurona (predicción de ingresos)
    modelRNIngCat2019.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
    modelRNIngCat2019.fit(VenCatTrainEsc, IngCatTrainEsc, epochs=100, batch_size=32)
    predIngCat2019RNEsc = modelRNIngCat2019.predict(VenCatTestEsc)
    return predIngCat2019RNEsc
predIngCat2019RN = scalerIngCat.inverse_transform(prediccionIng2019())
predIngCat2019 = pd.DataFrame(predIngCat2019RN,columns = prediccionVentas2019Cat.columns, index = prediccionVentas2019Cat.index.values)
#obtener ingresos totales 5 años, ventas totales 5 años, ventas categoría 5 años, e ingresos por categoría 5 años
#ingresos por categoría 5 años
IngCatFinal = pd.concat([df_resultadoIng2, predIngCat2019], axis=0)
VentCatFinal = pd.concat([df_resultadoVent2[["Technology","Office Supplies","Furniture"]], prediccionVentas2019Cat], axis=0)
IngTotFinal = pd.concat([df_resultadoIng, dfy_predSE2019])
VentTotFinal = pd.concat([df_resultadoVent, X_predRL2019])
IngCatFinal["Total"] = IngTotFinal
VentCatFinal["Total"] = VentTotFinal


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
meses_disponibles = ['Todos'] + list(np.unique(VentCatFinal.index.month))
meses_disponiblesI = ['Todos'] + list(np.unique(IngCatFinal.index.month))
# Mostrar los gráficos en las columnas
with colMes1:
    st.write("Ventas")
    mes_seleccionado = st.selectbox("Mes Venta", meses_disponibles)
with colMes2:
    st.write("Ingresos")
    mes_seleccionadoI = st.selectbox("Mes Ingreso", meses_disponiblesI)
#para filtras por meses las ventas
if mes_seleccionado != 'Todos':
    df_filtered = VentCatFinal[VentCatFinal.index.month == mes_seleccionado]
else:
    df_filtered = VentCatFinal
#para filtrar por meses los ingresos
if mes_seleccionadoI != 'Todos':
    df_filteredI = IngCatFinal[IngCatFinal.index.month == mes_seleccionadoI]
else:
    df_filteredI = IngCatFinal

#a continuación, filtramos por el año
años_disponibles = ['Todos'] + list(np.unique(df_filtered.index.year))
años_disponiblesI = ['Todos'] + list(np.unique(df_filteredI.index.year))

with colMes1: #reutilizamos las columnas anteriores
    año_seleccionado = st.selectbox("Año Venta", años_disponibles)

with colMes2: #reutilizamos las columnas anteriores
    año_seleccionadoI = st.selectbox("Año Ingreso", años_disponiblesI)
#filtrar por años las ventas
if año_seleccionado != 'Todos':
    df_filtered2 = df_filtered[df_filtered.index.year == año_seleccionado]
else:
    df_filtered2 = df_filtered
#para filtrar por meses los ingresos
if año_seleccionadoI != 'Todos':
    df_filteredI2 = df_filteredI[df_filteredI.index.year == año_seleccionadoI]
else:
    df_filteredI2 = df_filteredI


# Filtro por valor mínimo de ventas
min_sales = st.slider("Filtrar por ventas mínimas", min_value=(df_filtered2["Total"].min()+1), max_value=(df_filtered2["Total"].max()-1), value=df_filtered2["Total"].min())
df_filtered3 = df_filtered2[df_filtered2["Total"] >= min_sales].dropna()
# Filtro por valor máximo de ventas
max_sales = st.slider("Filtrar por ventas máximas", min_value=(df_filtered3["Total"].min()+1), max_value=(df_filtered3["Total"].max()-1), value=df_filtered3["Total"].max())
df_filtered4 = df_filtered3[df_filtered3["Total"] <= max_sales].dropna()
# Filtro por valor mínimo de ingresos
min_amount = st.slider("Filtrar por importe mínimo", min_value=(df_filteredI2["Total"].min()+1), max_value=(df_filteredI2["Total"].max()-1), value=df_filteredI2["Total"].min())
df_filteredI3 = df_filteredI2[df_filteredI2["Total"] >= min_amount].dropna()
# Filtro por valor máximo de ingresos
max_amount = st.slider("Filtrar por importe máximo", min_value=(df_filteredI3["Total"].min()+1), max_value=(df_filteredI3["Total"].max()-1), value=df_filteredI3["Total"].max())
df_filteredI4 = df_filteredI3[df_filteredI3["Total"] <= max_amount].dropna()

# Aplicar el filtro



st.header("Ventas por categoría")
st.dataframe(df_filtered4[["Technology","Office Supplies","Furniture"]])


st.header("Ingresos por categoría")
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
mediaVentasCat = df_filtered4[["Technology","Office Supplies","Furniture"]].mean().to_numpy()
fig1, ax = plt.subplots()
ax.pie(mediaVentasCat, labels=df_resultadoIng2.columns, colors = colors,autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Para asegurar que el gráfico sea un círculo

#segunda figura
mediaIngCat = df_filteredI4[["Technology","Office Supplies","Furniture"]].mean().to_numpy()
fig2, ax2 = plt.subplots()
ax2.pie(mediaIngCat, labels=df_resultadoIng2.columns, colors = colors,
      autopct='%1.1f%%', startangle=140)

ax2.axis('equal')  # Para asegurar que el gráfico sea un círculo
# Mostrar el gráfico en Streamlit
col1, col2 = st.columns(2)

# Mostrar los gráficos en las columnas
with col1:
    st.write("Ventas Por Categoría")
    st.pyplot(fig1)

with col2:
    st.write("Ingresos Por Categoría")
    st.pyplot(fig2)

###añadir gráfico de columnas de los 5 años. 
st.write("Ventas totales")
st.line_chart(df_filtered2[["Total"]])
st.write("Ingresos totales")
st.line_chart(df_filteredI2[["Total"]])
