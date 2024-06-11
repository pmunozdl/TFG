# 'dataset' contiene los datos de entrada para este script
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import  linear_model
from keras.src.models.sequential import Sequential
from keras.src.layers import Dense
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

df = pd.read_excel("../Desktop/TFG/Cuadro de Mando/ventasEcommerce.xlsx")
df["Order_Date"] = (pd.to_datetime(df["Order_Date"]))
df["Order_Date"] = df["Order_Date"].dt.normalize() 
df2 = df[["Order_Date","Total"]]
df_resultadoIng = df2.resample("ME",on="Order_Date").sum()
df_resultadoVent = df2.resample("ME",on="Order_Date").count()
modeloSE2019 = ExponentialSmoothing(df_resultadoIng[12:48],seasonal_periods=12, trend='add', seasonal='add')
ajusteSE2019 = modeloSE2019.fit()
y_predSE2019 = ajusteSE2019.forecast(steps = len(df_resultadoIng[36:48])) 
dfy_predSE2019 = pd.DataFrame(y_predSE2019,columns = ["Total"])
X_train = df_resultadoVent[12:48] 
Y_train = df_resultadoIng[12:48]
Y_test = dfy_predSE2019
scalerIng = StandardScaler()
scalerVent = StandardScaler()
X_trainEsc = scalerVent.fit_transform(X_train)
Y_trainEsc = scalerIng.fit_transform(Y_train)
Y_testEsc = scalerIng.transform(Y_test)
regr = linear_model.LinearRegression()
regr.fit(Y_trainEsc,X_trainEsc)
X_predRLEsc2019 = regr.predict(Y_testEsc)
X_predRL2019 = np.round(scalerVent.inverse_transform(X_predRLEsc2019)) 
X_predRL2019 = pd.DataFrame(X_predRL2019,index = y_predSE2019.index, columns=["Total"])
df3 = df[["Order_Date","Total","Category"]]
df3['Technology'] = (df3['Category'] == 'Technology').astype(int)
df3['Office Supplies'] = (df3['Category'] == 'Office Supplies').astype(int)
df3['Furniture'] = (df3['Category'] == 'Furniture').astype(int)
df_resultadoIng2 = df3.resample("ME",on="Order_Date").sum() 
df_resultadoVent2 = df3.resample("ME",on="Order_Date").count()
df_resultadoVent2 = df_resultadoIng2[["Technology","Office Supplies","Furniture"]]
df_resultadoVent2["Ventas Totales"] = df_resultadoVent2["Technology"] + df_resultadoVent2["Office Supplies"] + df_resultadoVent2["Furniture"]
PorcentajeTecnología = df_resultadoVent2["Technology"] / df_resultadoVent2["Ventas Totales"]
PorcentajeOfficeSupplies = df_resultadoVent2["Office Supplies"] / df_resultadoVent2["Ventas Totales"]
PorcentajeFurniture = df_resultadoVent2["Furniture"] / df_resultadoVent2["Ventas Totales"]
array_concatenado = np.column_stack((PorcentajeTecnología,PorcentajeOfficeSupplies,PorcentajeFurniture))
pdArray_concatenado = pd.DataFrame(array_concatenado,index=df_resultadoVent2.index.values,columns=["Technology","Office Supplies","Furniture"])
pdArray_concatenadoTrain = pdArray_concatenado[12:48] 
df_resultadoIng2 = df_resultadoIng[12:48] 
scalerVentasCat = StandardScaler()
VentasCatEsc = scalerVentasCat.fit_transform(pdArray_concatenadoTrain)
regr = linear_model.LinearRegression()
regr.fit(X_trainEsc,VentasCatEsc)
X_predRLCatEsc = regr.predict(X_predRLEsc2019) 
X_predRLCat = scalerVentasCat.inverse_transform(X_predRLCatEsc)
X_predRLCat = pd.DataFrame(X_predRLCat,index = y_predSE2019.index,columns = pdArray_concatenado.columns)
prediccionVentas2019Cat = np.round(X_predRL2019.values * X_predRLCat.values)
prediccionVentas2019Cat = pd.DataFrame(prediccionVentas2019Cat,index = y_predSE2019.index, columns = pdArray_concatenado.columns)
df_resultadoIng2 = df3.groupby('Category').resample("ME",on="Order_Date").sum() 
df_resultadoIng2 = df_resultadoIng2[["Total"]]
df_resultadoIng2 = df_resultadoIng2.unstack(level=0)
df_resultadoIng2.columns = df_resultadoIng2.columns.droplevel(0)
df_resultadoIng2 = df_resultadoIng2[["Technology","Office Supplies","Furniture"]]
scalerIngCat = StandardScaler()
scalerVentasCat2 = StandardScaler() 
VenCatTrain = df_resultadoVent2[["Technology","Office Supplies","Furniture"]][12:48] 
VenCatTest = prediccionVentas2019Cat 
IngCatTrain = df_resultadoIng2[12:48] 
VenCatTrainEsc = scalerVentasCat2.fit_transform(VenCatTrain)
VenCatTestEsc =  scalerVentasCat2.transform(VenCatTest)
IngCatTrainEsc = scalerIngCat.fit_transform(IngCatTrain)
def prediccionIng2019():
    modelRNIngCat2019 = Sequential()
    modelRNIngCat2019.add(Dense(64, input_dim=3, activation='relu')) 
    modelRNIngCat2019.add(Dense(3))  
    modelRNIngCat2019.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
    modelRNIngCat2019.fit(VenCatTrainEsc, IngCatTrainEsc, epochs=100, batch_size=32)
    predIngCat2019RNEsc = modelRNIngCat2019.predict(VenCatTestEsc)
    return predIngCat2019RNEsc
predIngCat2019RN = scalerIngCat.inverse_transform(prediccionIng2019())
predIngCat2019 = pd.DataFrame(predIngCat2019RN,columns = prediccionVentas2019Cat.columns, index = prediccionVentas2019Cat.index.values)
IngCatFinal = pd.concat([df_resultadoIng2, predIngCat2019], axis=0)
VentCatFinal = pd.concat([df_resultadoVent2[["Technology","Office Supplies","Furniture"]], prediccionVentas2019Cat], axis=0)
IngTotFinal = pd.concat([df_resultadoIng, dfy_predSE2019])
VentTotFinal = pd.concat([df_resultadoVent, X_predRL2019])
dataset[VentasTotales] = VentTotFinal
dataset[VentasTecnologia] = VentCatFinal[["Technology"]]
dataset[VentasMaterialOficina] = VentCatFinal[["Office Supplies"]]
dataset[VentasHerramientas] = VentCatFinal[["Furniture"]]
dataset[IngresosTotales] = IngTotFinal
dataset[IngresosTecnologia] = IngCatFinal[["Technology"]]
dataset[IngresosMaterialOficina] = IngCatFinal[["Office Supplies"]]
dataset[IngresosHerramientas] = IngCatFinal[["Furniture"]]