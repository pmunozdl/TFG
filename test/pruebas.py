import unittest
import numpy as np

from predicciones import VentasCategoria, UnidadesCategoria, leerExcel

class TestPredicciones(unittest.TestCase):
    def test_ventasAnualesTotales(self): #TC1
        predicciones = VentasCategoria()
        predicciones = predicciones["Total"].sum()
        self.assertEqual(round(predicciones,2), 3)
    
    def test_ventasAnuales2015(self): #TC2
        predicciones = VentasCategoria()
        predicciones2015 = predicciones[predicciones.index.year == 2015]
        predicciones2015 = predicciones2015["Total"].sum()
        self.assertEqual(round(predicciones2015/1000,2),722.05)

    def test_ventasAnuales2016(self): #TC3
        predicciones = VentasCategoria()
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        predicciones2016 = predicciones2016["Total"].sum()
        self.assertEqual(round(predicciones2016/1000,2),722.05)
    
    def test_ventasAnuales2017(self): #TC4
        predicciones = VentasCategoria()
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        predicciones2017 = predicciones2017["Total"].sum()
        self.assertEqual(round(predicciones2017/1000,2),722.05)

    def test_ventasAnuales2018(self): #TC5
        predicciones = VentasCategoria()
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        predicciones2018 = predicciones2018["Total"].sum()
        self.assertEqual(round(predicciones2018/1000,2),722.05)
    
    def test_ventasAnuales2019(self): #TC6
        predicciones = VentasCategoria()
        predicciones2019 = predicciones[predicciones.index.year == 2019]
        predicciones2019 = predicciones2019["Total"].sum()
        self.assertEqual(round(predicciones2019/1000,2),722.05)
    
    def test_UnidadesVendidasTotales(self): #TC7
        predicciones = UnidadesCategoria()
        predicciones = predicciones["Total"].sum()
        self.assertEqual(predicciones,722.05)

    def test_UnidadesVendidas2015(self): #TC8
        predicciones = UnidadesCategoria()
        predicciones2015 = predicciones[predicciones.index.year == 2015]
        predicciones2015 = predicciones2015["Total"].sum()
        self.assertEqual(predicciones2015,722.05)
    
    def test_Unidadesanuales2016(self): #TC9
        predicciones = UnidadesCategoria()
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        predicciones2016 = predicciones2016["Total"].sum()
        self.assertEqual(predicciones2016,722.05)

    def test_Unidadesanuales2017(self): #TC10
        predicciones = UnidadesCategoria()
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        predicciones2017 = predicciones2017["Total"].sum()
        self.assertEqual(predicciones2017,722.05)
    
    def test_Unidadesanuales2018(self): #TC11
        predicciones = UnidadesCategoria()
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        predicciones2018 = predicciones2018["Total"].sum()
        self.assertEqual(predicciones2018,722.05)
    
    def test_Unidadesanuales2019(self): #TC12
        predicciones = UnidadesCategoria()
        predicciones2019 = predicciones[predicciones.index.year == 2019]
        predicciones2019 = predicciones2019["Total"].sum()
        self.assertEqual(predicciones2019,722.05)
    
    def test_PedidosTotales(self): #TC13
        df = leerExcel()
        numeroPedidos = df["Order_ID"].nunique()
        self.assertEqual(numeroPedidos,722.05)

    def test_Pedidos2015(self): #TC14
        df = leerExcel()
        numeroPedidos2015 = df[df["Order_Date"].dt.year == 2015]
        numeroPedidos2015 = numeroPedidos2015["Order_ID"].nunique()
        self.assertEqual(numeroPedidos2015,722.05)
    
    def test_Pedidos2016(self): #TC15
        df = leerExcel()
        numeroPedidos2016 = df[df["Order_Date"].dt.year == 2016]
        numeroPedidos2016 = numeroPedidos2016["Order_ID"].nunique()
        self.assertEqual(numeroPedidos2016,722.05)
    
    def test_Pedidos2017(self): #TC16
        df = leerExcel()
        numeroPedidos2017 = df[df["Order_Date"].dt.year == 2017]
        numeroPedidos2017 = numeroPedidos2017["Order_ID"].nunique()
        self.assertEqual(numeroPedidos2017,722.05)
    
    def test_Pedidos2018(self): #TC17
        df = leerExcel()
        numeroPedidos2018 = df[df["Order_Date"].dt.year == 2018]
        numeroPedidos2018 = numeroPedidos2018["Order_ID"].nunique()
        self.assertEqual(numeroPedidos2018,722.05)
    
    def test_unidadesVendidasCategoría2015(self): #TC18
        predicciones = UnidadesCategoria()
        predicciones2015 = predicciones[predicciones.index.year == 2015]
        prediccionesTotal = predicciones2015["Total"].sum()
        prediccionesCategoría = predicciones2015[["Technology","Office Supplies", "Furniture"]].values.sum()
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_unidadesVendidasCategoría2016(self): #TC19
        predicciones = UnidadesCategoria()
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        prediccionesTotal = predicciones2016["Total"].sum()
        prediccionesCategoría = predicciones2016[["Technology","Office Supplies", "Furniture"]].values.sum()
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_unidadesVendidasCategoría2017(self): #TC20
        predicciones = UnidadesCategoria()
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        prediccionesTotal = predicciones2017["Total"].sum()
        prediccionesCategoría = predicciones2017[["Technology","Office Supplies", "Furniture"]].values.sum()
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_unidadesVendidasCategoría2018(self): #TC21
        predicciones = UnidadesCategoria()
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        prediccionesTotal = predicciones2018["Total"].sum()
        prediccionesCategoría = predicciones2018[["Technology","Office Supplies", "Furniture"]].values.sum()
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_unidadesVendidasCategoría2019(self): #TC22
        predicciones = UnidadesCategoria()
        predicciones2019 = predicciones[predicciones.index.year == 2019]
        prediccionesTotal = predicciones2019["Total"].sum()
        prediccionesCategoría = predicciones2019[["Technology","Office Supplies", "Furniture"]].values.sum()
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ventasAnualesCategoria2015(self): #TC23
        predicciones = VentasCategoria()
        predicciones2015 = predicciones[predicciones.index.year == 2015]
        prediccionesTotal = round(predicciones2015["Total"].sum(),0)
        prediccionesCategoría = round(predicciones2015[["Technology","Office Supplies","Furniture"]].values.sum(),0)
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ventasAnualesCategoria2016(self): #TC24
        predicciones = VentasCategoria()
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        prediccionesTotal = round(predicciones2016["Total"].sum(),0)
        prediccionesCategoría = round(predicciones2016[["Technology","Office Supplies","Furniture"]].values.sum(),0)
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ventasAnualesCategoria2017(self): #TC25
        predicciones = VentasCategoria()
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        prediccionesTotal = round(predicciones2017["Total"].sum(),0)
        prediccionesCategoría = round(predicciones2017[["Technology","Office Supplies","Furniture"]].values.sum(),0)
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ventasAnualesCategoria2018(self): #TC26
        predicciones = VentasCategoria()
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        prediccionesTotal = round(predicciones2018["Total"].sum(),0)
        prediccionesCategoría = round(predicciones2018[["Technology","Office Supplies","Furniture"]].values.sum(),0)
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ventasAnualesCategoria2019(self): #TC27
        predicciones = VentasCategoria()
        predicciones2019 = predicciones[predicciones.index.year == 2019]
        prediccionesTotal = round(predicciones2019["Total"].sum(),0)
        prediccionesCategoría = round(predicciones2019[["Technology","Office Supplies","Furniture"]].values.sum(),0)
        self.assertEqual(prediccionesTotal,prediccionesCategoría)
    
    def test_ratioIngresos1000habitantes(self): #TC28
        predicciones = VentasCategoria()
        prediccionesTotal = predicciones["Total"].sum()
        habitantes = 330000000
        ratio = round((prediccionesTotal/habitantes)*1000,2)
        self.assertEqual(ratio,0.22)
    
    def test_PorcentajeClientesRepiten(self):
        df = leerExcel()
        veces = df["Customer_ID"].value_counts()
        valores_repetidos = veces[veces > 1]
        clientesRepetidos = len(valores_repetidos)
        clientesTotales = df["Customer_ID"].nunique()
        porcentaje = round((clientesRepetidos/clientesTotales)*100,2)
        self.assertEqual(porcentaje,0)
    
    def test_ventasMediasPorCliente(self):
        df = leerExcel()
        saldoTotal = df["Total"].sum()
        clientes = df["Customer_ID"].nunique()
        ventasMedias = round(saldoTotal/clientes,2)
        self.assertEqual(ventasMedias,0)
    
    def test_pedidosMediosPorCliente(self):
        df = leerExcel()
        numeroPedidos = df["Order_ID"].nunique()
        clientes = df["Customer_ID"].nunique()
        pedidosMedios = round(numeroPedidos/clientes,2)
        self.assertEqual(pedidosMedios,0)
    
    def test_unidadesVendidasDeMediaPorCliente(self):
        df = leerExcel()
        unidadesVendidas = df["Product_ID"].count()
        clientes = df["Customer_ID"].nunique()
        unidadesVendidasMedias = round(unidadesVendidas/clientes,2)
        self.assertEqual(unidadesVendidasMedias,0)
    
    def test_variacionAnualVentas20152016(self):
        predicciones = VentasCategoria()
        predicciones2015 = predicciones[predicciones.index.year == 2015]
        prediccionesTotal2015 = round(predicciones2015["Total"].sum(),0)
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        prediccionesTotal2016 = round(predicciones2016["Total"].sum(),0)
        variacion = round(((prediccionesTotal2016/prediccionesTotal2015)-1)*100,2)
        self.assertEqual(variacion,0)
    
    def test_variacionAnualVentas20162017(self):
        predicciones = VentasCategoria()
        predicciones2016 = predicciones[predicciones.index.year == 2016]
        prediccionesTotal2016 = round(predicciones2016["Total"].sum(),0)
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        prediccionesTotal2017 = round(predicciones2017["Total"].sum(),0)
        variacion = round(((prediccionesTotal2017/prediccionesTotal2016)-1)*100,2)
        self.assertEqual(variacion,0)
    
    def test_variacionAnualVentas20172018(self):
        predicciones = VentasCategoria()
        predicciones2017 = predicciones[predicciones.index.year == 2017]
        prediccionesTotal2017 = round(predicciones2017["Total"].sum(),0)
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        prediccionesTotal2018 = round(predicciones2018["Total"].sum(),0)
        variacion = round(((prediccionesTotal2018/prediccionesTotal2017)-1)*100,2)
        self.assertEqual(variacion,0)
    
    def test_variacionAnualVentas20182019(self):
        predicciones = VentasCategoria()
        predicciones2018 = predicciones[predicciones.index.year == 2018]
        prediccionesTotal2018 = round(predicciones2018["Total"].sum(),0)
        predicciones2019 = predicciones[predicciones.index.year == 2019]
        prediccionesTotal2019 = round(predicciones2019["Total"].sum(),0)
        variacion = round(((prediccionesTotal2019/prediccionesTotal2018)-1)*100,2)
        self.assertEqual(variacion,0)
    



    
if __name__ == '__main__':
    unittest.main()
