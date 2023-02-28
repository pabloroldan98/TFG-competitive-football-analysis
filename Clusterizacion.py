# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:03:05 2021

@author: Pablo
"""

import os
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
import copy
from scipy.cluster.hierarchy import dendrogram, linkage
import cmath
from sklearn.cluster import AgglomerativeClustering


def getDataExcel(fromFile):
    data = pd.read_excel (fromFile) # Lee los datos
    df = data.iloc[: , 10:] # Se quita las 10 primeras columnas que no valen para nada, son datos extra


    #this basically converts your table into 0s and 1s where 0 is NaN and 1 for non NaN 
    binary_rep = np.array(df.notnull().astype('int'))

    list_of_dataframes = []
    l = label(binary_rep)
    for s in regionprops(l):
        #the bbox contains the extremes of the bounding box. So the top left and bottom right cell locations of the table.
        list_of_dataframes.append(df.iloc[s.bbox[0]:s.bbox[2],s.bbox[1]:s.bbox[3]].to_numpy())
    
    return list_of_dataframes


def calc_S_i_tabla(misLigas):
    resultLigas=copy.deepcopy(misLigas) #para tener una copia real y no algo que apunte al array
    i=0
    for miLiga in resultLigas:
        j=1
        #Me salto la primera iteración por ser la que pone las etiquetas de las columnas
        for miEquipo in miLiga[1:]:
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y quiero coger el elemento anterior
            k=2
            for miJornada in miEquipo[2:]:
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo[k-1]
                puntosGanados = int(miJornada)-int(misLigas[i][j][k-1]) #hecho así y no con "prevJornadaTabla" porque el dato de miEquipo[k] lo sobreescribo
                #Calculo S_i(1)
                if k == 2:
                    if puntosGanados == 3:
                        miEquipo[k] = np.array([3, 0])
                    elif puntosGanados == 1:
                        miEquipo[k] = np.array([1, 2])
                    elif puntosGanados == 0:
                        miEquipo[k] = np.array([0, 3])
                #Calculo S_i(k)
                else:
                    if puntosGanados == 3:
                        miEquipo[k] = np.add(prevJornadaTabla, [3, 0])
                    elif puntosGanados == 1:
                        miEquipo[k] = np.add(prevJornadaTabla, [1, 2])
                    elif puntosGanados == 0:
                        miEquipo[k] = np.add(prevJornadaTabla, [0, 3])
                k=k+1
            #S_i(0)=[0, 0]
            resultLigas[i][j][1] = np.array([0, 0])
            j=j+1
        #print(resultLigas[i])
        i=i+1
    
    return resultLigas


def calc_S_i_real(misLigas):
    resultLigas=copy.deepcopy(misLigas) #para tener una copia real y no algo que apunte al array
    i=0
    for miLiga in resultLigas:
        j=1
        #Me salto la primera iteración por ser la que pone las etiquetas de las columnas
        for miEquipo in miLiga[1:]:
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y quiero coger el elemento anterior
            k=2
            for miJornada in miEquipo[2:]:
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo[k-1]
                puntosGanados = int(miJornada)-int(misLigas[i][j][k-1]) #hecho así y no con "prevJornadaTabla" porque el dato de miEquipo[k] lo sobreescribo
                #Calculo S_i(1)
                if k == 2:
                    if puntosGanados == 3:
                        miEquipo[k] = 3
                    elif puntosGanados == 1:
                        miEquipo[k] = 1+2*cmath.sqrt(-1)
                    elif puntosGanados == 0:
                        miEquipo[k] = 0+3*cmath.sqrt(-1)
                #Calculo S_i(k)
                else:
                    if puntosGanados == 3:
                        miEquipo[k] = prevJornadaTabla+3
                    elif puntosGanados == 1:
                        miEquipo[k] = prevJornadaTabla+1+2*cmath.sqrt(-1)
                    elif puntosGanados == 0:
                        miEquipo[k] = prevJornadaTabla+0+3*cmath.sqrt(-1)
                k=k+1
            #S_i(0)=0
            resultLigas[i][j][1] = 0
            j=j+1
        #print(resultLigas[i])
        i=i+1
    
    return resultLigas



def calcDibuja_Cluster(misLigas_S_i):
    for miLiga_S_i in misLigas_S_i:
        
        ligaConcreta = miLiga_S_i[1:,-1].tolist() #S_iR
        
        labelList = miLiga_S_i[1:,0]
        
        
        linked = linkage(ligaConcreta,
                         method='average',# Por algún motivo, a pesar de que el artículo nos dice que va a usar el método de la media 'average', cuando se hacen pruebas utiliza el completo 'complete'
                         metric='euclidean',# Aunque en el MDS usa 'cityblock', en los clústeres no deja claro si usa 'euclidean' o 'cityblock', pero la fórmula parece ser la 'euclidean' (si se interpreta como módulo --> 'euclidean', si se interpreta como valor absoluto --> 'cityblock')
                         optimal_ordering=True)
        
        plt.figure(figsize=(12, 7))
        plt.title(miLiga_S_i[0][0])
        dendrogram(linked, 
                   get_leaves = True,
                   orientation='left',
                   labels=labelList,
                   count_sort=False,
                   distance_sort=False,
                   show_leaf_counts=True,
                   show_contracted=True)
        
        plt.xlim(xmax=-1)# hace que los nombres no estén tan pegados
        #plt.axvline(x=19, color='red', linestyle='--')
        
        plt.show()
        
        
        #Para enseñar los clusteres y sus colores (5 clusters para todas las ligas porque el número de clusters en hierirchical clustering (agrupamiento jerárquico) es subjetivo: https://stats.stackexchange.com/questions/66128/choosing-the-number-of-clusters-in-hierarchical-agglomerative-clustering)
        
        cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
        cluster.fit_predict(ligaConcreta)
        ligaConcreta = np.array(ligaConcreta)
        
        plt.figure(figsize=(12, 7))
        plt.title(miLiga_S_i[0][0])
        i=0
        for miEquipo_S_i in ligaConcreta:
            plt.annotate(labelList[i], miEquipo_S_i)
            i=i+1
        
        plt.scatter(ligaConcreta[:,0],ligaConcreta[:,1], c=cluster.labels_, cmap='rainbow')
        plt.show()



def dibuja_S_i(misLigas_S_i):
    for miLiga_S_i in misLigas_S_i:
        j=1
        plt.figure(figsize=(12, 10))
        for miEquipo_S_i in miLiga_S_i[1:]:
            x=[]
            y=[]
            for miJornada_S_i in miEquipo_S_i[1:]:
                x.append(miJornada_S_i[0])
                y.append(miJornada_S_i[1])
            
            plt.plot(x, y, label=miLiga_S_i[j][0])
            plt.annotate(miLiga_S_i[j][0], miLiga_S_i[j][-1]) #Para poner el nombre del equipo al final
            j=j+1

        diagFinal=(len(miLiga_S_i[0])-2)*3
        plt.plot([diagFinal,0], [0,diagFinal],'b--', label="k="+str(len(miLiga_S_i[0])-2))
        plt.xlim(xmin=-1)
        plt.ylim(ymin=-1)
        plt.legend(loc='best')
        plt.xlabel("P")
        plt.ylabel("Q")
        plt.title(miLiga_S_i[0][0])
        plt.show()




#Cogemos los datos
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Datos Ligas.xlsx')
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Conexion_DatosLigas.xlsx')

#"Clusterización.py" y "Conexion_DatosLigas.xlsx" tienen que estar en el mismo directorio
# Get the current working directory
cwd = os.getcwd()

listaLigas = getDataExcel(cwd+'\\Conexion_DatosLigas.xlsx')



listaLigas_S_i_tabla = calc_S_i_tabla(listaLigas)

#listaLigas_S_i_real = calc_S_i_real(listaLigas)




dibuja_S_i(listaLigas_S_i_tabla)



calcDibuja_Cluster(listaLigas_S_i_tabla)





#print(listaLigas_S_i_real)







#print (listaLigas_S_i_tabla[0][0][0])




