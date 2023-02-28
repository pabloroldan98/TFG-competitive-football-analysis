# -*- coding: utf-8 -*-
"""
Created on Sun May 30 17:07:30 2021

@author: Pablo
"""

import os
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
import copy

#importing necessary packages for MDS
from sklearn.metrics import pairwise_distances #jaccard diss.
from sklearn import manifold  # multidimensional scaling

import math


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
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y porque quiero coger el elemento anterior
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
        
        
        
    


        

def calc_MDS(misLigas):
    resultLigas=copy.deepcopy(misLigas) #para tener una copia real y no algo que apunte al array
    #i=0
    #for miLiga in resultLigas[:-6]:
    for miLiga in resultLigas:
        j=1
        #Para tener una matriz de distancias con N*(R+1) elementos
        myS_i = miLiga[1:,1:].flatten().tolist()
        
        dis_matrix = pairwise_distances(myS_i, myS_i, metric='cityblock')
        
        mds_model = manifold.MDS(n_components = 2, metric=True, random_state=5, dissimilarity = 'precomputed')
        print(mds_model.get_params(deep=True))
        mds_coords = mds_model.fit_transform(dis_matrix)
        #Me salto la primera iteración de filas y columnas por ser la que ponen las etiquetas de jornadas y equipos respectivamente
        for miEquipo in miLiga[1:,1:]:
            #k=1, ya que la primera "jornada" corresponde al nombre del equipo
            k=1
            
            for miJornada in miEquipo:
                #Asigno a cada elemento, su coordenada del MDS, k-1 porque se salta la primera iteración
                miLiga[j][k]=mds_coords[(j-1)*len(miEquipo)+(k-1)]
                k=k+1
            j=j+1
        #i=i+1
    
    return resultLigas




def calc_MDS_T(misLigas):
    resultLigas=copy.deepcopy(misLigas) #para tener una copia real y no algo que apunte al array
    i=0
    for miLiga in resultLigas:
        #La transponemos para el cálculo
        miLiga = miLiga.T
        j=1
        #Para tener una matriz de distancias con N*(R+1) elementos
        myS_i = miLiga[1:,1:].flatten().tolist()
        
        dis_matrix = pairwise_distances(myS_i, myS_i, metric='cityblock')
        
        mds_model = manifold.MDS(n_components = 2, metric=True, random_state=5, dissimilarity = 'precomputed')
        print(mds_model.get_params(deep=True))
        mds_coords = mds_model.fit_transform(dis_matrix)
        #Me salto la primera iteración de filas y columnas por ser la que ponen las etiquetas de jornadas y equipos respectivamente
        for miJornada in miLiga[1:,1:]:
            #k=1, ya que la primera "jornada" corresponde al nombre del equipo
            k=1
            
            for miEquipo in miJornada:
                #Asigno a cada elemento, su coordenada del MDS, k-1 porque se salta la primera iteración
                miLiga[j][k]=mds_coords[(j-1)*len(miJornada)+(k-1)]
                k=k+1
            j=j+1
        #La destransponemos para el resultado con los equipos en las filas
        miLiga = miLiga.T
        i=i+1
    
    return resultLigas





def dibuja_MDS(misLigas_MDS):
    for miLiga_MDS in misLigas_MDS:
        j=1
        plt.figure(figsize=(12, 10))
        for miEquipo_MDS in miLiga_MDS[1:]:
            x=[]
            y=[]
            for miJornada_MDS in miEquipo_MDS[1:]:
                x.append(miJornada_MDS[0])
                y.append(miJornada_MDS[1])
            #Se representa puntos y líneas de cada jornada
            #plt.scatter(x, y, label=miLiga_MDS[j][0])
            plt.scatter(x, y)
#            plt.scatter(x, y, c='blue')
            plt.plot(x, y, label=miLiga_MDS[j][0])
            plt.annotate(miLiga_MDS[j][0], miLiga_MDS[j][-1]) #Para poner el nombre del equipo al final
            j=j+1

        plt.legend(loc='best')
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')
        plt.title(miLiga_MDS[0][0])
        plt.show()
        
    #dibuja_MDS_Simple(misLigas_MDS)



def dibuja_MDS_Simple(misLigas_MDS):
    for miLiga_MDS in misLigas_MDS:
        j=1
        plt.figure(figsize=(12, 10))
        for miEquipo_MDS in miLiga_MDS[1:]:
            #Solo representamos los puntos de la última jornada en el simple
            plt.scatter(miEquipo_MDS[-1][0], miEquipo_MDS[-1][1], label=miLiga_MDS[j][0])
#            plt.scatter(x, y, c='blue')
            plt.annotate(miLiga_MDS[j][0], miLiga_MDS[j][-1]) #Para poner el nombre del equipo al final
            #plt.annotate(miLiga_MDS[j][0], miLiga_MDS[j][-1]) #Para poner el nombre del equipo al final si está traspuesta?
            j=j+1

        plt.legend(loc='best')
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')
        plt.title(miLiga_MDS[0][0])
        plt.show()






def fix_MDS_ejeSim(misLigas_MDS):
    i=0
    for miLiga_MDS in misLigas_MDS:
        j=1
#        teamsToNotCheck = int((len(misLigas_MDS)-1) * (4/5)) #Solo corregimos los primeros equipos, pues son los que dan problemas (el primer quinto)
#        for miEquipo_MDS in miLiga_MDS[1:-teamsToNotCheck]:
        for miEquipo_MDS in miLiga_MDS[1:]:
            baseDist = calcDist(miEquipo_MDS[1], miEquipo_MDS[2])
            distToCheck = baseDist*3 #Checkearemos si es 3 veces mayor a la distancia base, para saber si sacar la linea de este
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y porque quiero coger el elemento anterior
            k=2
            equipoIrregular=False
            
#            midXTotal = 0
#            midYTotal = 0
#            for row_miJornada_MDS in range(len(miLiga_MDS[1:])): #Para calcular el pnto medio, cojo todos los penúltimos puntos, que el penúltimo suele estar bien
#                midXTotal += miLiga_MDS[row_miJornada_MDS+1][len(miEquipo_MDS)-1][0]
#                midYTotal += miLiga_MDS[row_miJornada_MDS+1][len(miEquipo_MDS)-1][1]
#            midPoint = np.array([midXTotal/len(miEquipo_MDS)-1, midYTotal/len(miEquipo_MDS)-1])
            
            for miJornada_MDS in reversed(miEquipo_MDS[2:]): #Nos la recorremos de atras a alante para un mejor midPoint
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo_MDS[-k] #El último elemento -2 es el que corresponde al anterior
                distJornadas = calcDist(prevJornadaTabla, miJornada_MDS)
                if (distJornadas >= distToCheck):
                    #print(miLiga_MDS[j][0])
                    #print(len(miEquipo_MDS)-k)
                    equipoIrregular=True
                    midPoint = np.array([(miJornada_MDS[0] + prevJornadaTabla[0])/2, (miJornada_MDS[1] + prevJornadaTabla[1])/2])
                    #midPoint = np.array([0, 0])
                    #print(midPoint)
                    equipoFixed = fixEquipo_ejeSim(miEquipo_MDS[1], midPoint, miEquipo_MDS, miLiga_MDS[-1][-1]) #Esta función me devuelve una lista con los puntos bien
                    break
                k=k+1
            if equipoIrregular==True: #Si True, entonces asigno la lista corregida
                l=1
                for miJornada_MDS in miEquipo_MDS[1:]:
                    misLigas_MDS[i][j][l] = equipoFixed[l]
                    l=l+1
                equipoIrregular=False
            j=j+1
        i=i+1
    
    return misLigas_MDS






def fixEquipo_ejeSim(iniEje, finEje, puntosEquipo, lastPointsLastTeam):#Se calculará el punto simétrico si se le ha pirado la pinza y lo ha puesto donde le ha dado la gana
    puntosEquipoSimetrico=copy.deepcopy(puntosEquipo) #para tener una copia real y no algo que apunte al array, además, lo quiero del mismo tamaño
    k1=1
    for miJornada in puntosEquipo[1:]: #Me salto la primera porque es igual para los 2
        puntoSimetrico = calcPuntoSim(iniEje, finEje, miJornada)
        puntosEquipoSimetrico[k1] = puntoSimetrico
        k1=k1+1
    
    resultEquipoOG=copy.deepcopy(puntosEquipo) #para tener una copia real y no algo que apunte al array, de los del punto de partida
    resultEquipoSim=copy.deepcopy(puntosEquipoSimetrico) #para tener una copia real y no algo que apunte al array, de los simétricos
    k2=3
    for miJornadaSim in puntosEquipoSimetrico[3:]:#Me salto los 2 primeros (+1 del nombre) porque serán equidistantes, quiero sacar de aquí 2 listas, la buena y la simétrica, y luego quedarme con la buena
        prevJornada = resultEquipoOG[k2-1]
        prevJornadaSim = resultEquipoSim[k2-1]
        miJornada = puntosEquipo[k2]

        puntosCercanos = [miJornada,miJornadaSim]
        #Cálculo del índice del punto más cercano entre el simétrico y el original
        indexOG = calcPuntoMasCercano(puntosCercanos, prevJornada)
        indexSim = calcPuntoMasCercano(puntosCercanos, prevJornadaSim)
        
        resultEquipoOG[k2] = puntosCercanos[indexOG]
        resultEquipoSim[k2] = puntosCercanos[indexSim]
        
        k2=k2+1
    
    lastDistOG = calcDist(resultEquipoOG[-1], lastPointsLastTeam)
    lastDistSim = calcDist(resultEquipoSim[-1], lastPointsLastTeam)
    
    if(lastDistOG < lastDistSim): #Se comprueba cuál está más cerca del último punto del último equipo que siempre está bien colocado, ese será el malo
        resultEquipo = resultEquipoSim
    else: #lastDistOG >= lastDistSim
        resultEquipo = resultEquipoOG
    
    return resultEquipo



def calcDist(p1,p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1]) #math.hypot(x2 - x1, y2 - y1)

def calcPuntoMasCercano(listaPuntos, punto):
    return np.argmin(np.sum((np.array(listaPuntos) - np.array(punto))**2, axis=1))


def calcPuntoSim(iniEje, finEje, punto):
    x1,y1 = iniEje[0],iniEje[1]
    x2,y2 = finEje[0],finEje[1]
    x3,y3 = punto[0],punto[1]
    if x1==x2:
        puntoSimetrico = caculate1(x1,x3,y3)
    elif y1==y2:
        puntoSimetrico = caculate2(y1,y3,x3)
    else:
        A=y1-y2
        B=x2-x1
        C=x1*y2-y1*x2
        puntoSimetrico = caculate3(A,B,C,x3,y3)
    return puntoSimetrico

def caculate1(x1,x3,y3):
    #"" "Calculate the symmetrical point of the line under special circumstances, the coordinates X of the two points entered are the same, that is, the symmetrical point about the line parallel to the Y axis" ""
    x4=2*x1-x3
    y4=y3
    return np.array([x4, y4])

def caculate2(y1,y3,x3):
    #"" "Calculate the symmetrical point of the line under special circumstances, the coordinates Y of the two input points are the same, that is, the symmetrical point about the line parallel to the X axis" ""
    x4=x3
    y4=2*y1-y3
    return np.array([x4, y4])

def caculate3(A,B,C,x3,y3):
    #"" "Calculate the straight line symmetry point of the general case, and derive the mathematical relationship" " based on the slope relationship "
    x4=x3-2*A*((A*x3+B*y3+C)/(A*A+B*B))
    y4=y3-2*B*((A*x3+B*y3+C)/(A*A+B*B))
    return np.array([x4, y4])




def fix_MDS_intermedios(misLigas_MDS):
    i=0
    for miLiga_MDS in misLigas_MDS:
        j=1        
        #iniEje = miLiga_MDS[len(miLiga_MDS)//2][1] #Se coge el punto inicial de un equipo que esté en la mitad (los puntos iniciales son iguales para todos los equipos realmente)
        #finEje = np.array([(miLiga_MDS[len(miLiga_MDS)//2][-1][0] + miLiga_MDS[(len(miLiga_MDS)+1)//2][-1][0])/2, (miLiga_MDS[(len(miLiga_MDS)//2)-1][-1][1] + miLiga_MDS[(len(miLiga_MDS)+1)//2][-1][1])/2])
        for miEquipo_MDS in miLiga_MDS[1:]:
            baseDist = calcDist(miEquipo_MDS[1], miEquipo_MDS[2])
            distToCheck = baseDist*3 #Checkearemos si es 3 veces mayor a la distancia base, para saber si sacar la linea de este
            
            iniEje = miEquipo_MDS[1]
            finEje = np.array([0, 0])
            puntoFinalSim = calcPuntoSim(iniEje, finEje, miEquipo_MDS[-1])
            
            lastDistOG = calcDist(miEquipo_MDS[-1], miLiga_MDS[-1][-1])
            lastDistSim = calcDist(puntoFinalSim, miLiga_MDS[-1][-1])

            #Checkea si el punto final está bien en función del último equipo de la última jornada dependiendo de la distancia a la que esté de este
            #También puede checkear en qué dirección está el último punto de la última jornada (este siempre está bien), para saber si ir hacia arriba es bueno o malo cuando hay un gap grande, pero en múltiples ejemplos siempre se cumple que, si va hacia arriba, va a una posición incorrecta, si va hacia abajo, va a una correcta, así que posible mejora y fuera
            if j<(len(miLiga_MDS)/2):
                if(lastDistOG < lastDistSim): #Si está más cerca del Original, que del simétrico para los primeros equipos --> está mal
                    puntoFinalMal = True
                else: #lastDistOG >= lastDistSim
                    puntoFinalMal = False
            else:
                if(lastDistOG > lastDistSim): #Si está más cerca del Simétrico, que del original para los últimos equipos --> está mal
                    puntoFinalMal = True
                else: #lastDistOG >= lastDistSim
                    puntoFinalMal = False
            isPuntoFinalNotFixed = True
            
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y porque quiero coger el elemento anterior
            k=2
            equipoIrregular=False
            e=1 #Me salto el nombre del equipo
            for miJornada_MDS in miEquipo_MDS[2:]:
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo_MDS[k-1]
                distJornadas = calcDist(prevJornadaTabla, miJornada_MDS)
                if (distJornadas >= distToCheck): #Si la distancia es más grande que 3 veces la base, significa que hay que arreglar algo
                    equipoIrregular=True
                    puntoSim = calcPuntoSim(iniEje, finEje, miJornada_MDS)
                    
                    distOGaFinal = calcDist(miJornada_MDS, miEquipo_MDS[-1])
                    distSimaFinal = calcDist(puntoSim, miEquipo_MDS[-1])
                    if (puntoFinalMal == False):#Checkea en función de si el punto final está bien colocado o no, cuál está más cerca, si el simétrico o el original
                        if (distSimaFinal > distOGaFinal):#Si el punto bien colocado (si ha pasado de mal colocado a bien colocado) (el simétrico está más alejado que el original del punto final bien colocado)
                            #fixing from e to k
                            fixedPoints = intermediates(miEquipo_MDS[e], miEquipo_MDS[k], k-(e+1)) #k>=2 siempre porque sino no se da (distJornadas >= distToCheck)
                            fixIndex = 0
                            for x in range(e+1, k): #desde e+1, porque ni e ni k tienen que ser fixeados, solo los de en medio
                                miEquipo_MDS[x] = np.array(fixedPoints[fixIndex])
                                fixIndex = fixIndex+1
                            #updating e
                            e=k
                        else: #Si el punto está mal colocado (si ha pasado de estar bien colocado a estar mal colocado)
                            #updating e
                            e=k-1 #a k-1 y no a k, porque esta k si que querré fixearla
                    else: #Si el punto final está mal colocado
                        if (distSimaFinal < distOGaFinal):#Si el punto bien colocado (si ha pasado de mal colocado a bien colocado) (el simétrico está más cerca que el original del punto final con el punto final mal colocado)
                            #fixing from e to k
                            fixedPoints = intermediates(miEquipo_MDS[e], miEquipo_MDS[k], k-(e+1)) #k>=2 siempre porque sino no se da (distJornadas >= distToCheck)
                            fixIndex = 0
                            for x in range(e+1, k): #desde e+1, porque ni e ni k tienen que ser fixeados, solo los de en medio
                                miEquipo_MDS[x] = np.array(fixedPoints[fixIndex])
                                fixIndex = fixIndex+1
                            #updating e
                            e=k
                        else: #Si el punto está mal colocado (si ha pasado de estar bien colocado a estar mal colocado)
                            #updating e
                            e=k-1 #a k-1 y no a k, porque esta k si que querré fixearla
                            
#                    print(miEquipo_MDS[0])
#                    print(k)
                k=k+1
            
            #No se puede hacer a la vez, porque los últimos mal se corrigen con un método diferente
            if (equipoIrregular and puntoFinalMal and isPuntoFinalNotFixed):
                #Para corregir los últimos, se pone con la distancia base a en la misma dirección que desde el inicio al último que estaba bien (e)
#                print(e)
                print(miEquipo_MDS[0])
                iniRecta = miEquipo_MDS[1]
                finRecta = miEquipo_MDS[e]
                dist = baseDist*2 #La multiplicamos x2 para que se quede más cercano a como debe
                
                for x in range(e+1, len(miEquipo_MDS)): #hasta len(miEquipo_MDS), porque quiero fixear hasta el último elemento
                    #print(x)
                    miEquipo_MDS[x] = calcPuntoConRectaDistPunto(iniRecta, finRecta, dist, miEquipo_MDS[x-1])
                
                isPuntoFinalNotFixed = False #Se acaba de arreglar
                        
            j=j+1
        i=i+1
    
    return misLigas_MDS



def intermediates(p1, p2, nb_points):
#    """"Return a list of nb_points equally spaced points
#    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)]
   
   
def calcPuntoConRectaDistPunto(iniRecta, finRecta, dist, punto):
    m = (finRecta[1]-iniRecta[1])/(finRecta[0]-iniRecta[0])
    return (punto[0]+dx(dist,m), punto[1]+dy(dist,m))
#   return (punto[0]-dx(dist,m), punto[1]-dy(dist,m)) # going the other way

def dy(distance, m):
    return m*dx(distance, m)

def dx(distance, m):
    return math.sqrt(distance/(m**2+1))





#Cogemos los datos
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Datos Ligas.xlsx')
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Conexion_DatosLigas.xlsx')

#"MDS.py" y "Conexion_DatosLigas.xlsx" tienen que estar en el mismo directorio
# Get the current working directory
cwd = os.getcwd()

listaLigas = getDataExcel(cwd+'\\Conexion_DatosLigas.xlsx')



listaLigas_S_i_tabla = calc_S_i_tabla(listaLigas)
        
        
        
#FUNCIONA MEJOR:
listaLigas_MDS = calc_MDS(listaLigas_S_i_tabla)

#funciona peor:
#listaLigas_MDS = calc_MDS_T(listaLigas_S_i_tabla)# Método alternativo


#funciona peor:
#listaLigas_MDS = fix_MDS_ejeSim(listaLigas_MDS) #Para arreglar los valores que cruzan toda la gráfica para los primeros equipos

#FUNCIONA MEJOR:
listaLigas_MDS = fix_MDS_intermedios(listaLigas_MDS) #Para arreglar los valores que cruzan toda la gráfica para los primeros equipos de forma alternativa



#print(listaLigas_MDS[0])




dibuja_MDS(listaLigas_MDS)
#dibuja_MDS_Simple(listaLigas_MDS)



