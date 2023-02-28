# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:54:28 2021

@author: Pablo
"""

import os
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
import copy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
#from scipy.spatial.distance import cdist

#importing necessary packages
from sklearn.metrics import pairwise_distances #jaccard diss.
from sklearn import manifold  # multidimensional scaling

import math

from skimage.io import imread
from skimage import color, img_as_ubyte
from skimage.feature import greycomatrix

import matplotlib.patches as mpatches
import seaborn as sns

#import dataframe_image as dfi #Para guardar tablas



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
                #Calculo S_1
                if k == 2:
                    if puntosGanados == 3:
                        miEquipo[k] = np.array([3, 0])
                    elif puntosGanados == 1:
                        miEquipo[k] = np.array([1, 2])
                    elif puntosGanados == 0:
                        miEquipo[k] = np.array([0, 3])
                #Calculo S_i
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


def calcCluster(misLigas_S_i):
    for miLiga_S_i in misLigas_S_i:
        
        ligaConcreta = miLiga_S_i[1:,-1].tolist() #S_iR
        
        labelList = miLiga_S_i[1:,0]
        
        
        linked = linkage(ligaConcreta,
                         method='average',# Por algún motivo, a pesar de que el artículo nos dice que va a usar el método de la media 'average', cuando se hacen pruebas utiliza el completo 'complete'
                         metric='cityblock',# Aunque en el MDS usa 'cityblock', en los clústeres no deja claro si usa 'euclidean' o 'cityblock', pero por la fórmula, parece ser la 'cityblock'
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

        diagFinal=(len(miLiga_S_i[0])-1)*3
        plt.plot([diagFinal,0], [0,diagFinal],'b--', label="k="+str(len(miLiga_S_i[0])-1))
        plt.xlim(xmin=-1)
        plt.ylim(ymin=-1)
        plt.legend(loc='best')
        plt.xlabel("P")
        plt.ylabel("Q")
        plt.title(miLiga_S_i[0][0])
        plt.show()
        
        


        

def calc_MDS(misLigas):
    resultLigas=copy.deepcopy(misLigas) #para tener una copia real y no algo que apunte al array
    #i=0
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
                iniRecta = miEquipo_MDS[1]
                finRecta = miEquipo_MDS[e]
                dist = baseDist*2 #La multiplicamos x2 para que se quede más cercano a como debe
                
                for x in range(e+1, len(miEquipo_MDS)): #hasta len(miEquipo_MDS), porque quiero fixear hasta el último elemento
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









def calc_Entropia(misLigas):
    resultEntropias = []
    
    for miLiga in misLigas:
        j=1
        plt.figure(figsize=(12, 12))
        for miEquipo in miLiga[1:]:
            x=[]
            y=[]
            for miJornada in miEquipo[1:]:
                x.append(miJornada[0])
                y.append(miJornada[1])
            #Se representan con PUNTOS negros los equipos durante sus jornadas
            plt.scatter(x, y, c='black')
            j=j+1

        plt.axis('off')
        
        imgName = miLiga[0][0].replace("/", "")+'_entropia.png'
        # Get the current working directory
        cwd = os.getcwd()
        
        fileName = cwd+'\\'+imgName
        
        #Se guarda la imagen para poder abrirla con imread
        plt.savefig(imgName)
        #Se abre la imagen en forma de array 2D
        img = imread(fileName)
        #img = img_as_ubyte(color.rgb2gray(img)) #Lo paso a blanco y negro, es decir, de un array 3D a un array 2D
        img = img_as_ubyte(color.rgb2gray(color.rgba2rgb(img))) #Lo paso a blanco y negro, es decir, de un array 3D a un array 2D
        
        #Cálculo de la Entropía con GLCM
        entropia = entropy(img)
        
        #Elimino la imagen creada
        os.remove(fileName)
        
        # #Printeo los resultados
        # print("Entropía de "+miLiga[0][0]+": "+entropia)
        plt.show()
        
        resultEntropias.append(entropia)
        
    return resultEntropias


def entropy(imagen):
    #Cálculo de la Entropía con GLCM
    glcm = np.squeeze(greycomatrix(imagen, distances=[2], 
                                angles=[0], symmetric=True, 
                                normed=True))
    
    return -np.sum(glcm*np.log2(glcm + (glcm==0)))





def calc_DimFractal(misLigas):
    resultDimFractal = []
    
    for miLiga in misLigas:
        j=1
        plt.figure(figsize=(12, 12))
        for miEquipo in miLiga[1:]:
            x=[]
            y=[]
            for miJornada in miEquipo[1:]:
                x.append(miJornada[0])
                y.append(miJornada[1])
            #Se representan con LÍNEAS negras los equipos durante sus jornadas
            plt.plot(x, y, c='black')
            j=j+1

        plt.axis('off')
        
        imgName = miLiga[0][0].replace("/", "")+'_fractal.png'
        # Get the current working directory
        cwd = os.getcwd()
        
        fileName = cwd+'\\'+imgName
        
        #Se guarda la imagen para poder abrirla con imread
        plt.savefig(imgName)
        #Se abre la imagen en forma de array 2D
        img = imread(fileName)
        img = img_as_ubyte(color.rgb2gray(color.rgba2rgb(img))) #Lo paso a blanco y negro, es decir, de un array 3D a un array 2D
        
        #Cálculo de la Dimensión fractal con algoritmo de Box counting (Minkowski–Bouligand dimension)
        dimFractal = fractal_dimension(img/256.0) #/256.0 para que pueda convertir la matriz a binario fácilmente
        
        #Elimino la imagen creada
        os.remove(fileName)
        
        # #Printeo los resultados
        # print("Dimensión fractal de "+miLiga[0][0]+": ")
        # print(dimFractal)
        plt.show()
        
        resultDimFractal.append(dimFractal)
        
    return resultDimFractal



def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]






def calc_DesvTipica(misLigas):
    desvTipicaLigas=[] #La lista de desviaciones típicas de cada liga
    # i=0
    for miLiga in misLigas:
        listaW_i=[]
        # j=1
        #Me salto la primera iteración por ser la que pone las etiquetas de las columnas
        for miEquipo in miLiga[1:]:
            w_i=0
            numJornadas = len(miEquipo)-2 #menos 2 porque tengo en el 0 el nombre del equipo y en el 1 la jornada 0
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y porque quiero coger el elemento anterior
            k=2
            for miJornada in miEquipo[2:]:
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo[k-1]
                puntosGanados = int(miJornada)-int(prevJornadaTabla) #Los puntos de la jornada menos la anterior
                #Calculo w_i para cada equipo
                if puntosGanados == 3:
                    w_i = w_i + 1
                k=k+1
                #Añado los resultados al total
            listaW_i.append(w_i/numJornadas)
            # j=j+1
        desvTipica = desviacion_tipica(listaW_i)
        desvTipicaLigas.append(desvTipica)
        # i=i+1
    
    return desvTipicaLigas


def desviacion_tipica(allW_i): #Calc sigma=((1/N)*sum(w_i-1/2)^2)^(1/2)
    n = len(allW_i)
    varianza = sum((w_i - 0.5) ** 2 for w_i in allW_i) / n
    return math.sqrt(varianza)



def calc_HICB(misLigas):
    HICBLigas=[] #La lista de desviaciones típicas de cada liga
    # i=0
    for miLiga in misLigas:
        totalPuntos = 0 #Se divide por los puntos totales de todos los equipos, no por los que podría haber conseguido ese equipo
        listaS_i=[]
        # j=1
        #Me salto la primera iteración por ser la que pone las etiquetas de las columnas
        for miEquipo in miLiga[1:]:
            s_i=0
            #k=2 y no k=0 o k=1, porque las 2 primeras iteraciones me las salto, ya que la primera "jornada" corresponde al nombre del equipo y porque quiero coger el elemento anterior
            k=2
            for miJornada in miEquipo[2:]:
                #Cojo el elemento anterior, da igual que se haga con k-1, porque se salta las 2 primeras iteraciones
                prevJornadaTabla = miEquipo[k-1]
                puntosGanados = int(miJornada)-int(prevJornadaTabla) #Los puntos de la jornada menos la anterior
                #Calculo s_i para cada equipo
                if puntosGanados == 3:
                    s_i = s_i + 3
                    totalPuntos = totalPuntos + 3
                elif puntosGanados == 1:
                    s_i = s_i + 1
                    totalPuntos = totalPuntos + 1
                k=k+1
                #Añado los resultados al total
            listaS_i.append(s_i)
            # j=j+1
        #Dividimos todos los puntos de cada equipo al final de temporada entre el total de puntos de cada equipo al final de temporada
        listaS_i = np.array(listaS_i)/totalPuntos
        
        HICB = HICB_index(listaS_i)
        HICBLigas.append(HICB)
        # i=i+1
    
    return HICBLigas


def HICB_index(allS_i): #Calc HICB=100*N*sum(s_i^2)
    n = len(allS_i)
    return 100*n*sum((s_i) ** 2 for s_i in allS_i)




def calc_Correlacion(medidasCompetitividad):
    corPearson = np.corrcoef(medidasCompetitividad)
    return corPearson




def dibuja_MedidasComp(dfMedidasComp):
    labelsMedidasComp = dfMedidasComp.index.tolist()
    labelsLigas = dfMedidasComp.columns.tolist()
    medidasComp = dfMedidasComp.to_numpy()
    
    x = []
    for i in range(len(labelsLigas)):
        x.append(i+1)
    
    i=0
    for medida in medidasComp:
        plt.figure(figsize=(12, 10))
        j=0
        for valorMedida in medida:
            plt.scatter(x[j], valorMedida, c="red") #Punto rojo al final
            
            plt.plot([x[j], x[j]], [0, valorMedida], c="red") #Línea desde 0 al punto rojo
            
            j=j+1
        
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(x, labelsLigas, rotation='vertical')
        
        plt.ylim(ymin=min(medida)*0.9)
        
        plt.title(labelsMedidasComp[i])
        plt.show()
        
        i=i+1
    
    




def dibuja_CorrelacionMedidas(labelsMedidas, correlacionMedidas):
    plt.figure(figsize=(12, 10))
    
    coordMedidas = []
    
    n=len(correlacionMedidas)
    radius=-3
    for i in range(n):
        x = radius * math.cos(2 * math.pi * i / n);
        y = radius * math.sin(2 * math.pi * i / n);
        coordMedidas.append([x,y])
        
        plt.scatter(x, y)
        plt.annotate(labelsMedidas[i], (x,y)) #Para poner el nombre de la medida en ese punto
    
    for i in range(n):   
        for j in range(n):
            if i==j:
                break
            if 0.9 <= abs(correlacionMedidas[i][j]) < 1:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="red")
            elif 0.8 <= abs(correlacionMedidas[i][j]) < 0.9:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="orange")
            elif 0.7 <= abs(correlacionMedidas[i][j]) < 0.8:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="yellow")
            elif 0.6 <= abs(correlacionMedidas[i][j]) < 0.7:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="lime")
            elif 0.5 <= abs(correlacionMedidas[i][j]) < 0.6:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="cyan")
            elif 0.4 <= abs(correlacionMedidas[i][j]) < 0.5:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="blue")
            elif 0.3 <= abs(correlacionMedidas[i][j]) < 0.4:
                plt.plot([coordMedidas[i][0], coordMedidas[j][0]], [coordMedidas[i][1], coordMedidas[j][1]], c="darkblue")
    
    red_patch = mpatches.Patch(color="red", label="|0.9| - |1|")
    orange_patch = mpatches.Patch(color="orange", label="|0.8| - |0.9|")
    yellow_patch = mpatches.Patch(color="yellow", label="|0.7| - |0.8|")
    lime_patch = mpatches.Patch(color="lime", label="|0.6| - |0.7|")
    cyan_patch = mpatches.Patch(color="cyan", label="|0.5| - |0.6|")
    blue_patch = mpatches.Patch(color="blue", label="|0.4| - |0.5|")
    darkblue_patch = mpatches.Patch(color="darkblue", label="|0.3| - |0.4|")

    plt.legend(handles=[red_patch, orange_patch, yellow_patch, lime_patch, cyan_patch, blue_patch, darkblue_patch], loc='best')
    plt.axis('off')
    plt.title("Correlación de Pearson entre las distintas medidas de competitividad")
    plt.show()




def dibuja_MatrizCorrelacion(dfCorMatrix):
    plt.figure(figsize=(12, 10))
    
    #sns.heatmap(dfCorMatrix, vmin=0.4, vmax=1, annot=True, fmt="g", cmap='Spectral_r')
    sns.heatmap(dfCorMatrix, annot=True, fmt="g", cmap='Spectral_r')
    plt.show()
    




#Cogemos los datos
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Datos Ligas.xlsx')
#listaLigas = getDataExcel(r'C:\Users\Pablo\Documents\TFGs\Informática\Conexion_DatosLigas.xlsx')

#"Competitividad.py" y "Conexion_DatosLigas.xlsx" tienen que estar en el mismo directorio
# Get the current working directory
cwd = os.getcwd()

listaLigas = getDataExcel(cwd+'\\Conexion_DatosLigas.xlsx')



listaLigas_S_i_tabla = calc_S_i_tabla(listaLigas)
        
        
        

listaLigas_MDS = calc_MDS(listaLigas_S_i_tabla)

listaLigas_MDS = fix_MDS_intermedios(listaLigas_MDS) #Para arreglar los valores que cruzan toda la gráfica para los primeros equipos de forma alternativa


#print(listaLigas_MDS[0])



ligasMedidasCompetitividad = []

dimfractalLigas_S_i = calc_DimFractal(listaLigas_S_i_tabla)
dimfractalLigas_MDS = calc_DimFractal(listaLigas_MDS)

entropiaLigas_S_i = calc_Entropia(listaLigas_S_i_tabla)
entropiaLigas_MDS = calc_Entropia(listaLigas_MDS)


desvtipicaLigas = calc_DesvTipica(listaLigas)
HICBLigas = calc_HICB(listaLigas)


ligasMedidasCompetitividad.append(dimfractalLigas_S_i)
ligasMedidasCompetitividad.append(dimfractalLigas_MDS)
ligasMedidasCompetitividad.append(entropiaLigas_S_i)
ligasMedidasCompetitividad.append(entropiaLigas_MDS)
ligasMedidasCompetitividad.append(desvtipicaLigas)
ligasMedidasCompetitividad.append(HICBLigas)


labelsMedidas = ["dimfractalLigas_S_i", "dimfractalLigas_MDS", "entropiaLigas_S_i", "entropiaLigas_MDS", "desvtipicaLigas", "HICBLigas"]

labelsLigas = []
for liga in listaLigas:
    labelsLigas.append(liga[0][0])



#Me construyo un dataFrame con todas las medidas y las ligas
i=0
data = []
for medida in labelsMedidas:
    data.append(ligasMedidasCompetitividad[i])
    i=i+1




dfMedidasCompetitividad = pd.DataFrame(data, index=labelsMedidas, columns=labelsLigas)


dfMedidasCompetitividad = dfMedidasCompetitividad.sort_values(by=['HICBLigas'], axis=1) #Las ordeno de menor a mayor HICB

dibuja_MedidasComp(dfMedidasCompetitividad)



correlacionMedidas = calc_Correlacion(dfMedidasCompetitividad.to_numpy())

dibuja_CorrelacionMedidas(labelsMedidas, correlacionMedidas)





dfMedidasCompetitividad = dfMedidasCompetitividad.T #Se transpone porque para enseñarlas quiero las ligas en las filas y las medidas en las columnas
print(dfMedidasCompetitividad)
#dfi.export(dfMedidasCompetitividad,"tablaMedidas.png") #Para guardar la tabla


dfCorrelacionMedidas = pd.DataFrame(correlacionMedidas, index=labelsMedidas, columns=labelsMedidas)
dibuja_MatrizCorrelacion(dfCorrelacionMedidas)
#print(dfCorrelacionMedidas)



# i=0
# for liga in listaLigas:
#     print("Dimensión fractal (S_i) de "+liga[0][0]+": "+str(dimfractalLigas_S_i[i]))
#     print("Dimensión fractal (MDS) de "+liga[0][0]+": "+str(dimfractalLigas_MDS[i]))
#     print()
#     print("Entropía (S_i) de "+liga[0][0]+": "+str(entropiaLigas_S_i[i]))
#     print("Entropía (MDS) de "+liga[0][0]+": "+str(entropiaLigas_MDS[i]))
#     print()
#     print()
#     i=i+1
    
    




