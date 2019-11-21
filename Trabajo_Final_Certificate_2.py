# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:19:43 2019

@author: fariasc
"""

import os 

import numpy as np #linear algebra
import pandas as pd #dataframes
import seaborn as sb
import sklearn.preprocessing #raw data preprocessing
import sklearn.model_selection #grid search + cross validation
import sklearn.ensemble #random forest
import sklearn.tree #decision trees
import sklearn.linear_model #logistic regression + perceptron
import sklearn.svm #support vector machines
import sklearn.neighbors #k-nearest neighbors
import sklearn.neural_network #multilayer perceptron
import matplotlib.pyplot as plt #visualization
import xgboost as xgb #extreme gradient boosting


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree

##Reading the data from a csv file and saving it into a dataframe


#Seteamos directorio de trabajo

import os 
cwd= os.chdir('C:\\Users\\fariasc\\Desktop\\Fernando\\Diplomado MIT - U de Chile\\Trabajo Final Certificate')

Base_Fuga = pd.read_csv('BASEFUGA_GENERAL.csv')


################### 1. PROBLEMAS DE LA BASE DE DATOS Y MEDIDAS DE CORRECCIÓN ######################################

Base_Fuga.shape #Base de datos con 2294 filas y 20 columnas
Base_Fuga.describe()

Base_Fuga.dtypes 
Base_Fuga.info() # Información de las variables de la base de datos indicando el tipo de variables y si posee o no valores nulos.

# vemos cantidad exacta de valores nulos por variables
Base_Fuga.isnull().sum() #Encontramos 11 valores nulos en Genero, 2 en Edad, 11 en NIV_Educ, 11 en E_Civil, 3 en COD_COM, 3 en CIUDAD

# Cómo son muy pocos valores con nulos, eliminamos filas con valores nulos
Base_Fuga = Base_Fuga.dropna() # Borra 38 filas con valores nulos y la base de datos queda con 2256 filas.


#veamos valores descriptivos de todas las variables
Descrip=Base_Fuga.describe(include = "all")

# Por simple inspección vemos que la variable EDAD posee valores negativos y además edades por sobre los 100 años, que podrian ser outliers
# Generamos un gráfico de blox plot para identificar outliers
plt.boxplot(Base_Fuga['EDAD'],labels=['Edad'])

 # Vemos las filas con valores negativos. 
Aux_Edad_Neg=Base_Fuga[Base_Fuga['EDAD']<0]
#Existen 3 filas con valores de edad negativos. eliminaremos los valors, ya que por simple inspección no pareciera que fue sólo un cambio de signo. 
#Por ejemplo sale una persona con -21 años, que si asumimos que su edad en realidad es 21 años, también sale que termino sus estudios universitarios y es separado. Creemos poco vezar esa información.



 # Vemos las filas con valores por sobre los 95 años 
Aux_Edad_Sup=Base_Fuga[Base_Fuga['EDAD']>95] 
# Existen 3 filas con valores de edad superiores a los 90 años. Los eliminaremos también dado que en la realidad las entidades financieras no financian créditos a personas por sobre esa edad.

#Eliminamos filas con valores de Edad < 0
Base_Fuga = Base_Fuga[Base_Fuga['EDAD']>0] 
#Eliminamos filas con valores de Edad > 99
Base_Fuga = Base_Fuga[Base_Fuga['EDAD']<95] 

#Verificamos la eliminación de Outliers
plt.boxplot(Base_Fuga['EDAD'],labels=['Edad'])


Base_Fuga.describe() # verificamos que la base de datos queda con 2250 filas.

#De la misma forma visualizamos distrubución de la variable RENTA
plt.boxplot(Base_Fuga['RENTA'],labels=['Renta'])
# se vizualiza 




ax = sb.boxplot(x="RENTA", y="EDAD", data=Base_Fuga)


######################    2. ANALISIS DESCRIPTIVO DE VARIABLES #####################################



sb.catplot('FUGA',data=Base_Fuga,kind="count") #Distribución de la variable FUGA. Se visualiza que está balanceada
sb.catplot('GENERO',data=Base_Fuga,kind="count") #Distribución de la variable GENERO. Se visualiza que está desbalanceada. Hay mas Hombres que Mujeres
sb.catplot('GENERO',data=Base_Fuga,hue='FUGA',kind="count") #Distribucion de la variable GENERO y FUGA. Se aprecia que existe mayor FUGA en los Hombres que en las Mujeres.
sb.catplot('EDAD',data=Base_Fuga,kind="count",height=15)#Distribución de variable EDAD. Se aprecia en la distrubución que la mayor parte de las personas se concentra entre los 40 y 55 años de edad
sb.catplot('EDAD',data=Base_Fuga,hue='FUGA',kind="count",height=20) #Distribución de variable EDAD y FUGA. Se aprecia que la mayor FUGA se concentra entre los 40 y 55 años de edad.
sb.catplot('CIUDAD',data=Base_Fuga,kind="count",height=70)#Distribiución de variable CIUDAD. La mayoría de los registros está concentrado en Santiago, Arica y Concepción.
sb.catplot('COD_OFI',data=Base_Fuga[Base_Fuga['CIUDAD']=='ARICA'],hue='FUGA', kind="count",height=10) #Distribiución de variable COD_OFI y FUGA sólo en la ciudad de ARICA. Vemos en la oficina 138 de ARICA solo hay FUGA y en el resto de las oficinas (10, 22 y 37) no hay FUGA. 
sb.catplot('COD_COM',data=Base_Fuga[Base_Fuga['CIUDAD']=='SANTIAGO'],hue='FUGA', kind="count",height=20) #Distribiución de variable COD_COM y FUGA sólo en la ciudad de SANTIAGO. Vemos en la comuna 90 y 119 una mayor concentración de la FUGA que el resto de las comunas. 
sb.catplot('COD_OFI',data=Base_Fuga[(Base_Fuga['CIUDAD']=='SANTIAGO') & ((Base_Fuga['COD_COM']==90) | (Base_Fuga['COD_COM']==119)) ],hue='FUGA', kind="count",height=20) #Distribiución de variable COD_OFI y FUGA sólo en la ciudad de SANTIAGO para las Comunas = 90 y 119. Vemos en la oficina 31 hay mayor mente fuga y en la Oficina 55 sólo hay FUGA.  



########## HISTOGRAMAS ###################

Base_Fuga['RENTA'].hist() #Con el Histograma de la RENTa visualizamos que la mayor parte esta concentrada en niveles bajos de renta.
Base_Fuga[Base_Fuga['FUGA'] =='FUGA']['RENTA'].hist()
Base_Fuga[Base_Fuga['FUGA'] =='NO FUGA']['RENTA'].hist()

# Generamos una nueva variable con el LOG de la Renta para suavisar extremos y obtener mejor distribución.
Base_Fuga['RENTA_LOG']= np.log(Base_Fuga['RENTA'])

Base_Fuga['RENTA_LOG'].hist() 
Base_Fuga[Base_Fuga['FUGA'] =='FUGA']['RENTA_LOG'].hist()
Base_Fuga[Base_Fuga['FUGA'] =='NO FUGA']['RENTA_LOG'].hist()



Base_Fuga['EDAD'].hist()
Base_Fuga[Base_Fuga['FUGA'] =='FUGA']['EDAD'].hist()
Base_Fuga[Base_Fuga['FUGA'] =='NO FUGA']['EDAD'].hist()
Base_Fuga[(Base_Fuga['FUGA'] =='NO FUGA') & (Base_Fuga['GENERO'] =='M')]['EDAD'].hist()
Base_Fuga[(Base_Fuga['FUGA'] =='NO FUGA') & (Base_Fuga['GENERO'] =='F')]['EDAD'].hist()
Base_Fuga[(Base_Fuga['FUGA'] =='FUGA') & (Base_Fuga['GENERO'] =='M')]['EDAD'].hist()
Base_Fuga[(Base_Fuga['FUGA'] =='FUGA') & (Base_Fuga['GENERO'] =='F')]['EDAD'].hist()



############# GENERANDO PREPROCESAMIENTO DE VALORES EN VARIABLES ######################


# Generando Rangos de EDAD. Se consideran 4 segmentos de edad con igual cantidad de datos.
Base_Fuga['EDAD_RANGO']=pd.qcut(Base_Fuga['EDAD'],4) #rango de la edad
pd.unique(Base_Fuga['EDAD_RANGO']).tolist()
Base_Fuga['EDAD_RANGO_CAT']=pd.qcut(Base_Fuga['EDAD'],4,labels=["JOVEN","MEDIO","ADULTO", "SENIOR"]) # categoría edad
sb.catplot('EDAD_RANGO_CAT',data=Base_Fuga,hue='FUGA',kind="count")


Base_Fuga['EDAD_RANGO_2']=pd.qcut(Base_Fuga['EDAD'],5) #rango de la edad
pd.unique(Base_Fuga['EDAD_RANGO_2']).tolist()
Base_Fuga['EDAD_RANGO_CAT']=pd.qcut(Base_Fuga['EDAD'],5,labels=["JOVEN","MEDIO","ADULTO", "SENIOR","SENIOR2"]) # categoría edad
sb.catplot('EDAD_RANGO_CAT',data=Base_Fuga,hue='FUGA',kind="count")




# Generando Rangos de RENTA. Se consideran 4 rangos de edad con igual cantidad de datos.
Base_Fuga['RENTA_RANGO']=pd.qcut(Base_Fuga['RENTA'],4) #rango de la RENTA 
Base_Fuga['RENTA_RANGO_CAT']=pd.qcut(Base_Fuga['RENTA'],4,labels=["BAJA","MEDIA","ALTA", "MUY_ALTA"]) # categoría edad
pd.unique(Base_Fuga['RENTA_RANGO']).tolist()
sb.catplot('RENTA_RANGO_CAT',data=Base_Fuga,hue='FUGA',kind="count")

#Con las siguientes Variables vemos la cantidad de valores distintos con la finalidad de efectuar variables Dummies por su categoría.
pd.unique(Base_Fuga['GENERO']).tolist() # vemos valores distintos para Genero
#Vemos distribución de relacion entre estado civil y fuga
sb.catplot('GENERO',data=Base_Fuga,hue='FUGA',kind="count")
pd.unique(Base_Fuga['E_CIVIL']).tolist() # vemos valores distintos para Estado Civil
#Vemos distribución de relacion entre estado civil y fuga
sb.catplot('E_CIVIL',data=Base_Fuga,hue='FUGA',kind="count")
#vemos los valores distintos para la columna Nivel Educacional
pd.unique(Base_Fuga['NIV_EDUC']).tolist()
sb.catplot('NIV_EDUC',data=Base_Fuga,hue='FUGA',kind="count")
pd.unique(Base_Fuga['SEGURO']).tolist()
sb.catplot('SEGURO',data=Base_Fuga,hue='FUGA',kind="count")

#PREPROCESAMIENTO FUGA
#vemos los valores distintos para la columna Nivel Educacional
pd.unique(Base_Fuga['FUGA']).tolist()
# Reemplazando FUGA SI = 1, NO = 2
Base_Fuga["FUGA_COD"] = Base_Fuga["FUGA"].map({'FUGA':1,'NO FUGA':0}).astype(int)


#vemos los cambios efectuados
Base_Fuga['FUGA_COD'].head(10)


    #Para las variables que son categóricas, las transformo a dummies
#uso comando get.dummies
Base_Fuga_dummies=pd.get_dummies(Base_Fuga[['EDAD_RANGO_CAT','RENTA_RANGO_CAT','GENERO','E_CIVIL','NIV_EDUC','SEGURO']])
Base_Fuga_dummies.head()


pd.unique(Base_Fuga['CIUDAD']).tolist()


Base_Fuga["CIUDAD_COD"] = Base_Fuga["CIUDAD"].map({
 'LOS ANGELES':1001,
 'SANTIAGO':1002,
 'ANTOFAGASTA':1003,
 'ARICA':1004,
 'CONCEPCION':1005,
 'TALCAHUANO':1006,
 'TEMUCO':1007,
 'RANCAGUA':1008,
 'CHUQUICAMATA':1009,
 'PUERTO OCTAY':1010,
 'CALAMA':1011,
 'PUNTA ARENAS':1012,
 'CON-CON':1013,
 'VLLA ALEMANA':1014,
 'OSORNO':1015,
 'IQUIQUE':1016,
 'VINA DEL MAR':1017,
 'TALAGANTE':1018,
 'LA SERENA':1019,
 'SAN FELIPE':1020,
 'RENGO':1021,
 'CASTRO':1022,
 'CHILLAN':1023,
 'COLBUN':1024,
 'PENAFLOR':1025,
 'VALDIVIA':1026,
 'EL SALVADOR':1027,
 'COYHAIQUE':1028,
 'ANCUD':1029,
 'COQUIMBO':1030,
 'LONCOCHE':1031,
 'CURICO':1032,
 'LOS ANDES':1033,
 'SAN CLEMENTE':1034,
 'PALENA':1035,
 'LLANQUIHUE':1036,
 'SAN PABLO':1037,
 'QUILPUE':1038,
 'CALDERA':1039,
 'MELIPILLA':1040,
 'LLO-LLEO':1041,
 'TALCA':1042,
 'SAN FERNANDO':1043,
 'PUERTO MONTT':1044,
 'QUILLOTA':1045,
 'VALPARAISO':1046,
 'LA CALERA':1047,
 'STO DOMINGO':1048,
 'CURACAVI':1049,
 'AYSEN':1050,
 'LOS VILOS':1051,
 'PANGUIPULLI':1052,
 'CORONEL':1053,
 'COPIAPO':1054,
 'PUERTO VARAS':1055,
 'LINARES':1056,
 'SAN PEDRO':1057,
 'TOCOPILLA':1058,
 'ANGOL':1059,
 'PARRAL':1060,
 'SAN ESTEBAN':1061,
 'HIJUELAS':1062,
 'SAN ANTONIO':1063,
 'MACHALI':1064,
 'VALLENAR':1065,
 'CALBUCO':1066,
 'SANTA CRUZ':1067,
 'HUASCO':1068,
 'LOS LAGOS':1069,
 'VILLARRICA':1070,
 'LA CRUZ':1071,
 'PETORCA':1072,
 'OVALLE':1073,
 'PENCO':1074 }).astype(int) 
    

#Ahora genero una nueva Base que agrega las variables Dummies. Para lo anterior concateno la base originar con los valores Dummies
Base_Fuga_2=pd.concat([Base_Fuga,Base_Fuga_dummies],axis=1)    





# Mapping RENTA

"""
Base_Fuga.loc[ Base_Fuga['RENTA'] <= 100000, 'RENTA_Encoded'] = 1
Base_Fuga.loc[(Base_Fuga['RENTA'] >  100000) & (Base_Fuga['RENTA'] <= 250000), 'RENTA_Encoded'] = 2
Base_Fuga.loc[(Base_Fuga['RENTA'] >  250000) & (Base_Fuga['RENTA'] <= 500000), 'RENTA_Encoded'] = 3
Base_Fuga.loc[(Base_Fuga['RENTA'] >  500000) & (Base_Fuga['RENTA'] <= 1000000), 'RENTA_Encoded'] = 4
Base_Fuga.loc[(Base_Fuga['RENTA'] >  1000000) & (Base_Fuga['RENTA'] <= 2000000), 'RENTA_Encoded'] = 5
Base_Fuga.loc[ Base_Fuga['RENTA'] >  2000000, 'RENTA_Encoded'] = 6

sb.catplot('RENTA_Encoded',data=Base_Fuga,kind="count")





Base_Fuga.loc[ Base_Fuga['RENTA'] <= 250000, 'RENTA_Encoded_2'] = 1
Base_Fuga.loc[(Base_Fuga['RENTA'] >  250000) & (Base_Fuga['RENTA'] <= 500000), 'RENTA_Encoded_2'] = 2
Base_Fuga.loc[(Base_Fuga['RENTA'] >  500000) & (Base_Fuga['RENTA'] <= 1000000), 'RENTA_Encoded_2'] = 3
Base_Fuga.loc[(Base_Fuga['RENTA'] >  1000000) & (Base_Fuga['RENTA'] <= 2000000), 'RENTA_Encoded_2'] = 4
Base_Fuga.loc[(Base_Fuga['RENTA'] >  2000000) & (Base_Fuga['RENTA'] <= 5000000), 'RENTA_Encoded_2'] = 5
Base_Fuga.loc[ Base_Fuga['RENTA'] >  5000000, 'RENTA_Encoded_2'] = 6

sb.catplot('RENTA_Encoded_2',data=Base_Fuga,kind="count")

plt.boxplot(Base_Fuga['RENTA'], labels=['Renta'])

"""



colores=['red','blue']
tamanios=[60,40]

f1 = Base_Fuga_2['COD_OFI'].values.astype(int)
f2 = Base_Fuga_2['COD_COM'].values.astype(int)

asignar=[]
for index, row in Base_Fuga.iterrows():    
    asignar.append(colores[row['FUGA_COD']])

plt.scatter(f1, f2, c=asignar, s=10)
plt.axis([0,120,0,520])
plt.show()



#Base_Fuga.plot(x="COD_OFI",y="FUGA_Encoded", kind="scatter", logy=True, logx=True)



colores=['red','blue']
tamanios=[60,40]
f1 = Base_Fuga_2['EDAD'].values.astype(int)
f2 = Base_Fuga_2['RENTA'].values.astype(int)

asignar=[]
for index, row in Base_Fuga.iterrows():    
    asignar.append(colores[row['FUGA_COD']])

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([-50,130,0,9000000])
plt.show()




colores=['red','blue']
tamanios=[60,40]
f1 = Base_Fuga_2['EDAD'].values.astype(int)
f2 = Base_Fuga_2['MONTO'].values.astype(int)

asignar=[]
for index, row in Base_Fuga.iterrows():    
    asignar.append(colores[row['FUGA_COD']])

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([-50,130,0,9000000])
plt.show()



colores=['red','blue']
tamanios=[60,40]
f1 = Base_Fuga_2['RENTA'].values.astype(int)
f2 = Base_Fuga_2['MONTO'].values.astype(int)

asignar=[]
for index, row in Base_Fuga.iterrows():    
    asignar.append(colores[row['FUGA_COD']])

plt.scatter(f1, f2, c=asignar, s=30)
#plt.axis([0,8000000,0,9000000])
plt.show()

   

#veamos valores descriptivos de variables para Base_Fuga_2
Descrip2=Base_Fuga_2.describe()
    
Base_Fuga_2.dtypes 
Base_Fuga_2.info()




drop_elements = ['ID','GENERO','RENTA','EDAD','NIV_EDUC','E_CIVIL','CIUDAD','SEGURO','FUGA','EDAD_RANGO','EDAD_RANGO_CAT','RENTA_RANGO', 'RENTA_RANGO_CAT','EDAD_RANGO_2']

drop_elements = ['ID','GENERO','D_Marzo', 'D_Abril', 'D_Mayo','D_Junio', 'D_Julio','D_Agosto','D_Septiembre','RENTA','EDAD','NIV_EDUC','E_CIVIL','CIUDAD','SEGURO','FUGA','EDAD_RANGO','EDAD_RANGO_CAT','RENTA_RANGO', 'RENTA_RANGO_CAT']



Base_Fuga_2.set_index('ID',inplace=True)



drop_elements = ['GENERO','D_Marzo', 'D_Abril', 'D_Mayo','D_Junio', 'D_Julio','D_Agosto','D_Septiembre','RENTA','EDAD','NIV_EDUC','E_CIVIL','CIUDAD','SEGURO','FUGA','EDAD_RANGO','EDAD_RANGO_CAT','RENTA_RANGO', 'RENTA_RANGO_CAT']
features = Base_Fuga_2.drop(drop_elements, axis = 1)



drop_elements = ['GENERO','D_Marzo', 'D_Abril', 'D_Mayo','D_Junio', 'D_Julio','D_Agosto','D_Septiembre','RENTA','EDAD','NIV_EDUC','E_CIVIL','CIUDAD','SEGURO','FUGA','EDAD_RANGO','EDAD_RANGO_CAT','RENTA_RANGO', 'RENTA_RANGO_CAT','COD_OFI','COD_COM','CIUDAD_COD','MONTO','RENTA_LOG','SEGURO_SI','SEGURO_NO','NIV_EDUC_MED','NIV_EDUC_EUN','NIV_EDUC_BAS','E_CIVIL_SEP','RENTA_RANGO_CAT_MUY_ALTA','RENTA_RANGO_CAT_ALTA','RENTA_RANGO_CAT_MEDIA','RENTA_RANGO_CAT_BAJA','EDAD_RANGO_CAT_SENIOR','EDAD_RANGO_CAT_SENIOR']

features = Base_Fuga_2.drop(drop_elements, axis = 1)


# Matriz de Correlación
colormap = plt.cm.viridis
plt.figure(figsize=(30,30))
plt.title('Pearson Correlation of Features', y=1.05, size=30)
sb.heatmap(features.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)












#importo librerias

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale #importa libreria sklearn para escalar la data

#reescalo toda la data - necesario para un buen clustering
data_escalada=scale(features)
data_escalada=pd.DataFrame(data_escalada,index=features.index, columns=features.columns)

#clusterizo k=5
from sklearn import metrics #importa libreria para evaluar el resultado de los clusters-silueta
from sklearn.cluster import MiniBatchKMeans #importa algortimo de kmeans
K=4
modelo_clustering = MiniBatchKMeans(n_clusters=K).fit(data_escalada)
silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
ch=metrics.calinski_harabasz_score(data_escalada, modelo_clustering.labels_)

print(silueta)
print(ch)



#genero matrices vacias donde guardare los resultados de probar muchos Ks

Ks=[]
siluetas=[]
CHs=[]


#genero un for que probara desde K=2 hasta K=10
for K in range(2,15):
    modelo_clustering = MiniBatchKMeans(n_clusters=K).fit(data_escalada)
    silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
    ch=metrics.calinski_harabasz_score(data_escalada, modelo_clustering.labels_)
    
    #voy guardando el K, silueta y CH
    Ks.append(K)
    siluetas.append(silueta)
    CHs.append(ch)
    #muestro resultado en cada iteración
    print(K,silueta,ch)
    

#guardo los resultadods como DFs
DF_Ks=pd.DataFrame({'Ks': Ks})
DF_Siluetas=pd.DataFrame({'Siluetas': siluetas})
DF_CH=pd.DataFrame({'Calinsky-Harabaz': CHs})

#concateno en un sólo DF resultado
resultado=pd.concat([DF_Ks,DF_Siluetas,DF_CH],axis=1)

#muestro gráficamente en un gráfico de codo o elbow los resultados
resultado.plot(x='Ks', y='Siluetas')
resultado.plot(x='Ks',y='Calinsky-Harabaz')



#me quedo con K=7
K=4
from sklearn import metrics #importa libreria para evaluar el resultado de los clusters-silueta
from sklearn.cluster import MiniBatchKMeans #importa algortimo de kmeans
modelo_clustering = MiniBatchKMeans(n_clusters=K).fit(data_escalada)
     
#agrego etiquetas clusters a la data original
etiquetas=modelo_clustering.predict(data_escalada)
df_eti=pd.DataFrame(data=etiquetas,columns=['cluster'],index=data_escalada.index)

#veamos las etiquetas
c_data=pd.concat([df_eti,features],axis=1)
c_data.head(10)

#veamos las etiquetas en la data escalada
csdata=pd.concat([df_eti,data_escalada],axis=1) 
csdata.head(10)

''' creo estadistica descriptiva para cada cluster '''

stats_per_cluster=pd.DataFrame()
for cl in range(0,5):
    stcl=c_data[c_data['cluster']==cl]
    stcl=stcl.describe()
    stcl['cluster']=cl
    stats_per_cluster=pd.concat([stats_per_cluster,stcl])
 
#veamos uno de ejemplo
stats_per_cluster[stats_per_cluster['cluster']==1]

#centroides en la data sin escalar
centroides1=c_data.groupby('cluster').mean()
centroides1=centroides1.reset_index(drop=False)
centroides1

#centroides en la data escalada
centroides2=csdata.groupby('cluster').mean()
centroides2=centroides2.reset_index(drop=False)

centroides2

#grafico los centroides de la data escalada para que sea más fácil interpretar:

from pandas.plotting import parallel_coordinates

fig = plt.figure(figsize=(15, 5))
title = fig.suptitle("Centroide por Cluster - Data Escalada", fontsize=18)
fig.subplots_adjust(top=0.93, wspace=0)
parallel_coordinates(centroides2, 'cluster',colormap='viridis')

plt.xticks(rotation=90)
plt.show()

#grafico los centroides en la data original:
import numpy as np
df=centroides1.copy()
cols = df.loc[:, df.columns != 'cluster'].columns.tolist()
df['cluster']=df['cluster'].astype('category')

from matplotlib import ticker


x = [i for i, _ in enumerate(cols)]
colours = ['red', 'green', 'blue', 'yellow']

# create dict of categories: colours
colours = {df['cluster'].cat.categories[i]: colours[i] for i, _ in enumerate(df['cluster'].cat.categories)}

# Create (X-1) subplots along x axis
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(20,5))

# Get min, max and range for each column
# Normalize the data for each column
min_max_range = {}
for col in cols:
    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

# Plot each row
for i, ax in enumerate(axes):
    for idx in df.index:
        mpg_category = df.loc[idx, 'cluster']
        ax.plot(x, df.loc[idx, cols], colours[mpg_category])
    ax.set_xlim([x[i], x[i+1]])
    
# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, ticks):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    step = val_range / float(ticks-1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)

for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]])
    

# Move the final axis' ticks to the right-hand side
ax = plt.twinx(axes[-1])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]])


# Remove space between subplots
plt.subplots_adjust(wspace=0)

# Add legend to plot
plt.legend(
    [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df['cluster'].cat.categories],
    df['cluster'].cat.categories,
    bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.)
    
plt.title("Centroides por Cluster - Data Original")

plt.show()













USAR K_MEANS para SEGMENTAR y que el silueta sea mayor a 0,25



