
Código utlizado para clusterización de la encuesta Cep 88, utilizando el modelo K means, junto a sus loadings y sus components weigths

Al implementar el modelo Kmeans se logró dividir la data en clusters, se utilizó para la implementación el caso de 5 clusters y de 10. Al intentar analizar el como estos clusters se explicaban con la relación a la data que los representaba se encontraron 2 problemas, el primero corresponde a que el dataframe elaborado que explicaba los pesos de los componentes en cada cluster señalaba que no existía una representación de los componentes de forma clara, varíando los pesos en muy poco valor, por lo que en algunos casos todos los componentes explicaban casi de la misma forma la creación del cluster, el segundo problema fue que al realizar el dataframe  que señala qué variables explican en mayor medida los componentes tampoco se encontró una respuesta clara por lo que en algunos casos otra vez, todas las variables explicaban en misma medida a los componentes. Este problema nos lleva a la conclusión que el modelo de la forma que lo implementamos no logra hacer una buena clusterización por lo que no tiene sentido realizar los análisis definidos en el must accomplish 2 y 3. Pero a raíz de esto se generaron nuevas direcciones para el tratamiento de los datos:

  Suponemos que la data al tener tantos variables pudo haber complicado la clusterización, es un problema del que ya se tenía en consideración, por lo mismo se realizó la división en principal components, la cuál resultó ser insuficiente.
  Una de las ideas que se barajó fue la de dividir la base en temas, lo que nos hubiese dado como resultado en sub bases, las cuales al tratarse de porciones de la base original iba a significar una menor dimensionalidad lo que hubiese facilitado la clusterización.
 No tenemos conocimiento si la aplicación de otro modelo bajo las mismas condiciones hubiese dado un resultado diferente




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Leer los datos
file_path = "c:/Users/ianca/OneDrive/Documentos/1er semestre 2024/DataScience/Workspace python/Cep88 para modelo.xlsx"  # Ajusta la ruta
data = pd.read_excel(file_path)
features=data.drop(columns=["region","zona_u_r","gse","estrato"])

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Número de componentes principales y ajuste de PCA
n_components = 86
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(X_scaled)

# Crear un DataFrame para las componentes principales
pca_df = pd.DataFrame(data=pca_features, columns=[f'Componente Principal {i+1}' for i in range(n_components)])

# Implementar K-means con 10 clusters
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pca_features)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# Agregar las etiquetas de los clusters al DataFrame original
data['Cluster'] = labels

# Guardar los resultados en un archivo Excel
output_file_path = "c:/Users/ianca/OneDrive/Documentos/1er semestre 2024/DataScience/Workspace python/resultados_cluster.xlsx"
data.to_excel(output_file_path, index=False)
print(f"Resultados guardados en {output_file_path}")


#Archivo Excel de salida que señala qué variables explican en mayor medida los componentes
columns = features.columns
loadings = pca.components_.T  # Transponer para que las filas sean las variables y las columnas los componentes
loadings_df = pd.DataFrame(loadings, columns=[f'Component_{i+1}' for i in range(pca.n_components_)], index=columns)
output_file = 'loadings_pca.xlsx'
loadings_df.to_excel(output_file)

#Archivo Excel de salida que señala los pesos de los componentes en cada cluster
cluster_centers = kmeans.cluster_centers_

component_weights = pd.DataFrame(cluster_centers, columns=[f'Componente_{i+1}' for i in range(n_components)])
component_weights.insert(0, 'Cluster', range(1, k+1))  # Agregar columna de número de cluster

# Guardar el DataFrame en un archivo Excel
output_file = 'weights_per_cluster.xlsx'
component_weights.to_excel(output_file, index=False)

print(f"Datos de pesos de componentes por cluster exportados a '{output_file}'")
