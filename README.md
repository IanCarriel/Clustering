
Código utlizado para clusterización de la encuesta Cep 88, utilizando el modelo K means, junto a sus loadings y sus components weigths

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
