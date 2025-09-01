import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import time

load_dotenv()

# Carregar dados
df_municipios = pd.read_csv("C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\municipios.csv")

# Configuração da API
api_key = os.getenv('API_KEY_OPENROUTE')
api_url = 'https://api.openrouteservice.org/v2/matrix/driving-hgv'

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula a distância haversine entre dois pontos em km
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Raio da Terra em km
    return c * r

def get_closest_cities_haversine(target_city, df, n_cities=10, max_distance_km=500):
    """
    Encontra as cidades mais próximas usando distância haversine
    """
    target_lat = target_city['latitude'].iloc[0]
    target_lon = target_city['longitude'].iloc[0]
    target_state = target_city['capital'].iloc[0]
    
    # Calcular distâncias
    distances = []
    for idx, row in df.iterrows():
        if row['nome'] != target_city['nome'].iloc[0]:
            dist = haversine(target_lon, target_lat, row['longitude'], row['latitude'])
            if dist <= max_distance_km: 
                distances.append({
                    'nome': row['nome'],
                    'capital': row['capital'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'distancia_km': dist
                })
    
    # Ordenar por distância e retornar as n mais próximas
    distances_df = pd.DataFrame(distances)
    return distances_df.sort_values('distancia_km').head(n_cities)

def get_route_matrix(origins, destinations, api_key):
    """
    Consulta a API do OpenRouteService para obter matriz de distâncias
    """
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    
    body = {
        'locations': origins + destinations,
        'sources': list(range(len(origins))),
        'destinations': list(range(len(origins), len(origins) + len(destinations))),
        'metrics': ['distance', 'duration'],
        'units': 'km'
    }
    
    try:
        response = requests.post(api_url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na API: {e}")
        return None

def process_cities_with_api(target_city, closest_cities, api_key):
    """
    Processa as cidades mais próximas usando a API para obter distâncias rodoviárias
    """
    # Coordenadas da cidade alvo
    target_coords = [target_city['longitude'].iloc[0], target_city['latitude'].iloc[0]]
    
    # Coordenadas das cidades próximas
    destination_coords = []
    for _, row in closest_cities.iterrows():
        destination_coords.append([row['longitude'], row['latitude']])
    
    # Dividir em lotes para evitar limites da API
    batch_size = 10
    results = []
    
    for i in range(0, len(destination_coords), batch_size):
        batch = destination_coords[i:i+batch_size]
        print(f"Processando lote {i//batch_size + 1}...")
        
        matrix_result = get_route_matrix([target_coords], batch, api_key)
        
        if matrix_result and 'distances' in matrix_result:
            batch_distances = matrix_result['distances'][0]
            batch_durations = matrix_result['durations'][0]
            
            for j, (dist, duration) in enumerate(zip(batch_distances, batch_durations)):
                city_idx = i + j
                if city_idx < len(closest_cities):
                    results.append({
                        'nome': closest_cities.iloc[city_idx]['nome'],
                        'capital': closest_cities.iloc[city_idx]['capital'],
                        'latitude': closest_cities.iloc[city_idx]['latitude'],
                        'longitude': closest_cities.iloc[city_idx]['longitude'],
                        'distancia_aerea_km': closest_cities.iloc[city_idx]['distancia_km'],
                        'distancia_rodoviaria_km': dist,
                        'tempo_viagem_min': duration / 60  # Converter para minutos
                    })
        
        # Pausa para respeitar rate limits
        time.sleep(1)
    
    return pd.DataFrame(results)

# Obter informações das cidades principais
salvador = df_municipios[df_municipios['nome'] == 'Salvador'].copy()
recife = df_municipios[df_municipios['nome'] == 'Recife'].copy()

print("=== INFORMAÇÕES DAS CIDADES PRINCIPAIS ===")
print(f"Salvador: Lat {salvador['latitude'].iloc[0]}, Lon {salvador['longitude'].iloc[0]}")
print(f"Recife: Lat {recife['latitude'].iloc[0]}, Lon {recife['longitude'].iloc[0]}")
print()

# Encontrar cidades próximas usando distância haversine
print("=== CIDADES PRÓXIMAS DE SALVADOR ===")
salvador_proximas = get_closest_cities_haversine(salvador, df_municipios, n_cities=15, max_distance_km=300)
print(salvador_proximas.to_string(index=False))
print()

print("=== CIDADES PRÓXIMAS DE RECIFE ===")
recife_proximas = get_closest_cities_haversine(recife, df_municipios, n_cities=15, max_distance_km=300)
print(recife_proximas.to_string(index=False))
print()

# Usar API para obter distâncias rodoviárias (opcional)
if api_key:
    print("=== USANDO API PARA DISTÂNCIAS RODOVIÁRIAS ===")
    
    print("Processando Salvador...")
    salvador_com_api = process_cities_with_api(salvador, salvador_proximas.head(10), api_key)
    print("Top 10 cidades próximas de Salvador com distâncias rodoviárias:")
    print(salvador_com_api.to_string(index=False))
    print()
    
    print("Processando Recife...")
    recife_com_api = process_cities_with_api(recife, recife_proximas.head(10), api_key)
    print("Top 10 cidades próximas de Recife com distâncias rodoviárias:")
    print(recife_com_api.to_string(index=False))
    
    # Salvar resultados em CSV
    salvador_com_api.to_csv('cidades_proximas_salvador.csv', index=False)
    recife_com_api.to_csv('cidades_proximas_recife.csv', index=False)
    print("\nResultados salvos em 'cidades_proximas_salvador.csv' e 'cidades_proximas_recife.csv'")
else:
    print("API key não encontrada. Usando apenas distâncias haversine.")
    
    # Salvar resultados básicos
    salvador_proximas.to_csv('cidades_proximas_salvador_haversine.csv', index=False)
    recife_proximas.to_csv('cidades_proximas_recife_haversine.csv', index=False)
    print("Resultados salvos com distâncias haversine.")

# Estatísticas
print("\n=== ESTATÍSTICAS ===")
print(f"Salvador - Cidade mais próxima: {salvador_proximas.iloc[0]['nome']} ({salvador_proximas.iloc[0]['distancia_km']:.1f} km)")
print(f"Recife - Cidade mais próxima: {recife_proximas.iloc[0]['nome']} ({recife_proximas.iloc[0]['distancia_km']:.1f} km)")