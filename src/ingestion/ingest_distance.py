import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import time

load_dotenv()

df_municipios = pd.read_csv("C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\municipios.csv")

api_key = os.getenv('API_KEY_OPENROUTE')
api_url = 'https://api.openrouteservice.org/v2/matrix/driving-hgv'

# Definindo as capitais próximas específicas
CAPITAIS_PROXIMAS = {
    'Salvador': ['Aracaju', 'Maceió', 'Recife', 'João Pessoa', 'Vitória'],
    'Recife': ['João Pessoa', 'Maceió', 'Natal', 'Aracaju', 'Fortaleza']
}

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula a distância haversine entre dois pontos em km
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r

def get_capitais_proximas(target_city_name, df):
    """
    Obtém as capitais próximas específicas definidas para a cidade
    """
    capitais_lista = CAPITAIS_PROXIMAS.get(target_city_name, [])
    target_city = df[df['nome'] == target_city_name].iloc[0]
    target_lat = target_city['latitude']
    target_lon = target_city['longitude']
    
    capitais_proximas = []
    
    for capital_nome in capitais_lista:
        capital_row = df[df['nome'] == capital_nome]
        if not capital_row.empty:
            capital = capital_row.iloc[0]
            dist = haversine(target_lon, target_lat, capital['longitude'], capital['latitude'])
            
            capitais_proximas.append({
                'nome': capital['nome'],
                'capital': capital['capital'],
                'latitude': capital['latitude'],
                'longitude': capital['longitude'],
                'distancia_km': dist,
                'tipo': 'Capital'
            })
    
    return pd.DataFrame(capitais_proximas).sort_values('distancia_km')

def get_closest_cities_haversine(target_city, df, n_cities=10, max_distance_km=500):
    """
    Encontra as cidades mais próximas usando distância haversine
    """
    target_lat = target_city['latitude'].iloc[0]
    target_lon = target_city['longitude'].iloc[0]
    target_state = target_city['capital'].iloc[0]
    
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
                    'distancia_km': dist,
                    'tipo': 'Cidade'
                })
    
    distances_df = pd.DataFrame(distances)
    return distances_df.sort_values('distancia_km').head(n_cities)

def combine_cities_and_capitals(target_city_name, df, n_cities=10, max_distance_km=500):
    """
    Combina cidades próximas com as capitais específicas
    """
    target_city = df[df['nome'] == target_city_name]
    
    # Obter cidades próximas
    cidades_proximas = get_closest_cities_haversine(target_city, df, n_cities, max_distance_km)
    
    # Obter capitais específicas
    capitais_especificas = get_capitais_proximas(target_city_name, df)
    
    # Combinar e remover duplicatas
    combined = pd.concat([capitais_especificas, cidades_proximas], ignore_index=True)
    combined = combined.drop_duplicates(subset=['nome'], keep='first')
    
    return combined.sort_values('distancia_km')

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

def process_cities_with_api(target_city_name, cities_df, api_key):
    """
    Processa as cidades usando a API para obter distâncias rodoviárias
    """
    target_city = df_municipios[df_municipios['nome'] == target_city_name].iloc[0]
    target_coords = [target_city['longitude'], target_city['latitude']]
    
    destination_coords = []
    for _, row in cities_df.iterrows():
        destination_coords.append([row['longitude'], row['latitude']])
    
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
                if city_idx < len(cities_df):
                    results.append({
                        'nome': cities_df.iloc[city_idx]['nome'],
                        'capital': cities_df.iloc[city_idx]['capital'],
                        'latitude': cities_df.iloc[city_idx]['latitude'],
                        'longitude': cities_df.iloc[city_idx]['longitude'],
                        'tipo': cities_df.iloc[city_idx]['tipo'],
                        'distancia_aerea_km': cities_df.iloc[city_idx]['distancia_km'],
                        'distancia_rodoviaria_km': dist,
                        'tempo_viagem_min': duration / 60  
                    })
        
        time.sleep(1)
    
    return pd.DataFrame(results)

# Processamento principal
salvador = df_municipios[df_municipios['nome'] == 'Salvador'].copy()
recife = df_municipios[df_municipios['nome'] == 'Recife'].copy()

print("=== INFORMAÇÕES DAS CIDADES PRINCIPAIS ===")
print(f"Salvador: Lat {salvador['latitude'].iloc[0]}, Lon {salvador['longitude'].iloc[0]}")
print(f"Recife: Lat {recife['latitude'].iloc[0]}, Lon {recife['longitude'].iloc[0]}")
print()

# Obter capitais específicas
print("=== CAPITAIS PRÓXIMAS DE SALVADOR (ESPECÍFICAS) ===")
salvador_capitais = get_capitais_proximas('Salvador', df_municipios)
print(salvador_capitais.to_string(index=False))
print()

print("=== CAPITAIS PRÓXIMAS DE RECIFE (ESPECÍFICAS) ===")
recife_capitais = get_capitais_proximas('Recife', df_municipios)
print(recife_capitais.to_string(index=False))
print()

# Combinar capitais com cidades próximas
print("=== CIDADES E CAPITAIS PRÓXIMAS DE SALVADOR (COMBINADO) ===")
salvador_combinado = combine_cities_and_capitals('Salvador', df_municipios, n_cities=10, max_distance_km=300)
print(salvador_combinado.to_string(index=False))
print()

print("=== CIDADES E CAPITAIS PRÓXIMAS DE RECIFE (COMBINADO) ===")
recife_combinado = combine_cities_and_capitals('Recife', df_municipios, n_cities=10, max_distance_km=300)
print(recife_combinado.to_string(index=False))
print()

if api_key:
    print("=== USANDO API PARA DISTÂNCIAS RODOVIÁRIAS ===")
    
    print("Processando Salvador (com capitais específicas)...")
    salvador_com_api = process_cities_with_api('Salvador', salvador_combinado.head(15), api_key)
    print("Cidades e capitais próximas de Salvador com distâncias rodoviárias:")
    print(salvador_com_api.to_string(index=False))
    print()
    
    print("Processando Recife (com capitais específicas)...")
    recife_com_api = process_cities_with_api('Recife', recife_combinado.head(15), api_key)
    print("Cidades e capitais próximas de Recife com distâncias rodoviárias:")
    print(recife_com_api.to_string(index=False))
    
    # Salvar resultados
    salvador_com_api.to_csv('cidades_proximas_salvador_com_capitais.csv', index=False)
    recife_com_api.to_csv('cidades_proximas_recife_com_capitais.csv', index=False)
    print("\nResultados salvos em 'cidades_proximas_salvador_com_capitais.csv' e 'cidades_proximas_recife_com_capitais.csv'")
    
    # Salvar apenas as capitais específicas
    salvador_capitais_api = salvador_com_api[salvador_com_api['tipo'] == 'Capital']
    recife_capitais_api = recife_com_api[recife_com_api['tipo'] == 'Capital']
    
    salvador_capitais_api.to_csv('capitais_proximas_salvador.csv', index=False)
    recife_capitais_api.to_csv('capitais_proximas_recife.csv', index=False)
    print("Capitais específicas salvas em 'capitais_proximas_salvador.csv' e 'capitais_proximas_recife.csv'")
    
else:
    print("API key não encontrada. Usando apenas distâncias haversine.")
    
    salvador_combinado.to_csv('cidades_proximas_salvador_haversine_com_capitais.csv', index=False)
    recife_combinado.to_csv('cidades_proximas_recife_haversine_com_capitais.csv', index=False)
    print("Resultados salvos com distâncias haversine.")

print("\n=== ESTATÍSTICAS ===")
print("SALVADOR:")
print(f"  Capital mais próxima: {salvador_capitais.iloc[0]['nome']} ({salvador_capitais.iloc[0]['distancia_km']:.1f} km)")
print(f"  Cidade mais próxima geral: {salvador_combinado.iloc[0]['nome']} ({salvador_combinado.iloc[0]['distancia_km']:.1f} km)")

print("\nRECIFE:")
print(f"  Capital mais próxima: {recife_capitais.iloc[0]['nome']} ({recife_capitais.iloc[0]['distancia_km']:.1f} km)")
print(f"  Cidade mais próxima geral: {recife_combinado.iloc[0]['nome']} ({recife_combinado.iloc[0]['distancia_km']:.1f} km)")

print(f"\nTotal de capitais específicas para Salvador: {len(salvador_capitais)}")
print(f"Total de capitais específicas para Recife: {len(recife_capitais)}")