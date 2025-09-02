import pandas as pd
distancias_recife = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\cidades_proximas_recife_com_capitais.csv')
socio_recife_capitais = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_socioeconomicos_capitais_proximas_recife.csv')
socio_recife_regiao = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_socioeconomicos_recife_regiao.csv')

# Junta dados socioeconômicos das capitais e da região
dados_socio_recife = pd.concat([socio_recife_capitais, socio_recife_regiao], ignore_index=True)
dados_gerais_recife = pd.merge(distancias_recife, dados_socio_recife, on='nome', how='left')

distancias_salvador = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\cidades_proximas_salvador_com_capitais.csv')
socio_salvador_capitais = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_socioeconomicos_capitais_proximas.csv')
socio_salvador_regiao = pd.read_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_socioeconomicos_salvador_regiao.csv')

# Junta dados socioeconômicos das capitais e da região
dados_socio_salvador = pd.concat([socio_salvador_capitais, socio_salvador_regiao], ignore_index=True)
dados_gerais_salvador = pd.merge(distancias_salvador, dados_socio_salvador, on='nome', how='left')
dados_gerais_salvador.to_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_gerais_salvador.csv', index=False)
dados_gerais_recife.to_csv('C:\\Users\\vinia\\Documents\\analise_cd\\src\\data\\dados_gerais_recife.csv', index=False)
