import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuração da Página ---
st.set_page_config(
    page_title="Magalu - Recife x Salvador",
    page_icon="🚚",
    layout="wide"
)

# --- Funções ---

@st.cache_data
def carregar_dados():
    """Carrega os dados de Recife e Salvador e garante que os tipos de dados estão corretos."""
    try:
        recife = pd.read_csv('src/data/dados_gerais_recife.csv')
        salvador = pd.read_csv('src/data/dados_gerais_salvador.csv')

        colunas_numericas = ['populacao_2025', 'pib_per_capita', 'empregos_formais', 'tempo_viagem_min']
        for df in [recife, salvador]:
            for col in colunas_numericas:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return recife, salvador
        
    except FileNotFoundError:
        st.error("Erro: Arquivos de dados não encontrados. Verifique os caminhos 'src/data/dados_gerais_recife.csv' e 'src/data/dados_gerais_salvador.csv'.")
        return None, None

def plotar_mapa_comparativo(recife_df, salvador_df):
    """Cria e exibe dois mapas, um para cada cidade, lado a lado."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alcance Geográfico a partir de Recife")
        fig_rec = px.scatter_mapbox(
            recife_df, lat='latitude', lon='longitude', color='tipo_x',
            size='populacao_2025', hover_name='nome', zoom=4.5, mapbox_style='carto-positron'
        )
        fig_rec.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_rec, use_container_width=True)

    with col2:
        st.subheader("Alcance Geográfico a partir de Salvador")
        fig_sal = px.scatter_mapbox(
            salvador_df, lat='latitude', lon='longitude', color='tipo_x',
            size='populacao_2025', hover_name='nome', zoom=4.5, mapbox_style='carto-positron'
        )
        fig_sal.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_sal, use_container_width=True)

def plotar_tempo_capitais_comparativo(recife_df, salvador_df):
    """Cria um gráfico de barras agrupado comparando o tempo de viagem para as capitais."""
    st.subheader("Tempo de Viagem para as Capitais (Recife vs. Salvador)")
    
    # Prepara os dados
    recife_capitais = recife_df[recife_df['capital_x'] == 1][['nome', 'tempo_viagem_min']].copy()
    recife_capitais['Origem'] = 'Recife'
    
    salvador_capitais = salvador_df[salvador_df['capital_x'] == 1][['nome', 'tempo_viagem_min']].copy()
    salvador_capitais['Origem'] = 'Salvador'
    
    # Combina os dataframes para plotagem
    df_comparativo = pd.concat([recife_capitais, salvador_capitais])
    
    fig = px.bar(
        df_comparativo.sort_values(['nome', 'Origem']),
        x='nome',
        y='tempo_viagem_min',
        color='Origem',
        barmode='group', # Essencial para agrupar as barras
        title='Comparativo de Tempo de Viagem para Capitais do Nordeste',
        labels={'tempo_viagem_min': 'Tempo de Viagem (minutos)', 'nome': 'Capital de Destino'},
        text_auto='.0f',
        color_discrete_map={'Recife': '#00BFFF', 'Salvador': '#FF6F00'}
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


# --- Interface Principal ---

st.title('🚚 Análise Comparativa: Novo CD Magalu no Nordeste')

st.markdown('''
### Cenário
A decisão de abrir um novo Centro de Distribuição no Nordeste exige uma análise criteriosa entre **Recife** e **Salvador**. Este painel compara diretamente as duas cidades em fatores-chave como **logística**, **alcance de mercado** e **potencial econômico** para apoiar uma decisão estratégica baseada em dados.
''')

recife_df, salvador_df = carregar_dados()

if recife_df is not None and salvador_df is not None:
    
    # --- 1. Resumo dos Indicadores Chave ---
    st.header("Resumo Estratégico (Dashboard)", divider='orange')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recife (PE)")
        tempo_medio_rec = recife_df[recife_df['capital_x'] == 1]['tempo_viagem_min'].mean()
        pop_total_rec = recife_df['populacao_2025'].sum()
        pib_medio_rec = recife_df['pib_per_capita'].mean()
        
        st.metric(label="Tempo Médio para Capitais (min)", value=f"{tempo_medio_rec:.0f}")
        st.metric(label="População Total na Área de Alcance", value=f"{pop_total_rec/1e6:.2f} milhões")
        st.metric(label="PIB per Capita Médio da Região", value=f"R$ {pib_medio_rec:,.2f}")

    with col2:
        st.subheader("Salvador (BA)")
        tempo_medio_sal = salvador_df[salvador_df['capital_x'] == 1]['tempo_viagem_min'].mean()
        pop_total_sal = salvador_df['populacao_2025'].sum()
        pib_medio_sal = salvador_df['pib_per_capita'].mean()

        st.metric(label="Tempo Médio para Capitais (min)", value=f"{tempo_medio_sal:.0f}", delta=f"{tempo_medio_sal - tempo_medio_rec:.0f} min (vs Recife)")
        st.metric(label="População Total na Área de Alcance", value=f"{pop_total_sal/1e6:.2f} milhões", delta=f"{(pop_total_sal - pop_total_rec)/1e6:.2f} M (vs Recife)")
        st.metric(label="PIB per Capita Médio da Região", value=f"R$ {pib_medio_sal:,.2f}", delta=f"R$ {pib_medio_sal - pib_medio_rec:,.2f} (vs Recife)")
    
    # --- 2. Análise Logística Comparativa ---
    st.header("Análise Logística e Geográfica", divider='orange')
    plotar_tempo_capitais_comparativo(recife_df, salvador_df)
    plotar_mapa_comparativo(recife_df, salvador_df)

    # --- 3. Conclusão da Análise ---
    st.header("Conclusão da Análise", divider='orange')
    st.markdown("""
    A análise dos dados revela um trade-off claro entre as duas cidades, cada uma com vantagens estratégicas distintas.

    #### Pontos Fortes de Recife (PE):
    - **📍 Centralidade Geográfica:** Recife demonstra ser um hub logístico mais central para a região Nordeste como um todo. Apresenta tempos de viagem significativamente menores para capitais ao norte como **João Pessoa, Natal e Fortaleza**.
    - **🚀 Agilidade para Mercados do Norte:** Para uma estratégia que prioriza a velocidade de entrega nos estados de Pernambuco, Paraíba, Rio Grande do Norte e Ceará, Recife é a escolha superior.

    #### Pontos Fortes de Salvador (BA):
    - **📈 Maior Mercado Imediato:** A área de influência direta de Salvador, abrangendo o estado da Bahia, possui uma população e um potencial de consumo (PIB) ligeiramente superior no agregado.
    - **🔗 Conexão com o Sudeste:** Por sua localização mais ao sul, Salvador oferece uma vantagem logística para mercados como **Aracaju (SE)** e tem uma conexão rodoviária mais curta com a região Sudeste do país.

    ### Recomendação Estratégica:
    - Se o objetivo principal do novo CD é **otimizar a logística e reduzir o tempo de entrega para a maior quantidade de capitais nordestinas**, **Recife** é a opção mais indicada devido à sua centralidade.
    - Se a estratégia é focar no **maior mercado consumidor da Bahia** e, ao mesmo tempo, criar um elo mais forte com o Sudeste, **Salvador** se torna a escolha mais atraente.
    """)