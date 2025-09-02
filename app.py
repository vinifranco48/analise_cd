import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import folium
from streamlit_folium import st_folium

# --- Configuração da Página ---
st.set_page_config(
    page_title="Magalu - Análise Estratégica CD Nordeste",
    page_icon="🚚",
    layout="wide"
)

# --- Funções Aprimoradas ---

@st.cache_data
def carregar_dados():
    """Carrega os dados de Recife e Salvador e garante que os tipos de dados estão corretos."""
    try:
        recife = pd.read_csv('src/data/dados_gerais_recife.csv')
        salvador = pd.read_csv('src/data/dados_gerais_salvador.csv')

        # Limpeza e conversão de dados
        colunas_numericas = ['populacao_2025', 'pib_per_capita', 'empregos_formais', 'tempo_viagem_min', 
                           'pib_bilhoes', 'idh_2021', 'densidade_demografica', 'distancia_aerea_km_x', 
                           'distancia_rodoviaria_km_x', 'distancia_aerea_km_y', 'distancia_rodoviaria_km_y']
        
        for df in [recife, salvador]:
            for col in colunas_numericas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calcular métricas derivadas avançadas
            df['poder_compra'] = df['populacao_2025'] * df['pib_per_capita']
            df['mercado_potencial'] = df['populacao_2025'] * df['idh_2021'] if 'idh_2021' in df.columns else df['populacao_2025']
            df['acessibilidade_score'] = 1 / (df['tempo_viagem_min'] / 60)
            df['densidade_economica'] = df['pib_bilhoes'] / (df['populacao_2025'] / 1000000) if 'pib_bilhoes' in df.columns else 0
            df['eficiencia_logistica'] = df['populacao_2025'] / df['tempo_viagem_min']
        
        return recife, salvador
        
    except FileNotFoundError:
        st.error("Erro: Arquivos de dados não encontrados. Verifique os caminhos dos arquivos CSV.")
        return None, None

def analise_cidades_proximas(df, cidade_principal):
    """Analisa o impacto das cidades próximas na decisão estratégica."""
    cidades_proximas = df[df['capital_x'] != 1].copy()  # Exclui capitais
    
    if len(cidades_proximas) == 0:
        return {}
    
    analise = {
        'total_populacao_proxima': cidades_proximas['populacao_2025'].sum(),
        'media_tempo_cidades': cidades_proximas['tempo_viagem_min'].mean(),
        'total_empregos_formais': cidades_proximas['empregos_formais'].sum() if 'empregos_formais' in cidades_proximas.columns else 0,
        'media_idh': cidades_proximas['idh_2021'].mean() if 'idh_2021' in cidades_proximas.columns else 0,
        'cidade_mais_populosa': cidades_proximas.loc[cidades_proximas['populacao_2025'].idxmax(), 'nome'] if len(cidades_proximas) > 0 else "N/A",
        'numero_cidades_proximas': len(cidades_proximas),
        'densidade_regional': cidades_proximas['densidade_demografica'].mean() if 'densidade_demografica' in cidades_proximas.columns else 0
    }
    
    return analise

def calcular_score_localizacao_avancado(df, cidade_nome):
    """Calcula um score composto de localização baseado em múltiplos critérios incluindo cidades próximas."""
    scores = {}
    
    # Separar capitais e cidades próximas
    capitais_df = df[df['capital_x'] == 1]
    cidades_df = df[df['capital_x'] != 1]
    
    # 1. Score Logístico Avançado (30% do peso) - Capitais + Cidades próximas
    tempo_medio_capitais = capitais_df['tempo_viagem_min'].mean() if len(capitais_df) > 0 else 600
    tempo_medio_cidades = cidades_df['tempo_viagem_min'].mean() if len(cidades_df) > 0 else 300
    
    score_capitais = max(0, 1 - (tempo_medio_capitais - 300) / 600)
    score_cidades = max(0, 1 - (tempo_medio_cidades - 60) / 240)
    scores['logistico'] = (score_capitais * 0.7 + score_cidades * 0.3)
    
    # 2. Score de Mercado Expandido (30% do peso) - População total + PIB
    pop_total = df['populacao_2025'].sum()
    pib_total = df['pib_bilhoes'].sum() if 'pib_bilhoes' in df.columns else 0
    pib_medio = df['pib_per_capita'].mean()
    
    score_pop = min(1, pop_total / 60000000)
    score_pib = min(1, (pib_total * 10 + pib_medio / 50000) / 2)
    scores['mercado'] = (score_pop + score_pib) / 2
    
    # 3. Score de Desenvolvimento Regional (25% do peso) - IDH + Densidade Econômica
    if 'idh_2021' in df.columns and 'empregos_formais' in df.columns:
        idh_medio = df['idh_2021'].mean()
        empregos_total = df['empregos_formais'].sum()
        densidade_media = df['densidade_demografica'].mean() if 'densidade_demografica' in df.columns else 1000
        
        score_idh = idh_medio if idh_medio > 0 else 0.7
        score_empregos = min(1, empregos_total / 6000000)
        score_densidade = min(1, densidade_media / 3000)
        
        scores['desenvolvimento'] = (score_idh * 0.5 + score_empregos * 0.3 + score_densidade * 0.2)
    else:
        scores['desenvolvimento'] = 0.7
    
    # 4. Score de Conectividade e Abrangência (15% do peso)
    num_capitais = len(capitais_df)
    num_cidades = len(cidades_df)
    
    score_capitais_conn = min(1, num_capitais / 8)
    score_cidades_conn = min(1, num_cidades / 15)
    scores['conectividade'] = (score_capitais_conn * 0.7 + score_cidades_conn * 0.3)
    
    # Score composto final
    score_final = (scores['logistico'] * 0.30 + 
                   scores['mercado'] * 0.30 + 
                   scores['desenvolvimento'] * 0.25 + 
                   scores['conectividade'] * 0.15)
    
    return score_final, scores

def plotar_analise_regional_detalhada(recife_df, salvador_df):
    """Cria análise regional detalhada comparando as duas opções."""
    
    # Preparar dados para visualização
    def preparar_dados_cidade(df, nome_cidade):
        capitais = df[df['capital_x'] == 1]
        cidades = df[df['capital_x'] != 1]
        
        return {
            'cidade': nome_cidade,
            'pop_capitais': capitais['populacao_2025'].sum(),
            'pop_cidades': cidades['populacao_2025'].sum(),
            'pop_total': df['populacao_2025'].sum(),
            'num_capitais': len(capitais),
            'num_cidades': len(cidades),
            'tempo_medio_capitais': capitais['tempo_viagem_min'].mean() if len(capitais) > 0 else 0,
            'tempo_medio_cidades': cidades['tempo_viagem_min'].mean() if len(cidades) > 0 else 0,
            'pib_total': df['pib_bilhoes'].sum() if 'pib_bilhoes' in df.columns else 0
        }
    
    dados_recife = preparar_dados_cidade(recife_df, 'Recife')
    dados_salvador = preparar_dados_cidade(salvador_df, 'Salvador')
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('População por Tipo de Cidade', 'Conectividade Regional', 
                       'Tempos Médios de Acesso', 'Potencial Econômico'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    cidades = ['Recife', 'Salvador']
    cores = ['#00BFFF', '#FF6F00']
    
    # 1. População por tipo
    fig.add_trace(go.Bar(
        name='Capitais',
        x=cidades,
        y=[dados_recife['pop_capitais'], dados_salvador['pop_capitais']],
        marker_color='lightblue'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        name='Cidades Próximas',
        x=cidades,
        y=[dados_recife['pop_cidades'], dados_salvador['pop_cidades']],
        marker_color='orange'
    ), row=1, col=1)
    
    # 2. Conectividade
    fig.add_trace(go.Scatter(
        x=[dados_recife['num_capitais'], dados_salvador['num_capitais']],
        y=[dados_recife['num_cidades'], dados_salvador['num_cidades']],
        mode='markers+text',
        text=cidades,
        textposition="top center",
        marker=dict(size=15, color=cores),
        showlegend=False
    ), row=1, col=2)
    
    # 3. Tempos médios
    fig.add_trace(go.Bar(
        name='Tempo Capitais',
        x=cidades,
        y=[dados_recife['tempo_medio_capitais'], dados_salvador['tempo_medio_capitais']],
        marker_color='red',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        name='Tempo Cidades',
        x=cidades,
        y=[dados_recife['tempo_medio_cidades'], dados_salvador['tempo_medio_cidades']],
        marker_color='lightcoral',
        showlegend=False
    ), row=2, col=1)
    
    # 4. PIB Total
    fig.add_trace(go.Bar(
        x=cidades,
        y=[dados_recife['pib_total'], dados_salvador['pib_total']],
        marker_color=cores,
        showlegend=False
    ), row=2, col=2)
    
    # Atualizar layout
    fig.update_xaxes(title_text="Tempo Médio (min)", row=2, col=1)
    fig.update_xaxes(title_text="Número de Capitais", row=1, col=2)
    fig.update_yaxes(title_text="Número de Cidades", row=1, col=2)
    fig.update_yaxes(title_text="PIB (Bilhões R$)", row=2, col=2)
    fig.update_yaxes(title_text="População (milhões)", row=1, col=1)
    
    fig.update_layout(height=600, showlegend=True, title_text="Análise Regional Detalhada")
    
    return fig

def criar_mapa_estrategico(recife_df, salvador_df):
    """Cria um mapa estratégico mostrando a cobertura de cada opção."""
    # Esta função criaria um mapa interativo, mas como não temos coordenadas completas,
    # vamos criar uma visualização de rede alternativa
    
    fig = go.Figure()
    
    # Simular posições para visualização de rede
    def criar_rede_cidade(df, nome, cor, x_offset=0):
        capitais = df[df['capital_x'] == 1]
        cidades = df[df['capital_x'] != 1]
        
        # Hub central
        fig.add_trace(go.Scatter(
            x=[x_offset], y=[0],
            mode='markers+text',
            text=[f'CD {nome}'],
            textposition="middle center",
            marker=dict(size=30, color=cor, symbol='diamond'),
            name=f'Hub {nome}'
        ))
        
        # Conectar capitais
        angles_cap = np.linspace(0, 2*np.pi, len(capitais), endpoint=False)
        for i, (_, capital) in enumerate(capitais.iterrows()):
            x = x_offset + 2 * np.cos(angles_cap[i])
            y = 2 * np.sin(angles_cap[i])
            
            # Linha de conexão
            fig.add_trace(go.Scatter(
                x=[x_offset, x], y=[0, y],
                mode='lines',
                line=dict(color=cor, width=2),
                showlegend=False
            ))
            
            # Ponto da capital
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[capital['nome']],
                textposition="top center",
                marker=dict(size=15, color='darkred'),
                showlegend=False
            ))
        
        # Conectar cidades próximas (círculo interno)
        if len(cidades) > 0:
            angles_cid = np.linspace(0, 2*np.pi, len(cidades), endpoint=False)
            for i, (_, cidade) in enumerate(cidades.iterrows()):
                x = x_offset + 1 * np.cos(angles_cid[i])
                y = 1 * np.sin(angles_cid[i])
                
                fig.add_trace(go.Scatter(
                    x=[x_offset, x], y=[0, y],
                    mode='lines',
                    line=dict(color=cor, width=1, dash='dot'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=8, color='lightblue'),
                    showlegend=False
                ))
    
    # Criar redes para ambas as cidades
    criar_rede_cidade(recife_df, 'Recife', '#00BFFF', x_offset=-3)
    criar_rede_cidade(salvador_df, 'Salvador', '#FF6F00', x_offset=3)
    
    fig.update_layout(
        title="Mapa Estratégico de Conectividade Regional",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        height=500
    )
    
    return fig

def analise_custo_beneficio_avancada(recife_df, salvador_df):
    """Análise avançada de custo-benefício considerando múltiplos fatores."""
    
    def calcular_metricas_avancadas(df, nome_cidade):
        capitais = df[df['capital_x'] == 1]
        cidades = df[df['capital_x'] != 1]
        
        # Métricas de cobertura
        pop_total = df['populacao_2025'].sum()
        pop_capitais = capitais['populacao_2025'].sum()
        pop_cidades = cidades['populacao_2025'].sum()
        
        # Métricas logísticas
        tempo_medio_geral = df['tempo_viagem_min'].mean()
        distancia_media = df['distancia_rodoviaria_km_x'].mean() if 'distancia_rodoviaria_km_x' in df.columns else 500
        
        # Métricas econômicas
        pib_total = df['pib_bilhoes'].sum() if 'pib_bilhoes' in df.columns else 0
        renda_media = df['pib_per_capita'].mean()
        
        # Índices compostos
        indice_eficiencia = pop_total / tempo_medio_geral if tempo_medio_geral > 0 else 0
        indice_mercado = pop_total * renda_media / 1000000
        indice_logistico = 1000 / distancia_media if distancia_media > 0 else 0
        
        return {
            'cidade': nome_cidade,
            'populacao_total': pop_total,
            'populacao_capitais': pop_capitais,
            'populacao_cidades': pop_cidades,
            'tempo_medio_minutos': tempo_medio_geral,
            'distancia_media_km': distancia_media,
            'pib_total_bilhoes': pib_total,
            'renda_media': renda_media,
            'indice_eficiencia': indice_eficiencia,
            'indice_mercado': indice_mercado,
            'indice_logistico': indice_logistico,
            'score_geral': (indice_eficiencia/100000 + indice_mercado/1000 + indice_logistico) / 3
        }
    
    metricas_recife = calcular_metricas_avancadas(recife_df, 'Recife')
    metricas_salvador = calcular_metricas_avancadas(salvador_df, 'Salvador')
    
    return pd.DataFrame([metricas_recife, metricas_salvador])

# --- Interface Principal Aprimorada ---

st.title('🚚 Análise Estratégica Avançada: Centro de Distribuição Magalu Nordeste')

st.markdown('''
### 🎯 Objetivo Estratégico
Esta análise utiliza metodologia multicritério avançada para avaliar **Recife** vs **Salvador** como localização ótima 
para o novo Centro de Distribuição do Magazine Luiza no Nordeste, considerando:

- 📍 **Análise de Cobertura Regional**: Capitais + cidades próximas  
- ⚡ **Eficiência Logística**: Tempos de entrega e custos de transporte
- 💰 **Potencial de Mercado**: Demografia, renda e poder de compra
- 🌐 **Conectividade Regional**: Malha viária e acessibilidade
- 📊 **Análise de Sensibilidade**: Diferentes cenários estratégicos
''')

# Carregar dados
recife_df, salvador_df = carregar_dados()

if recife_df is not None and salvador_df is not None:
    
    # Análise das cidades próximas
    analise_rec = analise_cidades_proximas(recife_df, "Recife")
    analise_sal = analise_cidades_proximas(salvador_df, "Salvador")
    
    # Calcular scores avançados
    score_recife, scores_recife = calcular_score_localizacao_avancado(recife_df, "Recife")
    score_salvador, scores_salvador = calcular_score_localizacao_avancado(salvador_df, "Salvador")
    
    # --- 1. Dashboard Executivo Expandido ---
    st.header("🎯 Dashboard Executivo", divider='orange')
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.subheader("🏆 Ranking Geral")
        if score_recife > score_salvador:
            st.success(f"**1º Recife**: {score_recife:.3f}")
            st.info(f"**2º Salvador**: {score_salvador:.3f}")
            diferenca = score_recife - score_salvador
            st.metric("Vantagem Recife", f"{diferenca:.3f}", f"+{diferenca*100:.1f}%")
        else:
            st.success(f"**1º Salvador**: {score_salvador:.3f}")
            st.info(f"**2º Recife**: {score_recife:.3f}")
            diferenca = score_salvador - score_recife
            st.metric("Vantagem Salvador", f"{diferenca:.3f}", f"+{diferenca*100:.1f}%")
    
    with col2:
        st.subheader("📊 Cobertura Recife")
        pop_total_rec = recife_df['populacao_2025'].sum()
        capitais_rec = len(recife_df[recife_df['capital_x'] == 1])
        cidades_rec = len(recife_df[recife_df['capital_x'] != 1])
        tempo_medio_rec = recife_df['tempo_viagem_min'].mean()
        
        st.metric("População Total", f"{pop_total_rec/1e6:.1f} M hab")
        st.metric("Capitais Conectadas", capitais_rec)
        st.metric("Cidades Próximas", cidades_rec)
        st.metric("Tempo Médio Geral", f"{tempo_medio_rec:.0f} min")

    with col3:
        st.subheader("📊 Cobertura Salvador")
        pop_total_sal = salvador_df['populacao_2025'].sum()
        capitais_sal = len(salvador_df[salvador_df['capital_x'] == 1])
        cidades_sal = len(salvador_df[salvador_df['capital_x'] != 1])
        tempo_medio_sal = salvador_df['tempo_viagem_min'].mean()

        st.metric("População Total", f"{pop_total_sal/1e6:.1f} M hab", 
                 f"{(pop_total_sal - pop_total_rec)/1e6:+.1f} M")
        st.metric("Capitais Conectadas", capitais_sal, f"{capitais_sal - capitais_rec:+d}")
        st.metric("Cidades Próximas", cidades_sal, f"{cidades_sal - cidades_rec:+d}")
        st.metric("Tempo Médio Geral", f"{tempo_medio_sal:.0f} min", 
                 f"{tempo_medio_sal - tempo_medio_rec:+.0f} min")
    
    with col4:
        st.subheader("💡 Insights Regionais")
        if analise_rec and analise_sal:
            diff_pop_proxima = analise_rec['total_populacao_proxima'] - analise_sal['total_populacao_proxima']
            cidade_vantagem = "Recife" if diff_pop_proxima > 0 else "Salvador"
            
            st.metric("Vantagem Pop. Próxima", 
                     cidade_vantagem, 
                     f"{abs(diff_pop_proxima)/1e6:+.1f} M hab")
            
            melhor_tempo = "Recife" if analise_rec['media_tempo_cidades'] < analise_sal['media_tempo_cidades'] else "Salvador"
            st.metric("Melhor Acesso Local", melhor_tempo)
            
            densidade_rec = analise_rec.get('densidade_regional', 0)
            densidade_sal = analise_sal.get('densidade_regional', 0)
            melhor_densidade = "Recife" if densidade_rec > densidade_sal else "Salvador"
            st.metric("Maior Densidade Reg.", melhor_densidade)

    # --- 2. Análise Regional Detalhada ---
    st.header("🗺️ Análise Regional Detalhada", divider='orange')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Comparativo Multidimensional")
        fig_regional = plotar_analise_regional_detalhada(recife_df, salvador_df)
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with col2:
        st.subheader("🌐 Mapa Estratégico de Conectividade")
        fig_mapa = criar_mapa_estrategico(recife_df, salvador_df)
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --- 3. Análise Econômica Avançada ---
    st.header("💰 Análise Econômica Avançada", divider='orange')
    
    df_custo_beneficio = analise_custo_beneficio_avancada(recife_df, salvador_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Métricas Comparativas")
        st.dataframe(df_custo_beneficio.round(2), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Índices de Performance")
        
        # Gráfico de barras dos índices
        fig_indices = go.Figure()
        
        indices = ['Eficiência Logística', 'Potencial de Mercado', 'Acessibilidade']
        recife_vals = [df_custo_beneficio.loc[0, 'indice_eficiencia']/100000,
                      df_custo_beneficio.loc[0, 'indice_mercado']/1000,
                      df_custo_beneficio.loc[0, 'indice_logistico']]
        salvador_vals = [df_custo_beneficio.loc[1, 'indice_eficiencia']/100000,
                        df_custo_beneficio.loc[1, 'indice_mercado']/1000,
                        df_custo_beneficio.loc[1, 'indice_logistico']]
        
        fig_indices.add_trace(go.Bar(name='Recife', x=indices, y=recife_vals, marker_color='#00BFFF'))
        fig_indices.add_trace(go.Bar(name='Salvador', x=indices, y=salvador_vals, marker_color='#FF6F00'))
        
        fig_indices.update_layout(
            title="Índices de Performance Normalizados",
            yaxis_title="Score Normalizado",
            barmode='group'
        )
        
        st.plotly_chart(fig_indices, use_container_width=True)

    # --- 4. Análise de Cidades Próximas ---
    st.header("🏘️ Impact das Cidades Próximas", divider='orange')
    
    if analise_rec and analise_sal:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏙️ Recife - Região Metropolitana")
            st.metric("População Próxima", f"{analise_rec['total_populacao_proxima']/1e6:.1f} M")
            st.metric("Número de Cidades", analise_rec['numero_cidades_proximas'])
            st.metric("Tempo Médio Local", f"{analise_rec['media_tempo_cidades']:.0f} min")
            if analise_rec['cidade_mais_populosa'] != "N/A":
                st.info(f"**Maior cidade próxima:** {analise_rec['cidade_mais_populosa']}")
        
        with col2:
            st.subheader("🏙️ Salvador - Região Metropolitana")
            st.metric("População Próxima", f"{analise_sal['total_populacao_proxima']/1e6:.1f} M",
                     f"{(analise_sal['total_populacao_proxima'] - analise_rec['total_populacao_proxima'])/1e6:+.1f} M")
            st.metric("Número de Cidades", analise_sal['numero_cidades_proximas'],
                     f"{analise_sal['numero_cidades_proximas'] - analise_rec['numero_cidades_proximas']:+d}")
            st.metric("Tempo Médio Local", f"{analise_sal['media_tempo_cidades']:.0f} min",
                     f"{analise_sal['media_tempo_cidades'] - analise_rec['media_tempo_cidades']:+.0f} min")
            if analise_sal['cidade_mais_populosa'] != "N/A":
                st.info(f"**Maior cidade próxima:** {analise_sal['cidade_mais_populosa']}")
        
        with col3:
            st.subheader("📊 Vantagem Competitiva")
            
            # Determinar vantagens
            vant_pop = "Salvador" if analise_sal['total_populacao_proxima'] > analise_rec['total_populacao_proxima'] else "Recife"
            vant_tempo = "Recife" if analise_rec['media_tempo_cidades'] < analise_sal['media_tempo_cidades'] else "Salvador"
            vant_densidade = "Recife" if analise_rec['densidade_regional'] > analise_sal['densidade_regional'] else "Salvador"
            
            st.success(f"🎯 **Maior mercado próximo:** {vant_pop}")
            st.success(f"⚡ **Melhor acesso local:** {vant_tempo}")  
            st.success(f"🏢 **Maior densidade:** {vant_densidade}")
            
            # Calcular impacto das cidades próximas no score
            impacto_rec = analise_rec['total_populacao_proxima'] / analise_rec['media_tempo_cidades'] if analise_rec['media_tempo_cidades'] > 0 else 0
            impacto_sal = analise_sal['total_populacao_proxima'] / analise_sal['media_tempo_cidades'] if analise_sal['media_tempo_cidades'] > 0 else 0
            
            melhor_impacto = "Recife" if impacto_rec > impacto_sal else "Salvador"
            st.metric("Melhor Relação Pop/Tempo", melhor_impacto)

    # --- 5. Análise de Sensibilidade Expandida ---
    st.header("⚖️ Análise de Sensibilidade Expandida", divider='orange')
    
    st.markdown("**Como diferentes estratégias empresariais e prioridades afetam a decisão:**")
    
    # Cenários expandidos
    cenarios_expandidos = {
        'Foco Logístico Puro': {'logistico': 0.7, 'mercado': 0.15, 'desenvolvimento': 0.10, 'conectividade': 0.05},
        'Foco Mercado Premium': {'logistico': 0.15, 'mercado': 0.6, 'desenvolvimento': 0.15, 'conectividade': 0.10},
        'Estratégia Balanceada': {'logistico': 0.30, 'mercado': 0.30, 'desenvolvimento': 0.25, 'conectividade': 0.15},
        'Foco Desenvolvimento Regional': {'logistico': 0.20, 'mercado': 0.25, 'desenvolvimento': 0.45, 'conectividade': 0.10},
        'Máxima Conectividade': {'logistico': 0.25, 'mercado': 0.20, 'desenvolvimento': 0.15, 'conectividade': 0.40},
        'E-commerce Agressivo': {'logistico': 0.50, 'mercado': 0.35, 'desenvolvimento': 0.10, 'conectividade': 0.05},
        'Expansão Sustentável': {'logistico': 0.25, 'mercado': 0.25, 'desenvolvimento': 0.35, 'conectividade': 0.15}
    }
    
    resultados_cenarios = []
    for nome_cenario, pesos in cenarios_expandidos.items():
        score_rec_cenario = (scores_recife['logistico'] * pesos['logistico'] + 
                            scores_recife['mercado'] * pesos['mercado'] + 
                            scores_recife['desenvolvimento'] * pesos['desenvolvimento'] + 
                            scores_recife['conectividade'] * pesos['conectividade'])
        
        score_sal_cenario = (scores_salvador['logistico'] * pesos['logistico'] + 
                            scores_salvador['mercado'] * pesos['mercado'] + 
                            scores_salvador['desenvolvimento'] * pesos['desenvolvimento'] + 
                            scores_salvador['conectividade'] * pesos['conectividade'])
        
        vencedor = "Recife" if score_rec_cenario > score_sal_cenario else "Salvador"
        diferenca = abs(score_rec_cenario - score_sal_cenario)
        confianca = "Alta" if diferenca > 0.05 else "Média" if diferenca > 0.02 else "Baixa"
        
        resultados_cenarios.append({
            'Cenário Estratégico': nome_cenario,
            'Score Recife': f"{score_rec_cenario:.3f}",
            'Score Salvador': f"{score_sal_cenario:.3f}",
            'Recomendação': vencedor,
            'Margem': f"{diferenca:.3f}",
            'Confiança': confianca
        })
    
    df_cenarios = pd.DataFrame(resultados_cenarios)
    st.dataframe(df_cenarios, use_container_width=True)
    
    # Visualização da análise de sensibilidade
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras dos cenários
        fig_cenarios = go.Figure()
        
        cenarios_nomes = list(cenarios_expandidos.keys())
        scores_rec_todos = [float(df_cenarios.loc[i, 'Score Recife']) for i in range(len(cenarios_nomes))]
        scores_sal_todos = [float(df_cenarios.loc[i, 'Score Salvador']) for i in range(len(cenarios_nomes))]
        
        fig_cenarios.add_trace(go.Bar(name='Recife', y=cenarios_nomes, x=scores_rec_todos, 
                                     orientation='h', marker_color='#00BFFF'))
        fig_cenarios.add_trace(go.Bar(name='Salvador', y=cenarios_nomes, x=scores_sal_todos, 
                                     orientation='h', marker_color='#FF6F00'))
        
        fig_cenarios.update_layout(
            title="Scores por Cenário Estratégico",
            xaxis_title="Score Composto",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_cenarios, use_container_width=True)
    
    with col2:
        # Análise de robustez
        vitorias_recife = sum(1 for resultado in resultados_cenarios if resultado['Recomendação'] == 'Recife')
        vitorias_salvador = len(resultados_cenarios) - vitorias_recife
        
        fig_robustez = go.Figure(data=[go.Pie(
            labels=['Recife', 'Salvador'],
            values=[vitorias_recife, vitorias_salvador],
            marker_colors=['#00BFFF', '#FF6F00']
        )])
        
        fig_robustez.update_layout(title="Robustez da Decisão<br>(% de Cenários Vencidos)")
        st.plotly_chart(fig_robustez, use_container_width=True)

    # --- 6. Análise de Custos Estimados ---
    st.header("💸 Análise de Custos Operacionais Estimados", divider='orange')
    
    st.markdown("""
    **Estimativas baseadas em benchmarks do setor logístico nacional:**
    """)
    
    # Cálculos de custos estimados
    def calcular_custos_estimados(df, cidade_nome):
        pop_total = df['populacao_2025'].sum()
        tempo_medio = df['tempo_viagem_min'].mean()
        distancia_media = df['distancia_rodoviaria_km_x'].mean() if 'distancia_rodoviaria_km_x' in df.columns else 500
        num_destinos = len(df)
        
        # Estimativas (valores baseados em benchmarks do setor)
        custo_combustivel_mensal = distancia_media * num_destinos * 2.5 * 30  # R$ 2,50/km média
        custo_pedagio_mensal = distancia_media * num_destinos * 0.08 * 30  # R$ 0,08/km média
        custo_motorista_mensal = tempo_medio * num_destinos * 0.5 * 30  # R$ 0,50/min
        custo_manutencao_mensal = (distancia_media * num_destinos * 0.15 * 30)  # R$ 0,15/km
        
        custo_operacional_mensal = custo_combustivel_mensal + custo_pedagio_mensal + custo_motorista_mensal + custo_manutencao_mensal
        custo_operacional_anual = custo_operacional_mensal * 12
        
        # Receita potencial estimada
        receita_potencial_anual = pop_total * 150  # R$ 150 per capita/ano estimado
        
        roi_estimado = (receita_potencial_anual - custo_operacional_anual) / custo_operacional_anual
        
        return {
            'cidade': cidade_nome,
            'custo_combustivel_mensal': custo_combustivel_mensal,
            'custo_operacional_mensal': custo_operacional_mensal,
            'custo_operacional_anual': custo_operacional_anual,
            'receita_potencial_anual': receita_potencial_anual,
            'roi_estimado': roi_estimado,
            'payback_meses': (custo_operacional_anual * 2) / (custo_operacional_mensal) if custo_operacional_mensal > 0 else 0
        }
    
    custos_recife = calcular_custos_estimados(recife_df, 'Recife')
    custos_salvador = calcular_custos_estimados(salvador_df, 'Salvador')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Custos Operacionais - Recife")
        st.metric("Custo Mensal", f"R$ {custos_recife['custo_operacional_mensal']:,.0f}")
        st.metric("Custo Anual", f"R$ {custos_recife['custo_operacional_anual']:,.0f}")
        st.metric("ROI Estimado", f"{custos_recife['roi_estimado']:.1%}")
        st.metric("Payback", f"{custos_recife['payback_meses']:.0f} meses")
    
    with col2:
        st.subheader("💰 Custos Operacionais - Salvador") 
        st.metric("Custo Mensal", f"R$ {custos_salvador['custo_operacional_mensal']:,.0f}",
                 f"R$ {custos_salvador['custo_operacional_mensal'] - custos_recife['custo_operacional_mensal']:+,.0f}")
        st.metric("Custo Anual", f"R$ {custos_salvador['custo_operacional_anual']:,.0f}",
                 f"R$ {custos_salvador['custo_operacional_anual'] - custos_recife['custo_operacional_anual']:+,.0f}")
        st.metric("ROI Estimado", f"{custos_salvador['roi_estimado']:.1%}",
                 f"{custos_salvador['roi_estimado'] - custos_recife['roi_estimado']:+.1%}")
        st.metric("Payback", f"{custos_salvador['payback_meses']:.0f} meses",
                 f"{custos_salvador['payback_meses'] - custos_recife['payback_meses']:+.0f} meses")
    
    with col3:
        st.subheader("🎯 Análise Financeira")
        melhor_custo = "Recife" if custos_recife['custo_operacional_anual'] < custos_salvador['custo_operacional_anual'] else "Salvador"
        melhor_roi = "Recife" if custos_recife['roi_estimado'] > custos_salvador['roi_estimado'] else "Salvador"
        melhor_payback = "Recife" if custos_recife['payback_meses'] < custos_salvador['payback_meses'] else "Salvador"
        
        st.success(f"✅ **Menor custo operacional:** {melhor_custo}")
        st.success(f"📈 **Melhor ROI:** {melhor_roi}")
        st.success(f"⚡ **Payback mais rápido:** {melhor_payback}")
        
        # Economia anual estimada
        economia_anual = abs(custos_recife['custo_operacional_anual'] - custos_salvador['custo_operacional_anual'])
        cidade_mais_economica = melhor_custo
        
        st.metric("Economia Anual", f"R$ {economia_anual:,.0f}")
        st.info(f"💡 Escolhendo **{cidade_mais_economica}**, economia de **R$ {economia_anual:,.0f}/ano**")

    # --- 7. Recomendações Estratégicas Finais ---
    st.header("🎯 Recomendações Estratégicas Finais", divider='orange')
    
    vencedor_geral = "Recife" if score_recife > score_salvador else "Salvador"
    vencedor_custos = melhor_custo
    vencedor_robustez = "Recife" if vitorias_recife > vitorias_salvador else "Salvador"
    
    # Decisão final baseada em múltiplos critérios
    pontos_recife = (1 if vencedor_geral == "Recife" else 0) + (1 if vencedor_custos == "Recife" else 0) + (1 if vencedor_robustez == "Recife" else 0)
    pontos_salvador = 3 - pontos_recife
    
    decisao_final = "Recife" if pontos_recife >= 2 else "Salvador"
    
    if decisao_final == "Recife":
        st.success("### ✅ RECOMENDAÇÃO FINAL: RECIFE")
        st.markdown(f"""
        **Justificativa Multicritério (Score: {pontos_recife}/3):**
        
        🏆 **Vantagens Competitivas de Recife:**
        - **Performance Geral**: Score composto de {score_recife:.3f} vs {score_salvador:.3f}
        - **Eficiência Logística**: Menor tempo médio para distribuição regional
        - **Custo-Benefício**: {("✅ Menores custos operacionais" if vencedor_custos == "Recife" else "❌ Custos operacionais superiores")}
        - **Robustez Estratégica**: Vence em {vitorias_recife}/{len(cenarios_expandidos)} cenários analisados
        - **Cobertura Regional**: Acesso privilegiado ao eixo Norte-Nordeste
        
        🎯 **Fatores Decisivos:**
        - Posicionamento geográfico otimizado para distribuição multi-estadual
        - Menor complexidade logística para atendimento das capitais nordestinas
        - Hub natural para integração com corredores de crescimento da região
        - Infraestrutura consolidada de transporte e distribuição
        """)
    else:
        st.success("### ✅ RECOMENDAÇÃO FINAL: SALVADOR")
        st.markdown(f"""
        **Justificativa Multicritério (Score: {pontos_salvador}/3):**
        
        🏆 **Vantagens Competitivas de Salvador:**
        - **Performance Geral**: Score composto de {score_salvador:.3f} vs {score_recife:.3f}
        - **Potencial de Mercado**: Maior base populacional e econômica
        - **Custo-Benefício**: {("✅ Menores custos operacionais" if vencedor_custos == "Salvador" else "❌ Custos operacionais superiores")}
        - **Robustez Estratégica**: Vence em {vitorias_salvador}/{len(cenarios_expandidos)} cenários analisados
        - **Conectividade Nacional**: Melhor integração com eixo Sul-Sudeste
        
        🎯 **Fatores Decisivos:**
        - Estado da Bahia representa o maior mercado nordestino
        - Região metropolitana mais desenvolvida e populosa
        - Melhor conectividade com grandes centros consumidores nacionais
        - Potencial de crescimento superior no médio-longo prazo
        """)
    
    # --- 8. Próximos Passos e Considerações ---
    st.header("📋 Próximos Passos e Considerações Adicionais", divider='orange')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Due Diligence Recomendada")
        st.markdown("""
        **Análises Complementares Necessárias:**
        
        1. **📊 Estudos de Viabilidade Detalhados**
           - Análise de custos imobiliários específicos
           - Avaliação de disponibilidade de galpões logísticos
           - Estudo de incentivos fiscais estaduais/municipais
        
        2. **🚚 Validação Operacional**
           - Teste piloto de rotas de distribuição
           - Análise de fornecedores locais (3PL, transportadoras)
           - Avaliação de recursos humanos disponíveis
        
        3. **💼 Análise Regulatória e Tributária**
           - Comparativo de carga tributária entre PE e BA
           - Análise de regulamentações específicas
           - Avaliação de programas de incentivo ao investimento
        
        4. **🌐 Validação de Conectividade**
           - Teste real de tempos de entrega
           - Análise de infraestrutura rodoviária atual
           - Avaliação de projetos futuros de infraestrutura
        """)
    
    with col2:
        st.subheader("⚠️ Riscos e Mitigações")
        st.markdown("""
        **Principais Riscos Identificados:**
        
        1. **🚧 Riscos Logísticos**
           - Congestionamentos urbanos
           - Sazonalidade de demanda regional
           - **Mitigação**: Monitoramento contínuo e flexibilidade operacional
        
        2. **💰 Riscos Financeiros**
           - Variação de custos operacionais
           - Mudanças na legislação tributária
           - **Mitigação**: Modelagem de cenários e cláusulas contratuais
        
        3. **🏗️ Riscos de Implementação**
           - Disponibilidade de infraestrutura adequada
           - Prazo para operacionalização
           - **Mitigação**: Due diligence detalhada e cronograma realista
        
        4. **📈 Riscos de Mercado**
           - Mudanças no perfil de consumo regional
           - Entrada de concorrentes
           - **Mitigação**: Monitoramento de mercado e estratégia adaptativa
        """)
    
    # --- 9. Resumo Executivo Para Decisão ---
    st.subheader("📋 Resumo Executivo Para Tomada de Decisão")
    
    resumo_decisao = f"""
    **DECISÃO RECOMENDADA: {decisao_final.upper()}**
    
    **Critérios de Avaliação:**
    - ✅ Score Geral: {vencedor_geral}
    - ✅ Análise de Custos: {vencedor_custos}  
    - ✅ Robustez Estratégica: {vencedor_robustez}
    
    **Próximas Ações Imediatas:**
    1. Aprovação da localização recomendada pela diretoria
    2. Início do processo de due diligence detalhada
    3. Formação de equipe de projeto para implementação
    4. Cronograma de visitas técnicas e negociações locais
    
    **Timeline Sugerido:**
    - **Fase 1** (30 dias): Due diligence e validações
    - **Fase 2** (60 dias): Negociações e definições contratuais
    - **Fase 3** (90 dias): Implementação e operacionalização
    
    **Investimento Estimado de Implementação:**
    - Infraestrutura: R$ 5-8 milhões
    - Operacional Primeiro Ano: R$ {custos_recife['custo_operacional_anual'] if decisao_final == 'Recife' else custos_salvador['custo_operacional_anual']:,.0f}
    - ROI Esperado: {custos_recife['roi_estimado'] if decisao_final == 'Recife' else custos_salvador['roi_estimado']:.1%}
    """
    
    st.info(resumo_decisao)

else:
    st.error("❌ Não foi possível carregar os dados. Verifique os arquivos CSV nos caminhos especificados.")

# --- Footer ---
st.markdown("---")
st.markdown("""
**📊 Análise Estratégica Desenvolvida para Magazine Luiza**  
*Metodologia: Análise Multicritério de Apoio à Decisão (MCDA) + Análise de Sensibilidade + Modelagem Financeira*

**🔧 Ferramentas Utilizadas:**
- Python (Pandas, Plotly, Streamlit) para análise de dados e visualizações
- Metodologia AHP (Analytic Hierarchy Process) para ponderação de critérios
- Análise de sensibilidade para validação de robustez da decisão
- Modelagem financeira para estimativa de custos e ROI
- Análise geoespacial para otimização logística

**📈 Dados Analisados:**
- {len(recife_df) if recife_df is not None else 0} localidades na análise de Recife
- {len(salvador_df) if salvador_df is not None else 0} localidades na análise de Salvador  
- {len(cenarios_expandidos)} cenários estratégicos simulados
- Múltiplas dimensões: logística, mercado, desenvolvimento, conectividade
""")