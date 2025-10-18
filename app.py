import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import griddata
import plotly.graph_objs as go
import numpy as np
import nltk

# Scarica tokenizer per frasi (solo la prima volta)
nltk.download('punkt', quiet=True)

# Carica modello una sola volta
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 🎨 Imposta la pagina
st.set_page_config(page_title="Confronto Semantico 3D", layout="wide")
st.title("🔍 Confronto Semantico 3D tra Testi")
st.write("Incolla due o più testi per visualizzare la similarità semantica tra le frasi e i loro 'terreni' semantici in 3D.")

# 📝 Input dinamico dei testi
num_texts = st.number_input("Quanti testi vuoi confrontare?", min_value=2, max_value=6, value=2)
texts = []

for i in range(num_texts):
    text = st.text_area(f"Testo {i+1}", height=150, placeholder=f"Incolla qui il testo {i+1}...")
    if text.strip():
        texts.append(text)

if len(texts) >= 2 and st.button("🔎 Analizza testi"):
    sentences = []
    text_ids = []
    
    # 1️⃣ Tokenizza in frasi
    for i, text in enumerate(texts):
        sents = nltk.sent_tokenize(text, language='italian')
        sentences.extend(sents)
        text_ids.extend([i] * len(sents))
    
    # 2️⃣ Calcola embeddings
    with st.spinner("Calcolo degli embeddings..."):
        embeddings = model.encode(sentences)
    
    # 3️⃣ Riduzione a 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # 4️⃣ Visualizzazione globale
    colors = ['#ff9aa2', '#ffb7b2', '#ffdac1', '#e2f0cb', '#b5ead7', '#c7ceea']
    traces = []
    
    for i, text in enumerate(texts):
        mask = np.array(text_ids) == i
        trace = go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers+lines+text',
            marker=dict(size=6, color=colors[i % len(colors)], opacity=0.9),
            text=[f"Testo {i+1}: {s}" for s in np.array(sentences)[mask]],
            textposition="top center",
            name=f"Testo {i+1}"
        )
        traces.append(trace)
    
    layout = go.Layout(
        title="Confronto 3D tra frasi dei testi",
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        legend=dict(title="Origine del testo")
    )
    
    fig_global = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig_global, use_container_width=True)

    # 5️⃣ Similarità media tra testi
    st.subheader("📈 Similarità media tra testi")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(
                embeddings[np.array(text_ids) == i].mean(axis=0).reshape(1, -1),
                embeddings[np.array(text_ids) == j].mean(axis=0).reshape(1, -1)
            )[0, 0]
            st.write(f"**Testo {i+1} ↔ Testo {j+1}:** {sim:.3f}")

    # 6️⃣ Terrains semantici in stile Manga - tutti nella stessa scena 3D
    st.subheader("Terrain Semantici")
    #st.caption("Cel-shading con colori pastello • Navigazione sincronizzata tra tutti i terrain")
    
    # Palette manga con cel-shading - adattate per montagne e laghi
    manga_palettes = [
        [[0, '#b0c4de'], [0.3, '#e6d5f5'], [0.5, '#b8e6d5'], [0.75, '#ffd4e5'], [1, '#ffb3c1']],
        [[0, '#87ceeb'], [0.3, '#d5e6f5'], [0.5, '#d5f5e6'], [0.75, '#f5e6d5'], [1, '#f4a460']],
        [[0, '#40e0d0'], [0.3, '#f5d5e6'], [0.5, '#e6f5d5'], [0.75, '#d5e6f5'], [1, '#ff7f7f']],
        [[0, '#6495ed'], [0.3, '#e6e6f5'], [0.5, '#f5e6e6'], [0.75, '#e6f5e6'], [1, '#dda0dd']],
        [[0, '#7fffd4'], [0.3, '#f5e6f5'], [0.5, '#e6f5f5'], [0.75, '#f5f5e6'], [1, '#f0e68c']],
        [[0, '#add8e6'], [0.3, '#ffe6f0'], [0.5, '#e6fff0'], [0.75, '#f0e6ff'], [1, '#ffb6c1']]
    ]
    
    # Prepara tutte le superfici nella stessa figura
    all_traces = []
    valid_texts = []
    spacing = 50  # Spaziatura tra i terrains
    
    for i, text in enumerate(texts):
        mask = np.array(text_ids) == i
        emb3d = embeddings_3d[mask]

        if len(emb3d) < 4:
            st.warning(f"Il testo {i+1} ha troppe poche frasi per creare un terrain.")
            continue
        
        valid_texts.append(i + 1)
        
        # Calcola offset per posizionare i terrains in griglia
        row = i // 2
        col = i % 2
        offset_x = col * spacing
        offset_y = row * spacing

        # Calcola il range e crea una base quadrata
        x_range = np.max(emb3d[:, 0]) - np.min(emb3d[:, 0])
        y_range = np.max(emb3d[:, 1]) - np.min(emb3d[:, 1])
        
        # Usa il range maggiore per entrambe le dimensioni (base quadrata)
        max_range = max(x_range, y_range)
        margin = max_range * 0.25
        
        # Centro dei dati
        x_center = (np.max(emb3d[:, 0]) + np.min(emb3d[:, 0])) / 2
        y_center = (np.max(emb3d[:, 1]) + np.min(emb3d[:, 1])) / 2
        
        # Crea griglia quadrata centrata
        grid_x, grid_y = np.mgrid[
            x_center - max_range/2 - margin : x_center + max_range/2 + margin : 80j,
            y_center - max_range/2 - margin : y_center + max_range/2 + margin : 80j
        ]

        # Interpolazione con picchi definiti
        grid_z = griddata(
            points=emb3d[:, :2],
            values=emb3d[:, 2],
            xi=(grid_x, grid_y),
            method='cubic'
        )
        
        grid_z = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))
        
        # Normalizza per avere range consistente
        z_mean = np.mean(grid_z)
        z_std = np.std(grid_z)
        if z_std > 0:
            grid_z_norm = (grid_z - z_mean) / z_std
        else:
            grid_z_norm = grid_z - z_mean
        
        # Crea montagne (picchi positivi) e laghi (depressioni negative)
        np.random.seed(42 + i)
        
        # Per zone positive: effetto montagna (pendenze più ripide)
        mountain_mask = grid_z_norm > 0
        if np.any(mountain_mask):
            grid_z_norm[mountain_mask] = np.power(grid_z_norm[mountain_mask], 0.7) * 1.2
        
        # Per zone negative: effetto lago (depressioni più dolci e piatte)
        lake_mask = grid_z_norm < -0.3
        if np.any(lake_mask):
            grid_z_norm[lake_mask] = -np.power(np.abs(grid_z_norm[lake_mask]), 1.3) * 0.8
        
        # Rumore diverso per montagne e laghi
        noise_mountain = np.random.normal(0, 0.08, grid_z.shape)
        noise_lake = np.random.normal(0, 0.02, grid_z.shape)
        noise = np.where(mountain_mask, noise_mountain, noise_lake)
        
        # Ondulazioni più pronunciate sulle montagne
        x_norm_wave = (grid_x - np.min(grid_x)) / (np.max(grid_x) - np.min(grid_x) + 1e-8) * 10
        y_norm_wave = (grid_y - np.min(grid_y)) / (np.max(grid_y) - np.min(grid_y) + 1e-8) * 10
        wave = np.sin(x_norm_wave * 0.5) * np.cos(y_norm_wave * 0.5) * 0.15
        wave = np.where(mountain_mask, wave, wave * 0.3)
        
        grid_z = grid_z_norm + noise + wave

        # Normalizza le coordinate per la griglia
        scale = 20  # Scala per ogni terrain
        grid_x_norm = (grid_x - np.min(grid_x)) / (np.max(grid_x) - np.min(grid_x) + 1e-8) * scale
        grid_y_norm = (grid_y - np.min(grid_y)) / (np.max(grid_y) - np.min(grid_y) + 1e-8) * scale
        
        # Applica offset per posizionare in griglia
        grid_x_final = grid_x_norm + offset_x
        grid_y_final = grid_y_norm + offset_y

        # Superficie con effetto cel-shading manga
        trace_surface = go.Surface(
            x=grid_x_final,
            y=grid_y_final,
            z=grid_z,
            colorscale=manga_palettes[i % len(manga_palettes)],
            lighting=dict(
                ambient=0.7,
                diffuse=0.8,
                fresnel=0.1,
                specular=0.2,
                roughness=0.5
            ),
            opacity=0.95,
            showscale=False,
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="#c5b8d4",
                    project=dict(z=True),
                    width=2
                )
            ),
            hoverinfo='skip',
            name=f"Testo {i+1}",
            showlegend=True
        )
        
        all_traces.append(trace_surface)
        
        # Aggiungi etichetta di testo 3D per identificare il terrain
        trace_text = go.Scatter3d(
            x=[offset_x + scale/2],
            y=[offset_y + scale + 3],
            z=[np.max(grid_z) + 0.5],
            mode='text',
            text=[f'🌸 Testo {i+1}'],
            textfont=dict(size=14, color='#8b5a8e', family='Arial'),
            showlegend=False,
            hoverinfo='skip'
        )
        all_traces.append(trace_text)
    
    if len(valid_texts) == 0:
        st.warning("Nessun terrain valido da visualizzare.")
    else:
        # Crea figura unica con tutti i terrains
        fig = go.Figure(data=all_traces)
        
        # Calcola il centro della griglia
        max_row = (len(valid_texts) - 1) // 2
        center_x = spacing / 2
        center_y = (max_row * spacing) / 2
        
        # Layout unico con una sola scena 3D
        fig.update_layout(
            title=dict(
                text="🌸 Confronto Terrain Semantici - Navigazione Sincronizzata",
                font=dict(size=24, family='Arial', color='#8b5a8e')
            ),
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                aspectratio=dict(x=1, y=1, z=0.4),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='#fef5f8',
            height=700,
            margin=dict(l=10, r=10, t=80, b=10),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#c5b8d4',
                borderwidth=2
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Inserisci almeno due testi e premi 'Analizza testi' per iniziare.")
