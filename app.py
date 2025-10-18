import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import griddata
import plotly.graph_objs as go
import numpy as np
import nltk
import noise

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

    # 6️⃣ Terrains semantici organici
    st.subheader("🏞️ Paesaggi Semantici")
    st.caption("Ogni paesaggio è generato dalla 'geografia' semantica delle frasi di un testo.")

    # Palette di colori naturali per i paesaggi
    landscape_palettes = [
        # Foresta & Montagna
        [[0, '#2d5a2d'], [0.2, '#5a8b5a'], [0.4, '#8fa88f'], [0.6, '#a9a9a9'], [0.8, '#d3d3d3'], [1, '#ffffff']],
        # Deserto & Canyon
        [[0, '#8b4513'], [0.2, '#cd853f'], [0.5, '#f4a460'], [0.7, '#deb887'], [1, '#fffaf0']],
        # Mare & Scogliere
        [[0, '#000080'], [0.15, '#4682b4'], [0.4, '#add8e6'], [0.6, '#f0e68c'], [0.8, '#d2b48c'], [1, '#a0522d']],
        # Ghiacciaio & Tundra
        [[0, '#f5f5f5'], [0.2, '#dcdcdc'], [0.5, '#b0c4de'], [0.7, '#6a5acd'], [1, '#483d8b']],
        # Colline & Campi
        [[0, '#556b2f'], [0.25, '#6b8e23'], [0.5, '#9acd32'], [0.75, '#eee8aa'], [1, '#fafad2']],
        # Vulcano & Cenere
        [[0, '#1c1c1c'], [0.2, '#696969'], [0.4, '#a9a9a9'], [0.6, '#ff4500'], [0.8, '#ff8c00'], [1, '#ffd700']]
    ]
    
    # Prepara tutte le superfici nella stessa figura
    all_traces = []
    valid_texts = []
    
    for i, text in enumerate(texts):
        mask = np.array(text_ids) == i
        emb3d = embeddings_3d[mask]

        if len(emb3d) < 4:
            st.warning(f"Il testo {i+1} ha troppe poche frasi per creare un terrain.")
            continue
        
        valid_texts.append(i + 1)
        
        # Calcola il range e crea una base quadrata
        x_range = np.max(emb3d[:, 0]) - np.min(emb3d[:, 0])
        y_range = np.max(emb3d[:, 1]) - np.min(emb3d[:, 1])
        
        # Usa il range maggiore per entrambe le dimensioni (base quadrata)
        max_range = max(x_range, y_range) if max(x_range, y_range) > 0 else 1
        margin = max_range * 0.3 # Aumenta il margine per una griglia più ampia
        
        # Centro dei dati
        x_center = (np.max(emb3d[:, 0]) + np.min(emb3d[:, 0])) / 2
        y_center = (np.max(emb3d[:, 1]) + np.min(emb3d[:, 1])) / 2
        
        # Crea griglia quadrata centrata sulle coordinate reali
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
        
        # Usa la media dei valori Z esistenti per riempire i NaN
        z_mean_interp = np.nanmean(grid_z)
        grid_z = np.nan_to_num(grid_z, nan=z_mean_interp)
        
        # --- Effetti artistici basati su valori non normalizzati ---
        z_mean = np.mean(grid_z) # Calcola la media Z come riferimento
        
        # Crea montagne e laghi basandosi sulla deviazione dalla media
        np.random.seed(42 + i)
        
        # Effetto montagna per valori sopra la media
        mountain_mask = grid_z > z_mean
        if np.any(mountain_mask):
            # Rende i picchi più alti e ripidi in modo non lineare
            grid_z[mountain_mask] = z_mean + np.power(grid_z[mountain_mask] - z_mean, 1.1)
        
        # Effetto lago per valori sotto la media
        lake_mask = grid_z < z_mean
        if np.any(lake_mask):
            # Rende le depressioni più dolci
            grid_z[lake_mask] = z_mean - np.power(z_mean - grid_z[lake_mask], 0.9)

        # Aggiungi rumore Perlin per un aspetto più naturale
        scale = 10.0
        octaves = 4
        persistence = 0.6
        lacunarity = 2.0
        
        perlin_noise = np.zeros(grid_x.shape)
        for r in range(grid_x.shape[0]):
            for c in range(grid_x.shape[1]):
                perlin_noise[r][c] = noise.pnoise2(grid_x[r][c] / scale,
                                                   grid_y[r][c] / scale,
                                                   octaves=octaves,
                                                   persistence=persistence,
                                                   lacunarity=lacunarity,
                                                   repeatx=1024,
                                                   repeaty=1024,
                                                   base=42 + i)
        
        # Modula l'intensità del rumore in base all'altezza
        z_range_local = np.max(grid_z) - np.min(grid_z) if np.max(grid_z) > np.min(grid_z) else 1
        noise_intensity = perlin_noise * z_range_local * 0.15 # Rumore proporzionale all'altezza locale
        grid_z += noise_intensity

        # Superficie del paesaggio con coordinate reali
        trace_surface = go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=landscape_palettes[i % len(landscape_palettes)],
            lighting=dict(
                ambient=0.6,
                diffuse=1.0,
                fresnel=0.2,
                specular=0.4,
                roughness=0.8
            ),
            opacity=1.0,
            showscale=False,
            hoverinfo='skip',
            name=f"Paesaggio {i+1}",
            showlegend=True
        )
        all_traces.append(trace_surface)
        
        # Etichetta di testo 3D per identificare il paesaggio
        label_x = x_center
        label_y = y_center + max_range/2 + margin * 0.5 # Posiziona sopra il terrain
        label_z = np.max(grid_z) + z_range_local * 0.1 # Poco sopra il picco massimo

        trace_text = go.Scatter3d(
            x=[label_x],
            y=[label_y],
            z=[label_z],
            mode='text',
            text=[f'Paesaggio {i+1}'],
            textfont=dict(size=12, color='#2F4F4F', family='Georgia'),
            showlegend=False,
            hoverinfo='skip'
        )
        all_traces.append(trace_text)

    # Aggiungi i punti originali delle frasi alla visualizzazione dei paesaggi
    scatter_points = []
    for i, text in enumerate(texts):
        mask = np.array(text_ids) == i
        scatter_points.append(go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            marker=dict(
                size=3.5,
                color=colors[i % len(colors)],
                opacity=0.8,
                symbol='circle'
            ),
            text=[f"Testo {i+1}: {s}" for s in np.array(sentences)[mask]],
            hoverinfo='text',
            name=f'Frasi Testo {i+1}',
            showlegend=False
        ))
    
    if len(valid_texts) > 0:
        # Crea figura unica con tutti i paesaggi e i punti
        fig = go.Figure(data=all_traces + scatter_points)
        
        # Layout della scena 3D
        fig.update_layout(
            title=dict(
                text="🌐 Confronto tra Paesaggi Semantici",
                font=dict(size=26, family='Georgia', color='#2F4F4F'),
                x=0.5
            ),
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                xaxis=dict(showbackground=False, zeroline=False),
                yaxis=dict(showbackground=False, zeroline=False),
                zaxis=dict(showbackground=False, zeroline=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectratio=dict(x=1, y=1, z=0.5), # Aspetto più bilanciato
                bgcolor='#e9f5f8'
            ),
            paper_bgcolor='#ffffff',
            height=800, # Aumenta altezza per migliore visuale
            margin=dict(l=20, r=20, t=100, b=20),
            legend=dict(
                title="Legenda",
                x=0.05,
                y=0.95,
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='#a9a9a9',
                borderwidth=1,
                font=dict(family="Georgia", size=12, color="#2F4F4F")
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Inserisci almeno due testi e premi 'Analizza testi' per iniziare.")
