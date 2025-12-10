from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "features" / "train_features.parquet"
EMBEDDINGS_PATH = BASE_DIR / "features" / "embeddings" / "train_umap_2d.parquet"
MODEL_PATH = BASE_DIR / "models" / "best_cluster_model_UMAP_kmeans.pkl"
MOVIES_PATH = BASE_DIR / "movies_train.csv"
MOVIES_PATH_SEARCH = BASE_DIR / "test_posters_df.csv"

COLOR_PREFIX = "color_hist_"
TOP_K = 10

st.set_page_config(
    page_title="Movie Visual Explorer",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_resources() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray, dict]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("No se encontró features/train_features.parquet. Genera las características antes de ejecutar la app.")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("No se encontró features/embeddings/train_umap_2d.parquet. Ejecuta la reducción antes de la app.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No se encontró models/best_cluster_model_UMAP_kmeans.pkl. Entrena y guarda el mejor modelo previamente.")
    if not MOVIES_PATH.exists():
        raise FileNotFoundError("No se encontró movies_train.csv. Asegúrate de tener el dataset base disponible.")
    if not MOVIES_PATH_SEARCH.exists():
        raise FileNotFoundError("No se encontró test_poster_df.csv. Asegúrate de tener el CSV de películas de test con posters.")

    features_df = pd.read_parquet(FEATURES_PATH)
    embeddings_df = pd.read_parquet(EMBEDDINGS_PATH)
    movies_meta = pd.read_csv(MOVIES_PATH, usecols=["movieId", "title", "genres"])
    
    # Cargar películas de test para búsqueda
    test_movies_df = pd.read_csv(MOVIES_PATH_SEARCH)
    test_movies_df["poster_abspath_search"] = test_movies_df["poster_path"].apply(
        lambda rel: str((BASE_DIR / rel).resolve()) if isinstance(rel, str) and rel else ""
    )

    with MODEL_PATH.open("rb") as fh:
        model_payload = pickle.load(fh)

    assignments_df = model_payload.get("assignments")
    if assignments_df is None:
        raise KeyError("El modelo guardado no contiene la tabla de asignaciones de clusters.")

    merged = (
        features_df.merge(assignments_df, on="movieId", how="inner")
        .merge(movies_meta, on="movieId", how="left")
        .merge(embeddings_df, on="movieId", how="left")
        .copy()
    )

    title_col = next((col for col in ("title", "title_x", "title_y") if col in merged.columns), None)
    if title_col is None:
        merged["title"] = "Título desconocido"
    else:
        merged["title"] = merged[title_col].fillna("Título desconocido")
    merged = merged.drop(columns=[col for col in ("title_x", "title_y") if col in merged.columns], errors="ignore")

    genres_col = next((col for col in ("genres", "genres_x", "genres_y") if col in merged.columns), None)
    if genres_col is None:
        merged["genres"] = "(no genres listed)"
    else:
        merged["genres"] = merged[genres_col].fillna("(no genres listed)")
    merged = merged.drop(columns=[col for col in ("genres_x", "genres_y") if col in merged.columns], errors="ignore")

    if "poster_path" not in merged.columns:
        merged["poster_path"] = ""

    color_columns = sorted(
        [col for col in merged.columns if col.startswith(COLOR_PREFIX)],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not color_columns:
        raise KeyError("No se encontraron columnas color_hist_* en el dataframe de características.")

    color_matrix = merged[color_columns].fillna(0.0).to_numpy(dtype=np.float32)
    norms = np.linalg.norm(color_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_colors = color_matrix / norms

    merged["year"] = merged["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    merged["poster_abspath"] = merged["poster_path"].apply(
        lambda rel: str((BASE_DIR / rel).resolve()) if isinstance(rel, str) and rel else ""
    )
    merged = merged.reset_index(drop=True)
    merged["row_index"] = merged.index

    return merged, test_movies_df, color_columns, normalized_colors, model_payload


def compute_color_histogram(image: Image.Image, bins_per_channel: int) -> np.ndarray:
    image = image.convert("RGB")
    array = np.array(image)
    parts: List[np.ndarray] = []
    for channel in range(3):
        channel_data = array[..., channel]
        hist, _ = np.histogram(channel_data, bins=bins_per_channel, range=(0, 256), density=False)
        parts.append(hist.astype(np.float32))
    vector = np.concatenate(parts)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(normalized_matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    vector = query_vector.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros(normalized_matrix.shape[0], dtype=np.float32)
    if not np.isclose(norm, 1.0):
        vector = vector / norm
    return normalized_matrix @ vector


def predict_cluster_for_query(
    model_payload: dict,
    query_vector: np.ndarray,
    dataset: pd.DataFrame,
    normalized_matrix: np.ndarray
) -> int:
    """
    Predice el cluster para un vector de consulta.
    Si el modelo tiene un reductor guardado, lo usa; si no, usa el vecino más cercano.
    """
    clustering_model = model_payload.get('model')
    reducer = model_payload.get('reducer')
    
    # Si tenemos el reductor y el modelo de clustering
    if reducer is not None and clustering_model is not None:
        try:
            # Transformar el query_vector al espacio de embeddings
            query_embedding = reducer.transform(query_vector.reshape(1, -1))
            predicted_cluster = clustering_model.predict(query_embedding)[0]
            return int(predicted_cluster)
        except Exception as e:
            st.warning(f"No se pudo usar el modelo directamente: {e}. Usando vecino más cercano.")
    
    # Fallback: usar el cluster del vecino más cercano
    sims = cosine_similarity(normalized_matrix, query_vector)
    nearest_idx = np.argmax(sims)
    return int(dataset.iloc[nearest_idx]['cluster'])


def find_similar_movies_with_cluster(
    dataset: pd.DataFrame,
    normalized_matrix: np.ndarray,
    query_vector: np.ndarray,
    model_payload: dict,
    top_k: int = TOP_K,
    exclude_movie_id: Optional[int] = None,
    cluster_weight: float = 0.7,
    genre_filter: Optional[str] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Encuentra películas similares usando clustering + similitud coseno.
    
    Args:
        dataset: DataFrame con todas las películas
        normalized_matrix: Matriz de vectores de color normalizados
        query_vector: Vector de color de la consulta
        model_payload: Diccionario con el modelo y metadata
        top_k: Número total de recomendaciones
        exclude_movie_id: ID de película a excluir (si es la misma consulta)
        cluster_weight: Proporción de resultados del mismo cluster (0.0 a 1.0)
        genre_filter: Género para filtrar (None = sin filtro)
        year_range: Tupla (año_min, año_max) para filtrar (None = sin filtro)
    
    Returns:
        DataFrame con las películas más similares
    """
    
    # Aplicar filtros de metadata
    filtered_dataset = dataset.copy()
    filtered_indices = dataset.index.to_numpy()
    
    if genre_filter and genre_filter != "Todos":
        genre_mask = filtered_dataset['genres'].str.contains(genre_filter, case=False, na=False)
        filtered_dataset = filtered_dataset[genre_mask]
        filtered_indices = filtered_dataset.index.to_numpy()
    
    if year_range:
        year_min, year_max = year_range
        year_mask = (filtered_dataset['year'] >= year_min) & (filtered_dataset['year'] <= year_max)
        filtered_dataset = filtered_dataset[year_mask]
        filtered_indices = filtered_dataset.index.to_numpy()
    
    if len(filtered_dataset) == 0:
        return pd.DataFrame()
    
    # Obtener matriz normalizada filtrada
    filtered_normalized = normalized_matrix[filtered_indices]
    
    # 1. Predecir cluster de la consulta
    predicted_cluster = predict_cluster_for_query(
        model_payload, query_vector, dataset, normalized_matrix
    )
    
    # 2. Calcular similitudes para las películas filtradas
    sims = cosine_similarity(filtered_normalized, query_vector)
    
    # 3. Separar por cluster
    same_cluster_mask = filtered_dataset['cluster'] == predicted_cluster
    
    # 4. Calcular cuántos tomar de cada grupo
    k_same_cluster = int(top_k * cluster_weight)
    k_other_clusters = top_k - k_same_cluster
    
    # 5. Ordenar dentro del mismo cluster
    same_cluster_local_indices = np.where(same_cluster_mask.to_numpy())[0]
    if len(same_cluster_local_indices) > 0:
        same_cluster_sims = sims[same_cluster_local_indices]
        same_cluster_order = np.argsort(same_cluster_sims)[::-1]
        
        if exclude_movie_id is not None:
            mask = filtered_dataset.iloc[same_cluster_local_indices[same_cluster_order]]["movieId"].to_numpy() != exclude_movie_id
            same_cluster_order = same_cluster_order[mask]
        
        top_same_local = same_cluster_local_indices[same_cluster_order[:k_same_cluster]]
    else:
        top_same_local = np.array([], dtype=int)
    
    # 6. Ordenar de otros clusters
    other_cluster_local_indices = np.where(~same_cluster_mask.to_numpy())[0]
    if len(other_cluster_local_indices) > 0:
        other_cluster_sims = sims[other_cluster_local_indices]
        other_cluster_order = np.argsort(other_cluster_sims)[::-1]
        
        if exclude_movie_id is not None:
            mask = filtered_dataset.iloc[other_cluster_local_indices[other_cluster_order]]["movieId"].to_numpy() != exclude_movie_id
            other_cluster_order = other_cluster_order[mask]
        
        top_other_local = other_cluster_local_indices[other_cluster_order[:k_other_clusters]]
    else:
        top_other_local = np.array([], dtype=int)
    
    # 7. Combinar resultados
    final_local_indices = np.concatenate([top_same_local, top_other_local])
    result = filtered_dataset.iloc[final_local_indices].copy()
    result["similarity"] = sims[final_local_indices]
    result["from_same_cluster"] = np.concatenate([
        np.ones(len(top_same_local), dtype=bool),
        np.zeros(len(top_other_local), dtype=bool)
    ])
    result["predicted_cluster"] = predicted_cluster
    
    return result


def load_image_bytes(path_str: str) -> Optional[bytes]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_bytes()


def render_recommendations(similar_df: pd.DataFrame) -> None:
    if similar_df.empty:
        st.info("No se encontraron coincidencias para recomendar.")
        return

    cols = st.columns(5)
    for idx, (_, row) in enumerate(similar_df.iterrows()):
        column = cols[idx % len(cols)]
        with column:
            poster_bytes = load_image_bytes(row.get("poster_abspath", ""))
            if poster_bytes is not None:
                try:
                    column.image(Image.open(io.BytesIO(poster_bytes)), use_container_width=True)
                except UnidentifiedImageError:
                    column.image(np.zeros((270, 180, 3), dtype=np.uint8), use_container_width=True)
            else:
                column.image(np.zeros((270, 180, 3), dtype=np.uint8), use_container_width=True)
            
            # Indicador visual: 🎯 mismo cluster, 🔄 otro cluster
            badge = "🎯" if row.get('from_same_cluster', False) else "🔄"
            column.caption(f"{badge} {row['title']} (ID {row['movieId']})")
            column.markdown(f"Cluster **{row['cluster']}** | Similitud: {row['similarity']:.3f}")


def build_submission_frame(query_id: str, similar_df: pd.DataFrame) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "query_movie_id": [query_id] * len(similar_df),
            "recommended_movie_id": similar_df["movieId"].astype(str).tolist(),
            "position": list(range(1, len(similar_df) + 1)),
        }
    )
    return submission


# Cargar recursos
data, test_movies, color_columns, normalized_colors, best_model = load_resources()
bins_per_channel = len(color_columns) // 3

# Interfaz principal
st.title("🎬 Movie Visual Explorer")
st.markdown("Explora recomendaciones combinando **similitud de color** con **clustering inteligente**.")

# Sidebar - Opciones
st.sidebar.header("⚙️ Opciones")
mode = st.sidebar.radio("Modo de consulta", ["Seleccionar película", "Subir poster"])
top_k = st.sidebar.slider("Número de recomendaciones", min_value=5, max_value=20, value=TOP_K, step=1)
cluster_weight = st.sidebar.slider(
    "% del mismo cluster", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.85,  # Mayor peso al modelo
    step=0.05,
    help="0.0 = solo similitud de color | 1.0 = solo del mismo cluster"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Estrategia actual:**
- {int(cluster_weight * 100)}% del cluster predicho
- {int((1 - cluster_weight) * 100)}% de otros clusters
""")

# Filtros adicionales
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros de Recomendaciones")

# Extraer géneros únicos del dataset de train
all_genres = set()
for genres_str in data['genres'].dropna():
    all_genres.update(genres_str.split('|'))
all_genres = sorted(list(all_genres))

genre_filter = st.sidebar.selectbox(
    "Filtrar por género",
    ["Todos"] + all_genres,
    help="Solo mostrará películas que contengan este género"
)

# Filtro por año
year_min = int(data['year'].min()) if pd.notna(data['year'].min()) else 1900
year_max = int(data['year'].max()) if pd.notna(data['year'].max()) else 2025

enable_year_filter = st.sidebar.checkbox("Filtrar por año", value=False)
year_range = None
if enable_year_filter:
    year_range = st.sidebar.slider(
        "Rango de años",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1
    )

selected_movie_id: Optional[int] = None
query_vector: Optional[np.ndarray] = None
similar_results: Optional[pd.DataFrame] = None

# Modo 1: Seleccionar película existente (de test con posters)
if mode == "Seleccionar película":
    sorted_test_movies = test_movies.sort_values("title")
    labels = sorted_test_movies.apply(
        lambda row: f"{row['title']} (ID {row['movieId']}) - {row['primary_genre']}", 
        axis=1
    ).tolist()
    
    choice = st.selectbox("Elige una película de test (con poster)", labels, key="movie_selector")
    if choice:
        idx = labels.index(choice)
        selected_test_row = sorted_test_movies.iloc[idx]
        selected_movie_id = int(selected_test_row["movieId"])
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("📽️ Referencia")
            ref_bytes = load_image_bytes(selected_test_row.get("poster_abspath_search", ""))
            if ref_bytes is not None:
                try:
                    st.image(Image.open(io.BytesIO(ref_bytes)), width=220)
                except UnidentifiedImageError:
                    st.image(np.zeros((330, 220, 3), dtype=np.uint8), width=220)
            else:
                st.image(np.zeros((330, 220, 3), dtype=np.uint8), width=220)
        
        with col2:
            st.subheader("ℹ️ Información")
            st.markdown(f"**Título:** {selected_test_row['title']}")
            st.markdown(f"**Género Principal:** {selected_test_row['primary_genre']}")
            st.markdown(f"**Géneros:** {selected_test_row['genres']}")
            st.markdown(f"**IMDb ID:** {selected_test_row['imdbId_str']}")
            st.info("🎯 Película de **TEST** - Buscando recomendaciones en el dataset de **TRAIN**")
        
        # Calcular histograma del poster de test
        try:
            test_image = Image.open(io.BytesIO(ref_bytes))
            query_vector = compute_color_histogram(test_image, bins_per_channel)
        except Exception as e:
            st.error(f"Error al procesar el poster: {e}")
            query_vector = None

# Modo 2: Subir poster personalizado
elif mode == "Subir poster":
    uploaded = st.file_uploader("📤 Carga una imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            uploaded_image = Image.open(uploaded)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("🖼️ Tu Poster")
                st.image(uploaded_image, width=220)
            
            with col2:
                st.subheader("🔍 Análisis")
                st.info("El sistema analizará los colores del poster y lo comparará con la base de datos de películas.")
            
            query_vector = compute_color_histogram(uploaded_image, bins_per_channel)
            selected_movie_id = "uploaded"
        except UnidentifiedImageError:
            st.error("❌ No se pudo leer la imagen. Intenta con otro archivo.")

# Generar recomendaciones si hay un query_vector
if query_vector is not None:
    exclude_id = selected_movie_id if isinstance(selected_movie_id, int) else None
    
    # Mostrar filtros activos
    active_filters = []
    if genre_filter != "Todos":
        active_filters.append(f"Género: {genre_filter}")
    if year_range:
        active_filters.append(f"Años: {year_range[0]}-{year_range[1]}")
    
    if active_filters:
        st.info(f"🔍 Filtros activos: {' | '.join(active_filters)}")
    
    with st.spinner("🔮 Generando recomendaciones inteligentes..."):
        similar_results = find_similar_movies_with_cluster(
            data,
            normalized_colors,
            query_vector,
            best_model,
            top_k=top_k,
            exclude_movie_id=exclude_id,
            cluster_weight=cluster_weight,
            genre_filter=genre_filter if genre_filter != "Todos" else None,
            year_range=year_range,
        )
    
    st.markdown("---")
    st.subheader("✨ Recomendaciones basadas en clustering y color")
    
    if similar_results.empty:
        st.warning("⚠️ No se encontraron películas que cumplan con los filtros seleccionados. Intenta ajustar los filtros.")
    elif not similar_results.empty:
        predicted_cluster = int(similar_results.iloc[0]["predicted_cluster"])
        cluster_counts = similar_results["cluster"].value_counts().to_dict()
        same_cluster_count = similar_results["from_same_cluster"].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cluster Predicho", predicted_cluster)
        with col2:
            st.metric("Del mismo cluster", f"{same_cluster_count}/{len(similar_results)}")
        with col3:
            avg_sim = similar_results["similarity"].mean()
            st.metric("Similitud Promedio", f"{avg_sim:.3f}")
        
        st.markdown(
            "**Distribución de clusters:** " + 
            " · ".join(f"C{cluster}: {count}" for cluster, count in sorted(cluster_counts.items()))
        )
    
    render_recommendations(similar_results)

    # Tabla de submission
    if not similar_results.empty:
        st.markdown("---")
        submission_df = build_submission_frame(str(selected_movie_id), similar_results)
        st.subheader("📊 Formato de Submisión")
        st.dataframe(submission_df, use_container_width=True)
        st.download_button(
            label="⬇️ Descargar CSV",
            data=submission_df.to_csv(index=False).encode("utf-8"),
            file_name="recomendaciones.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Selecciona una película o sube un poster para obtener recomendaciones.")

# Información del modelo en sidebar
st.sidebar.markdown("---")
st.sidebar.header("🤖 Modelo Activo")
st.sidebar.write(f"**Embedding:** {best_model.get('embedding_method', 'N/D')}")
st.sidebar.write(f"**Algoritmo:** {best_model.get('algorithm', 'N/D')}")
st.sidebar.write(f"**Config:** {best_model.get('config_label', 'N/D')}")

# Mostrar si tiene reductor guardado
has_reducer = best_model.get('reducer') is not None
st.sidebar.write(f"**Reductor guardado:** {'✅ Sí' if has_reducer else '❌ No'}")

metrics = best_model.get("metrics", {})
if metrics:
    st.sidebar.markdown("**Métricas:**")
    for key, value in metrics.items():
        st.sidebar.write(f"• {key}: {value:.3f}")
else:
    st.sidebar.write("Sin métricas registradas")