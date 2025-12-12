# application streamlit 
# pour la lancer, lance la commande 
# streamlit run app.py

# app_chanel_streamlit.py

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer

#configuaration generale 

DATA_PATH = "df_chanel_embeddings.pkl"     # Fichier Pickle généré par ton .py
IMAGE_FOLDER = "processed_img"             # Dossier d'images prétraitées (product_code.jpg)

# caché 

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs 1D."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


@st.cache_resource
def load_dataframe():
    """Charge le DataFrame avec embeddings depuis le Pickle."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Le fichier {DATA_PATH} est introuvable. "
            f"Pense à lancer ton script principal pour créer df_chanel_embeddings.pkl."
        )
    df = pd.read_pickle(DATA_PATH)

    # On vérifie la présence des colonnes attendues
    required_cols = ["product_code", "title", "price", "category2_code",
                     "resnet_embedding", "title_embedding_improved"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le DataFrame : {col}")

    # Conversion en matrice NumPy pour les similarités
    img_mat = np.stack(df["resnet_embedding"].values)
    txt_mat = np.stack(df["title_embedding_improved"].values)

    return df, img_mat, txt_mat


@st.cache_resource
def load_models():
    """Charge les modèles ResNet50 (visuel) et SentenceTransformer (texte)."""
    # Modèle visuel : ResNet50 pré-entraîné sur ImageNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet_feature_extractor.eval()

    # Transfos d'entrée cohérentes avec ImageNet
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Modèle textuel : même que dans ton projet (best_model = all-mpnet-base-v2)
    text_model = SentenceTransformer("all-mpnet-base-v2")

    return resnet_feature_extractor, resnet_transform, text_model


def embed_image_uploaded(pil_image: Image.Image,
                         resnet_feature_extractor,
                         resnet_transform) -> np.ndarray:
    """Calcule l'embedding ResNet d'une image uploadée (PIL)."""
    img = pil_image.convert("RGB")
    img_tensor = resnet_transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        emb = resnet_feature_extractor(img_tensor)
        emb = emb.squeeze().numpy()  # shape (2048,)

    return emb


def get_top_k(sim_scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Retourne les indices des k meilleurs scores de similarité (ordre décroissant)."""
    idx_sorted = np.argsort(sim_scores)[::-1]
    return idx_sorted[:k]


def display_results(df, indices, title="Articles recommandés"):
    """Affiche les résultats dans une grille Streamlit."""
    st.subheader(title)

    if len(indices) == 0:
        st.info("Aucun résultat trouvé.")
        return

    n_cols = 2
    for i in range(0, len(indices), n_cols):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            if i + col_idx >= len(indices):
                break
            idx = indices[i + col_idx]
            row = df.iloc[idx]

            with cols[col_idx]:
                # Chargement image locale à partir de product_code
                img_path = os.path.join(IMAGE_FOLDER, f"{row['product_code']}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True)
                else:
                    st.write("(Image non trouvée)")

                st.markdown(f"**{row['title']}**")
                st.write(f"Catégorie : `{row['category2_code']}`")
                st.write(f"Prix : {row['price']} €")
                st.caption(f"Product code : {row['product_code']}")
                st.markdown("---")


# interface streamlit 

def main():
    st.set_page_config(
        page_title="Plateforme de recommandation Chanel",
        page_icon="david",
        layout="wide"
    )

    st.title("Plateforme de recommandation des produits Chanel")
    st.write(
        "Prototype de système de recommandation basé sur les embeddings "
        "visuels (ResNet50) et textuels (all-mpnet-base-v2)."
    )

    # Chargement des données + modèles
    with st.spinner("Chargement des données et des modèles..."):
        df, img_mat, txt_mat = load_dataframe()
        resnet_feature_extractor, resnet_transform, text_model = load_models()

    st.success("Données et modèles chargés")

    mode = st.sidebar.radio(
        "Mode de recherche",
        ["Recherche par image", "Recherche par texte", "Recherche combinée (image + texte)"]
    )

   #imagerie 
    if mode.startswith("..."):
        st.header("Recherche par image")

        uploaded_file = st.file_uploader(
            "Téléverse une image de produit (JPEG/PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Image fournie", use_column_width=False)

            if st.button("Trouver les 10 produits les plus similaires"):
                with st.spinner("Calcul des similarités visuelles..."):
                    query_emb = embed_image_uploaded(
                        pil_img, resnet_feature_extractor, resnet_transform
                    )

                    # Similarités avec tous les embeddings images du dataset
                    sims = np.array([
                        cosine_sim(query_emb, img_emb)
                        for img_emb in img_mat
                    ])
                    top_idx = get_top_k(sims, k=10)

                display_results(df, top_idx, title="Top 10 produits similaires (visuel)")
    #  MODE TEXTE
    elif mode.startswith("..."):
        st.header("Recherche par texte")

        query_text = st.text_area(
            "Décris le produit que tu cherches",
            value="Red matte lipstick with long-lasting finish"
        )

        if st.button("Trouver les 10 produits les plus proches"):
            if query_text.strip() == "":
                st.warning("Merci d'entrer une description.")
            else:
                with st.spinner("Calcul des similarités textuelles..."):
                    query_emb = text_model.encode(query_text)

                    sims = np.array([
                        cosine_sim(query_emb, txt_emb)
                        for txt_emb in txt_mat
                    ])
                    top_idx = get_top_k(sims, k=10)

                display_results(df, top_idx, title="Top 10 produits similaires (texte)")
    #  MODE COMBINÉ
    else:
        st.header("Recherche combinée (image + texte)")

        uploaded_file = st.file_uploader(
            "Téléverse une image de produit (JPEG/PNG)",
            type=["jpg", "jpeg", "png"]
        )
        query_text = st.text_area(
            "Ajoute une description textuelle (optionnel mais recommandé)",
            value="Elegant black handbag for evening events"
        )
        alpha = st.slider(
            "Poids du visuel (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Score total = α * similarité visuelle + (1-α) * similarité textuelle"
        )

        if st.button("Trouver les 10 produits les plus pertinents"):
            if uploaded_file is None:
                st.warning("Merci de téléverser une image pour le mode combiné.")
            elif query_text.strip() == "":
                st.warning("Merci d'entrer une description textuelle pour le mode combiné.")
            else:
                pil_img = Image.open(uploaded_file)
                st.image(pil_img, caption="Image fournie", use_column_width=False)

                with st.spinner("Calcul des similarités combinées..."):
                    # Embedding image
                    img_emb = embed_image_uploaded(
                        pil_img, resnet_feature_extractor, resnet_transform
                    )
                    sims_img = np.array([
                        cosine_sim(img_emb, img_emb_ds)
                        for img_emb_ds in img_mat
                    ])
                    # embeding du texte 
                    txt_emb = text_model.encode(query_text)

                    sims_txt = np.array([
                        cosine_sim(txt_emb, txt_emb_ds)
                        for txt_emb_ds in txt_mat
                    ])
                    # combination de scrore 
                    sims_total = alpha * sims_img + (1.0 - alpha) * sims_txt
                    top_idx = get_top_k(sims_total, k=10)

                display_results(df, top_idx, title="Top 10 produits (score combiné)")

if __name__ == "__main__":
    main()
