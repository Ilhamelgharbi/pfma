"""
🏠 SalesHouses - Interface Streamlit
=====================================
Interface moderne pour prédire le prix des appartements au Maroc

Auteur : Assistant IA
Date : Janvier 2026
"""

import streamlit as st
import requests
import os
import json

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SalesHouses",
    page_icon="🏠",
    layout="centered"
)

# Style CSS moderne
st.markdown("""
<style>
    h1 { text-align: center; color: #1f2937; }
    .price-box {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    }
    .price-value { font-size: 3rem; font-weight: 800; }
    .price-per-m2 { font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem; }
    .equipment-tag {
        display: inline-block;
        background: #f3f4f6;
        color: #111827;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.875rem;
        border: 2px solid transparent;
        transition: all 0.2s;
    }
    .equipment-tag.selected {
        background: #dbeafe;
        color: #1e40af;
        border-color: #3b82f6;
        font-weight: 500;
    }
    .equipment-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #111827 !important;
    }
    .metric-card h4 {
        color: #1f2937 !important;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== DONNÉES STATIQUES ==========
# Villes disponibles (basé sur les données d'entraînement)
CITIES = [
    "Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir",
    "Meknès", "Salé", "Mohammedia", "Kénitra", "El Jadida",
    "Temara", "Bouskoura", "Autre"
]

# Équipements disponibles
EQUIPMENT_LIST = [
    "Ascenseur", "Balcon", "Chauffage", "Climatisation",
    "Concierge", "Cuisine Équipée", "Duplex", "Meublé",
    "Parking", "Sécurité", "Terrasse"
]

# ========== FONCTIONS UTILITAIRES ==========
def check_api_health():
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_price(apartment_data):
    """Envoie les données à l'API et retourne la prédiction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=apartment_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Erreur API: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API non disponible. Vérifiez que le backend tourne.")
        return None
    except Exception as e:
        st.error(f"⚠️ Erreur: {e}")
        return None

def format_price(price):
    """Formate le prix en MAD avec séparateurs"""
    return f"{price:,.0f} MAD"

def format_price_per_m2(price_per_m2):
    """Formate le prix au m²"""
    return f"{price_per_m2:,.0f} MAD/m²"

# ========== COMPOSANTS PERSONNALISÉS ==========
def equipment_selector():
    """Sélecteur d'équipements interactif"""
    st.markdown("### 🛠️ Équipements")

    # Session state pour les équipements sélectionnés
    if 'selected_equipment' not in st.session_state:
        st.session_state.selected_equipment = []

    # Utiliser des checkboxes pour éviter les rechargements de page
    st.markdown("**Sélectionnez les équipements disponibles:**")

    # Créer des colonnes pour une meilleure disposition
    cols = st.columns(3)
    equipment_options = {}

    for i, equipment in enumerate(EQUIPMENT_LIST):
        col_idx = i % 3
        with cols[col_idx]:
            equipment_options[equipment] = st.checkbox(
                equipment,
                value=equipment in st.session_state.selected_equipment,
                key=f"equip_{equipment}"
            )

    # Mettre à jour la session state basée sur les checkboxes
    selected_equipment = [equip for equip, selected in equipment_options.items() if selected]
    st.session_state.selected_equipment = selected_equipment

    # Affichage des équipements sélectionnés
    if st.session_state.selected_equipment:
        st.markdown("**Équipements sélectionnés:**")
        selected_text = ", ".join(st.session_state.selected_equipment)
        st.success(f"📋 {selected_text}")
    else:
        st.info("ℹ️ Aucun équipement sélectionné")

    return st.session_state.selected_equipment

# ========== APPLICATION PRINCIPALE ==========
def main():
    # En-tête
    st.markdown("<h1>🏠 SalesHouses</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Prédiction du prix des appartements au Maroc</p>",
                unsafe_allow_html=True)

    # Créer les onglets
    tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Modèle & Visualisations"])

    # ========== ONGLET PRÉDICTION ==========
    with tab1:
        prediction_tab()

    # ========== ONGLET MODÈLE & VISUALISATIONS ==========
    with tab2:
        model_visualizations_tab()

def prediction_tab():
    """Onglet principal de prédiction"""
    # Vérification de l'API
    if not check_api_health():
        st.error("❌ API non disponible")
        st.warning(f"Démarrez le backend sur: {API_URL}")
        st.stop()

    st.success("✅ API connectée")
    st.markdown("---")

    # Formulaire principal (2 colonnes)
    st.markdown("### 📝 Caractéristiques de l'appartement")

    col1, col2 = st.columns(2)

    # Colonne 1 - Localisation
    with col1:
        st.markdown("#### 🏙️ Localisation")
        city = st.selectbox("Ville", CITIES, index=0, key="city_select")
        surface_area = st.number_input(
            "Surface (m²)",
            min_value=20,
            max_value=500,
            value=80,
            step=5,
            key="surface_input"
        )
    # Colonne 2 - Caractéristiques
    with col2:
        st.markdown("#### 📐 Caractéristiques")


        total_rooms = st.number_input(
            "Nombre de pièces",
            min_value=1,
            max_value=15,
            value=3,
            step=1,
            key="rooms_input"
        )

        nb_baths = st.number_input(
            "Nombre de salles de bain",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            key="baths_input"
        )

    # Sélecteur d'équipements
    st.markdown("---")
    selected_equipment = equipment_selector()

    st.markdown("---")

    # Bouton de prédiction
    if st.button("🔮 PRÉDIRE LE PRIX", width='stretch', type="primary"):
        # Validation des données
        if surface_area < 20:
            st.error("⚠️ La surface doit être d'au moins 20 m²")
            return

        if total_rooms < 1:
            st.error("⚠️ L'appartement doit avoir au moins 1 pièce")
            return

        # Préparer les données pour l'API
        apartment_data = {
            "city": city,
            "surface_area": float(surface_area),
            "nb_baths": int(nb_baths),
            "total_rooms": int(total_rooms),
            "equipment_list": selected_equipment
        }

        # Appel API
        with st.spinner("🔄 Analyse en cours..."):
            result = predict_price(apartment_data)

        # Affichage du résultat
        if result:
            predicted_price = result['predicted_price']
            price_per_m2 = result['price_per_m2']
            confidence_interval = result['confidence_interval']

            # Boîte de prix principale
            st.markdown(
                f"""
                <div class="price-box">
                    <div style="font-size: 1.2rem; opacity: 0.9;">💰 Prix estimé</div>
                    <div class="price-value">{predicted_price:,.0f} MAD</div>
                    <div class="price-per-m2">{price_per_m2:,.0f} MAD/m²</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success("✅ Estimation réussie!", icon="✅")

            # Détails supplémentaires
            st.markdown("### 📈 Détails de l'estimation")

            col_detail1, col_detail2 = st.columns(2)

            with col_detail1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>🏠 Prix au mètre carré</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #1d4ed8;">
                            {price_per_m2:,.0f} MAD/m²
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_detail2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>📊 Intervalle de confiance</h4>
                        <p style="font-size: 1.1rem;">
                            {confidence_interval['lower']:,.0f} - {confidence_interval['upper']:,.0f} MAD
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Informations sur la localisation
            st.markdown("### 🗺️ Informations sur la localisation")
            st.info(f"📍 **{city}** - Prix basé sur les données du marché immobilier marocain")

            # Résumé des caractéristiques
            st.markdown("### 📋 Résumé des caractéristiques")
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown(f"""
                - **Surface**: {surface_area} m²
                - **Pièces**: {total_rooms}
                - **Salles de bain**: {nb_baths}
                """)

            with summary_col2:
                equipment_text = ", ".join(selected_equipment) if selected_equipment else "Aucun"
                st.markdown(f"""
                - **Ville**: {city}
                - **Équipements**: {equipment_text}
                """)

def model_visualizations_tab():
    """Onglet d'informations sur le modèle et visualisations"""
    st.markdown("## 🤖 Informations sur le modèle")

    # Métriques du modèle
    st.markdown("### 📊 Performances du modèle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Précision (R²)", "82.7%", "Excellent")

    with col2:
        st.metric("Erreur moyenne (MAE)", "185,807 MAD", "Fiable")

    with col3:
        st.metric("Erreur quadratique (RMSE)", "273,802 MAD", "Bon modèle")

    st.markdown("---")

    # Informations détaillées
    st.markdown("### 🔍 Détails techniques")

    with st.expander("📈 Métriques détaillées"):
        st.markdown("""
        - **Algorithme**: Gradient Boosting Regressor
        - **Précision (R²)**: 0.8269 (82.69%)
        - **Erreur absolue moyenne (MAE)**: 185,807 MAD
        - **Erreur quadratique moyenne (RMSE)**: 273,802 MAD
        - **Erreur relative (MAPE)**: 18.13%
        - **Nombre d'observations**: Base de données immobilière marocaine
        """)

    with st.expander("🏗️ Architecture du modèle"):
        st.markdown("""
        **Features utilisées:**
        - Variables numériques: surface, nombre de pièces, nombre de salles de bain
        - Features engineered: ratio salles de bain/pièces, surface par pièce, score équipements
        - Variables catégorielles: ville (encodage one-hot), équipements (présence/absence)

        **Preprocessing:**
        - Normalisation des features numériques (StandardScaler)
        - Encodage one-hot pour les variables catégorielles
        - Gestion des valeurs manquantes et outliers
        """)

    st.markdown("---")

    # Visualisations
    st.markdown("## 📊 Visualisations")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("### 🏙️ Statistiques par ville")
        try:
            st.image("../visualizations/city_statistics.png", caption="Prix moyens par ville", width='stretch')
        except:
            st.info("📁 Visualisation non disponible")

        st.markdown("### 📈 Distribution des prix")
        try:
            st.image("../visualizations/price_distribution.png", caption="Distribution des prix", width='stretch')
        except:
            st.info("📁 Visualisation non disponible")

    with viz_col2:
        st.markdown("### 🔗 Corrélations")
        try:
            st.image("../visualizations/correlation_matrix.png", caption="Matrice de corrélation", width='stretch')
        except:
            st.info("📁 Visualisation non disponible")

        st.markdown("### 🧹 Gestion des outliers")
        try:
            st.image("../visualizations/outliers_before_after.png", caption="Avant/après traitement des outliers", width='stretch')
        except:
            st.info("📁 Visualisation non disponible")

    # Comparaison des modèles
    st.markdown("### 🏆 Comparaison des modèles")
    try:
        st.image("../visualizations/model_comparison.png", caption="Performance des différents algorithmes", width='stretch')
    except:
        st.info("📁 Visualisation non disponible")

    st.markdown("---")

    # Informations supplémentaires
    st.markdown("## ℹ️ Informations supplémentaires")

    with st.expander("📚 À propos du projet"):
        st.markdown("""
        **SalesHouses** est une application de prédiction des prix immobiliers au Maroc utilisant
        l'intelligence artificielle pour estimer le prix des appartements.

        **Technologies utilisées:**
        - Machine Learning: Scikit-learn, Gradient Boosting
        - Backend: FastAPI (Python)
        - Frontend: Streamlit
        - Données: Base de données immobilière marocaine

        **Auteur:** Assistant IA
        **Date:** Janvier 2026
        """)

    with st.expander("🔧 Comment utiliser l'application"):
        st.markdown("""
        1. **Sélectionnez la ville** où se trouve l'appartement
        2. **Entrez les caractéristiques**: surface, nombre de pièces, salles de bain
        3. **Choisissez les équipements** disponibles (cliquez sur les cases à cocher)
        4. **Cliquez sur "PRÉDIRE LE PRIX"** pour obtenir l'estimation
        5. **Consultez les détails** de l'estimation et l'intervalle de confiance

        L'application utilise un modèle de machine learning entraîné sur des données
        réelles du marché immobilier marocain pour fournir des estimations précises.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #9ca3af;">© 2026 SalesHouses - Prédiction immobilière Maroc</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()