import streamlit as st
import pandas as pd
import sys
import os
import time
import plotly.express as px
import plotly.graph_objects as go

# Ajouter le dossier scripts au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from simulate_tournament import load_data, simulate_group_phase, simulate_knockout, predict_match_result, prepare_match_features
    from bracket_viz import display_bracket
except ImportError:
    st.error("Impossible d'importer les scripts de simulation. Vérifiez la structure du projet.")

st.set_page_config(page_title="PredictCAN 2025", page_icon="⚽", layout="wide")

# Liste des pays africains (CAF) pour le filtrage
AFRICAN_TEAMS = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", 
    "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "DR Congo", 
    "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", 
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", 
    "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", 
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", 
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", 
    "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
]

@st.cache_data
def get_elo_history(team_name):
    """Récupère l'historique ELO d'une équipe."""
    try:
        df = pd.read_csv('./data/processed/can_processed_data_with_elo.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Filtrer les matchs de l'équipe
        home_matches = df[df['home_team'] == team_name][['date', 'home_elo']].rename(columns={'home_elo': 'elo'})
        away_matches = df[df['away_team'] == team_name][['date', 'away_elo']].rename(columns={'away_elo': 'elo'})
        
        history = pd.concat([home_matches, away_matches]).sort_values('date')
        return history
    except Exception as e:
        return pd.DataFrame()

def main():
    st.title("⚽ PredictCAN 2025 - Simulation & Prédictions")
    st.markdown("""
    Bienvenue dans l'outil de simulation de la Coupe d'Afrique des Nations 2025.
    Ce projet utilise l'apprentissage automatique (Régression Logistique) et le classement ELO pour prédire les résultats.
    """)

    # Chargement des données
    with st.spinner('Chargement du modèle et des données...'):
        try:
            model, current_elo_ratings, mean_points = load_data()
            st.success("Données chargées avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return

    # Sidebar pour la navigation
    page = st.sidebar.selectbox("Navigation", ["Simulation Complète", "Prédire un Match", "Statistiques & Visualisations"])

    if page == "Simulation Complète":
        st.header("🏆 Simulation du Tournoi")
        
        if st.button("Lancer la Simulation"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 1. Phase de Groupes
            status_text.text("Simulation de la phase de groupes...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            df_standings, qualified_teams_ordered, r16_matchs_structured = simulate_group_phase(model, current_elo_ratings, mean_points)
            
            st.subheader("Phase de Groupes")
            
            # Affichage des groupes en colonnes
            groups = sorted(df_standings['Group'].unique())
            cols = st.columns(3)
            
            for i, group in enumerate(groups):
                with cols[i % 3]:
                    st.markdown(f"### Groupe {group}")
                    group_df = df_standings[df_standings['Group'] == group][['team', 'Pts', 'GD', 'GF', 'GA']]
                    st.dataframe(group_df, hide_index=True)

            # 2. Phase à élimination directe
            status_text.text("Simulation de la phase finale...")
            progress_bar.progress(60)
            time.sleep(0.5)
            
            knockout_results = simulate_knockout(r16_matchs_structured, model, current_elo_ratings, mean_points)
            
            st.subheader("Phase à Élimination Directe")
            
            # Affichage du Bracket
            st.markdown("### Tableau Final")
            display_bracket(knockout_results)
            
            # Affichage des résultats par tour (détails)
            with st.expander("Voir les détails des matchs"):
                rounds = ["Huitièmes de Finale", "Quarts de Finale", "Demi-Finales", "Finale"]
                
                for round_name in rounds:
                    round_matches = [m for m in knockout_results if m['Tour'] == round_name]
                    if round_matches:
                        st.markdown(f"#### {round_name}")
                        for match in round_matches:
                            st.write(f"**{match['Match']}** ➡️ {match.get('Score', '')} (Vainqueur : **{match['Vainqueur']}**)")
            
            # Vainqueur Final
            final_winner = knockout_results[-1]['Vainqueur']
            st.balloons()
            st.success(f"🏆 LE VAINQUEUR EST : **{final_winner.upper()}** 🏆")
            progress_bar.progress(100)
            status_text.text("Simulation terminée !")
            
            # Export CSV
            csv = pd.DataFrame(knockout_results).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger les résultats (CSV)",
                csv,
                "resultats_simulation.csv",
                "text/csv",
                key='download-csv'
            )

    elif page == "Prédire un Match":
        st.header("🔮 Prédire un Match Spécifique")
        
        # Filtrer pour ne garder que les équipes africaines présentes dans le modèle
        all_teams = sorted(list(current_elo_ratings.keys()))
        african_teams_in_data = [team for team in all_teams if team in AFRICAN_TEAMS]
        
        col1, col2 = st.columns(2)
        with col1:
            team_A = st.selectbox("Équipe A (Domicile)", african_teams_in_data, index=african_teams_in_data.index("Morocco") if "Morocco" in african_teams_in_data else 0)
        with col2:
            team_B = st.selectbox("Équipe B (Extérieur)", african_teams_in_data, index=african_teams_in_data.index("Senegal") if "Senegal" in african_teams_in_data else 1)
            
        is_neutral = st.checkbox("Terrain Neutre ?", value=True)
        
        if st.button("Prédire"):
            winner, prob_win, score_A, score_B = predict_match_result(model, team_A, team_B, current_elo_ratings, mean_points, 1 if is_neutral else 0)
            
            st.markdown("### Résultat Prédit")
            
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            with col_res2:
                st.markdown(f"<h1 style='text-align: center;'>{score_A} - {score_B}</h1>", unsafe_allow_html=True)
                if winner == "Draw":
                    st.info(f"Match Nul prédit ({prob_win:.1%} de confiance)")
                else:
                    st.success(f"Vainqueur : **{winner}** ({prob_win:.1%} de confiance)")
            
            # Visualisation des probabilités
            X = prepare_match_features(team_A, team_B, current_elo_ratings, mean_points, 1 if is_neutral else 0)
            proba = model.predict_proba(X)[0]
            
            probs_df = pd.DataFrame({
                'Résultat': [team_B, 'Nul', team_A],
                'Probabilité': [proba[0], proba[1], proba[2]]
            })
            
            fig = px.bar(probs_df, x='Résultat', y='Probabilité', color='Résultat', 
                         title="Probabilités du Match", text_auto='.1%')
            st.plotly_chart(fig)

    elif page == "Statistiques & Visualisations":
        st.header("📊 Statistiques & Visualisations")
        
        # Filtrer les données pour les équipes africaines
        african_elo_data = {k: v for k, v in current_elo_ratings.items() if k in AFRICAN_TEAMS}
        
        tab1, tab2 = st.tabs(["Classement ELO (Afrique)", "Historique Équipe"])
        
        with tab1:
            st.subheader("Classement ELO Actuel (Top 20 Africain)")
            elo_df = pd.DataFrame(list(african_elo_data.items()), columns=['Équipe', 'ELO'])
            elo_df = elo_df.sort_values('ELO', ascending=False).reset_index(drop=True)
            
            # Top 20 Chart
            fig = px.bar(elo_df.head(20), x='ELO', y='Équipe', orientation='h', 
                         title="Top 20 Équipes Africaines par Score ELO", color='ELO',
                         color_continuous_scale='Viridis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig)
            
            st.dataframe(elo_df)
            
        with tab2:
            st.subheader("Historique ELO par Équipe")
            teams = sorted(list(african_elo_data.keys()))
            selected_team = st.selectbox("Choisir une équipe pour voir son historique", teams)
            
            history_df = get_elo_history(selected_team)
            
            if not history_df.empty:
                fig = px.line(history_df, x='date', y='elo', title=f"Évolution du score ELO - {selected_team}")
                st.plotly_chart(fig)
                
                current_elo = current_elo_ratings.get(selected_team, "N/A")
                st.metric("Score ELO Actuel", f"{current_elo:.0f}")
            else:
                st.warning("Pas d'historique disponible pour cette équipe.")

if __name__ == "__main__":
    main()
