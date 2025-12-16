# ==============================================================================
# scripts/simulate_tournament.py
# Simulation complète de la CAN 2025 (VERSION FINALE AVEC CONTRAINTE R16 CORRIGÉE)
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import sys
from collections import defaultdict
import random

# --- Constantes ---
HOME_FIELD_ADVANTAGE_FEATURE = 100 
INITIAL_ELO = 1500

# --- 1. Définition des Groupes (OFFICIELS CAN 2025) ---

CAN_GROUPS = {
    'A': ['Morocco', 'Mali', 'Zambia', 'Comoros'],
    'B': ['Egypt', 'South Africa', 'Angola', 'Zimbabwe'],
    'C': ['Nigeria', 'Tunisia', 'Uganda', 'Tanzania'],
    'D': ['Senegal', 'DR Congo', 'Benin', 'Botswana'],
    'E': ['Algeria', 'Burkina Faso', 'Equatorial Guinea', 'Sudan'],
    'F': ['Ivory Coast', 'Cameroon', 'Gabon', 'Mozambique'],
}

# --- 2. Fonctions de Préparation des Données et de Prédiction ---

def load_data():
    """Charge le modèle et les scores Elo/Points FIFA."""
    try:
        model = joblib.load('./models/logistic_regression_model.joblib')
        df_elo = pd.read_csv('./data/processed/can_processed_data_with_elo.csv')
        df_elo['date'] = pd.to_datetime(df_elo['date'])
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers: {e}")
        print("Assurez-vous que 02_Modeling.ipynb a été exécuté et que le chemin des fichiers est correct.")
        sys.exit()
    
    # 2.1. Calcul des scores ELO actuels
    def get_final_elo_scores(df):
        last_home_match = df.groupby('home_team').last().reset_index()[['home_team', 'home_elo']]
        last_away_match = df.groupby('away_team').last().reset_index()[['away_team', 'away_elo']]
        last_home_match.columns = ['team', 'elo']
        last_away_match.columns = ['team', 'elo']
        final_elos = pd.concat([last_home_match, last_away_match])
        return final_elos.groupby('team')['elo'].max().to_dict()

    current_elo_ratings = get_final_elo_scores(df_elo)
    
    # 2.2. Récupérer les points FIFA moyens
    mean_points = df_elo[['home_points', 'home_team']].rename(columns={'home_points': 'points', 'home_team': 'team'})
    mean_points = mean_points.groupby('team')['points'].mean().to_dict()
    
    return model, current_elo_ratings, mean_points

def prepare_match_features(team_A, team_B, current_elo_ratings, mean_points, is_neutral=1):
    """Prépare le vecteur de features (X) pour un match donné."""
    
    home_elo = current_elo_ratings.get(team_A, INITIAL_ELO)
    away_elo = current_elo_ratings.get(team_B, INITIAL_ELO)
    
    home_points = mean_points.get(team_A, INITIAL_ELO)
    away_points = mean_points.get(team_B, INITIAL_ELO)
    
    elo_diff = home_elo - away_elo
    points_diff = home_points - away_points
    
    # Avantage du terrain pour le Maroc (pays hôte)
    if team_A == 'Morocco' and is_neutral == 0:
        home_adv = HOME_FIELD_ADVANTAGE_FEATURE
    else:
        home_adv = 0
        
    X = pd.DataFrame([[elo_diff, points_diff, home_adv]], 
                     columns=['elo_diff', 'points_diff', 'home_adv'])
    return X

def simulate_score(winner, team_A, team_B):
    """Simule un score réaliste basé sur le vainqueur."""
    if winner == "Draw":
        goals = np.random.choice([0, 1, 2, 3], p=[0.35, 0.4, 0.2, 0.05])
        return goals, goals
    
    # Le perdant marque peu de buts généralement
    loser_goals = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    
    # Le vainqueur marque au moins 1 but de plus
    margin = np.random.choice([1, 2, 3, 4], p=[0.55, 0.3, 0.1, 0.05])
    winner_goals = loser_goals + margin
    
    if winner == team_A:
        return winner_goals, loser_goals
    else:
        return loser_goals, winner_goals

def predict_match_result(model, team_A, team_B, current_elo_ratings, mean_points, is_neutral=1):
    """Prédit le résultat d'un match et retourne le résultat réel (par probabilité) et le score."""
    
    X = prepare_match_features(team_A, team_B, current_elo_ratings, mean_points, is_neutral)
    
    # [P(Away Win), P(Draw), P(Home Win)]
    proba = model.predict_proba(X)[0] 
    
    # Les labels du modèle sont : -1.0 (Away Win), 0.0 (Draw), 1.0 (Home Win)
    labels = model.classes_ 
    
    # 3. Tirage aléatoire pondéré par les probabilités
    result = np.random.choice(labels, p=proba)
    
    winner = "Draw"
    prob_win = proba[1]
    
    # 4. Déterminer le vainqueur
    if result == 1.0:
        winner = team_A
        prob_win = proba[2]
    elif result == -1.0:
        winner = team_B
        prob_win = proba[0]
        
    # 5. Simuler le score
    score_A, score_B = simulate_score(winner, team_A, team_B)
    
    return winner, prob_win, score_A, score_B
    
    
# --- 3. Simulation de la Phase de Groupes ---

def simulate_group_phase(model, current_elo_ratings, mean_points):
    
    group_standings = defaultdict(lambda: {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0})
    group_results = []
    
    # 3.1. Simuler tous les matchs de groupe
    for group_name, teams in CAN_GROUPS.items():
        
        # Initialiser le classement
        for team in teams:
            group_standings[team]['team'] = team
            group_standings[team]['Group'] = group_name
        
        # Chaque équipe joue 3 matchs
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_A = teams[i]
                team_B = teams[j]
                
                is_neutral = 0 if team_A == 'Morocco' or team_B == 'Morocco' else 1
                
                # Prédiction du vainqueur et du score
                winner, proba, score_A, score_B = predict_match_result(model, team_A, team_B, current_elo_ratings, mean_points, is_neutral)

                # Attribuer les points et les statistiques
                if winner == team_A: # Victoire A
                    group_standings[team_A]['Pts'] += 3
                    group_standings[team_A]['W'] += 1
                    group_standings[team_B]['L'] += 1
                elif winner == team_B: # Victoire B
                    group_standings[team_B]['Pts'] += 3
                    group_standings[team_B]['W'] += 1
                    group_standings[team_A]['L'] += 1
                else: # Match Nul
                    group_standings[team_A]['Pts'] += 1
                    group_standings[team_B]['Pts'] += 1
                    group_standings[team_A]['D'] += 1
                    group_standings[team_B]['D'] += 1
                
                # Mettre à jour les buts
                group_standings[team_A]['GF'] += score_A
                group_standings[team_A]['GA'] += score_B
                group_standings[team_B]['GF'] += score_B
                group_standings[team_B]['GA'] += score_A
                
    # 3.2. Finaliser le classement des groupes
    for team in group_standings:
        standing = group_standings[team]
        standing['GD'] = standing['GF'] - standing['GA']
        standing['P'] = standing['W'] + standing['D'] + standing['L']

    df_standings = pd.DataFrame(list(group_standings.values()))
    
    # 3.3. Trier le classement (Points > Différence de buts > Buts marqués)
    df_standings.sort_values(by=['Group', 'Pts', 'GD', 'GF'], 
                             ascending=[True, False, False, False], 
                             inplace=True)
    
    # 3.4. Identifier les qualifiés (1er, 2e et 4 meilleurs 3e)
    group_qualifiers = df_standings.groupby('Group').head(2) 
    
    third_place = df_standings.groupby('Group').nth(2).sort_values(
        by=['Pts', 'GD', 'GF'], 
        ascending=[False, False, False]
    ).head(4).reset_index(drop=True)
    
    # Stocker les qualifiés par position
    qualifiers = {
        '1A': df_standings[df_standings['Group'] == 'A'].iloc[0]['team'],
        '2A': df_standings[df_standings['Group'] == 'A'].iloc[1]['team'],
        '1B': df_standings[df_standings['Group'] == 'B'].iloc[0]['team'],
        '2B': df_standings[df_standings['Group'] == 'B'].iloc[1]['team'],
        '1C': df_standings[df_standings['Group'] == 'C'].iloc[0]['team'],
        '2C': df_standings[df_standings['Group'] == 'C'].iloc[1]['team'],
        '1D': df_standings[df_standings['Group'] == 'D'].iloc[0]['team'],
        '2D': df_standings[df_standings['Group'] == 'D'].iloc[1]['team'],
        '1E': df_standings[df_standings['Group'] == 'E'].iloc[0]['team'],
        '2E': df_standings[df_standings['Group'] == 'E'].iloc[1]['team'],
        '1F': df_standings[df_standings['Group'] == 'F'].iloc[0]['team'],
        '2F': df_standings[df_standings['Group'] == 'F'].iloc[1]['team'],
    }
    
    # 3.5. Déterminer l'appariement des 3e place (Logique officielle CAF/FIFA)
    
    # Récupérer les groupes des 4 meilleurs troisièmes
    best_thirds_groups = sorted(list(third_place['Group']))
    best_thirds_key = "".join(best_thirds_groups)
    
    # Dictionnaire pour retrouver l'équipe 3ème à partir de son groupe
    group_to_third_team = dict(zip(third_place['Group'], third_place['team']))
    
    # Table de correspondance (Combinations -> Adversaires pour 1A, 1B, 1C, 1D)
    # Format: { '1A': 'Groupe3e', '1B': 'Groupe3e', ... }
    mapping_table = {
        "ABCD": {"1A": "C", "1B": "D", "1C": "A", "1D": "B"},
        "ABCE": {"1A": "C", "1B": "A", "1C": "B", "1D": "E"},
        "ABCF": {"1A": "C", "1B": "A", "1C": "B", "1D": "F"},
        "ABDE": {"1A": "D", "1B": "A", "1C": "B", "1D": "E"},
        "ABDF": {"1A": "D", "1B": "A", "1C": "B", "1D": "F"},
        "ABEF": {"1A": "E", "1B": "A", "1C": "B", "1D": "F"},
        "ACDE": {"1A": "C", "1B": "D", "1C": "A", "1D": "E"},
        "ACDF": {"1A": "C", "1B": "D", "1C": "A", "1D": "F"},
        "ACEF": {"1A": "C", "1B": "A", "1C": "F", "1D": "E"},
        "ADEF": {"1A": "D", "1B": "A", "1C": "F", "1D": "E"},
        "BCDE": {"1A": "C", "1B": "D", "1C": "B", "1D": "E"},
        "BCDF": {"1A": "C", "1B": "D", "1C": "B", "1D": "F"},
        "BCEF": {"1A": "E", "1B": "C", "1C": "B", "1D": "F"},
        "BDEF": {"1A": "E", "1B": "D", "1C": "B", "1D": "F"},
        "CDEF": {"1A": "C", "1B": "D", "1C": "F", "1D": "E"},
    }
    
    # Récupérer le mapping correct, ou un par défaut si cas improbable (ex: < 4 troisièmes)
    current_mapping = mapping_table.get(best_thirds_key, {"1A": "C", "1B": "D", "1C": "A", "1D": "B"})
    
    # Définir les adversaires 3èmes
    T_vs_1A = group_to_third_team.get(current_mapping.get("1A"), "Bye")
    T_vs_1B = group_to_third_team.get(current_mapping.get("1B"), "Bye")
    T_vs_1C = group_to_third_team.get(current_mapping.get("1C"), "Bye")
    T_vs_1D = group_to_third_team.get(current_mapping.get("1D"), "Bye")

    # Équipes de position 1 et 2 (12 équipes)
    P1A, P2A = qualifiers['1A'], qualifiers['2A']
    P1B, P2B = qualifiers['1B'], qualifiers['2B']
    P1C, P2C = qualifiers['1C'], qualifiers['2C']
    P1D, P2D = qualifiers['1D'], qualifiers['2D']
    P1E, P2E = qualifiers['1E'], qualifiers['2E']
    P1F, P2F = qualifiers['1F'], qualifiers['2F']

    # Construction du Bracket (Structure standard CAN/Euro 24 équipes)
    # On structure pour que simulate_knockout prenne (Match 1, Match 2) pour faire QF1, etc.
    
    # QF1 : Vainqueur (2A vs 2C) vs Vainqueur (1D vs 3BEF)
    m1 = (P2A, P2C)
    m2 = (P1D, T_vs_1D)
    
    # QF2 : Vainqueur (1B vs 3ACD) vs Vainqueur (1F vs 2E)
    m3 = (P1B, T_vs_1B)
    m4 = (P1F, P2E)
    
    # QF3 : Vainqueur (1A vs 3CDE) vs Vainqueur (2B vs 2F)
    m5 = (P1A, T_vs_1A)
    m6 = (P2B, P2F)
    
    # QF4 : Vainqueur (1C vs 3ABF) vs Vainqueur (1E vs 2D)
    m7 = (P1C, T_vs_1C)
    m8 = (P1E, P2D)
    
    r16_matchs_structured = [m1, m2, m3, m4, m5, m6, m7, m8]

    # Liste ordonnée pour l'affichage
    qualified_teams = [
        m1[0], m1[1], m2[0], m2[1], 
        m3[0], m3[1], m4[0], m4[1], 
        m5[0], m5[1], m6[0], m6[1], 
        m7[0], m7[1], m8[0], m8[1]
    ]

    return df_standings, qualified_teams, r16_matchs_structured


# --- 4. Simulation de la Phase à Élimination Directe ---

def simulate_knockout(r16_matchs_structured, model, current_elo_ratings, mean_points):
    
    knockout_results = []
    
    current_round = r16_matchs_structured
    round_name = "Huitièmes de Finale"
    
    # Boucle pour les tours
    while len(current_round) > 0:
        next_round_teams = []
        
        print(f"\n--- {round_name} ---")
        
        for team_A, team_B in current_round:
            
            # Gestion du cas "Bye" si moins de 4 troisièmes se qualifient (ne devrait pas arriver ici)
            if team_A == "Bye" or team_B == "Bye":
                winner = team_A if team_A != "Bye" else team_B
                print(f"  {team_A} vs {team_B} -> {winner} avance (Bye)")
            else:
                winner, proba, score_A, score_B = predict_match_result(model, team_A, team_B, current_elo_ratings, mean_points, is_neutral=1)
                
                method = ""

                # Gestion du match nul en KO
                if winner == "Draw":
                    # Simulation Prolongation (30% de chance de but)
                    if random.random() < 0.3:
                        # But en prolongation pour l'équipe la plus forte (ELO)
                        X_features = prepare_match_features(team_A, team_B, current_elo_ratings, mean_points, is_neutral=1)
                        proba_solo = model.predict_proba(X_features)[0]
                        
                        if proba_solo[2] > proba_solo[0]: # Avantage A
                             winner = team_A
                             score_A += 1
                        elif proba_solo[0] > proba_solo[2]: # Avantage B
                             winner = team_B
                             score_B += 1
                        else:
                             if random.random() > 0.5:
                                 winner = team_A
                                 score_A += 1
                             else:
                                 winner = team_B
                                 score_B += 1
                        
                        method = " (a.p.)"
                        print(f"  {team_A} vs {team_B} -> {score_A}-{score_B}{method} (Vainqueur: {winner})")

                    else:
                        # Toujours nul -> Tirs au but
                        X_features = prepare_match_features(team_A, team_B, current_elo_ratings, mean_points, is_neutral=1)
                        proba_solo = model.predict_proba(X_features)[0] 
                        
                        if proba_solo[2] > proba_solo[0]: 
                            winner = team_A
                        elif proba_solo[0] > proba_solo[2]:
                            winner = team_B
                        else: 
                            winner = random.choice([team_A, team_B])
                        
                        method = " (t.a.b)"
                        print(f"  {team_A} vs {team_B} -> {score_A}-{score_B}{method} (Vainqueur: {winner})")
                else:
                    print(f"  {team_A} vs {team_B} -> {score_A}-{score_B} (Vainqueur: {winner})")
            
            knockout_results.append({
                'Tour': round_name,
                'Match': f'{team_A} vs {team_B}',
                'Vainqueur': winner,
                'Score': f"{score_A}-{score_B}{method}"
            })
            
            next_round_teams.append(winner)

        # Mettre à jour pour le tour suivant
        next_round = []
        for i in range(0, len(next_round_teams), 2):
            if i + 1 < len(next_round_teams):
                next_round.append((next_round_teams[i], next_round_teams[i+1]))
            
        current_round = next_round
        
        # Logique de changement de nom de tour
        if round_name == "Huitièmes de Finale":
            round_name = "Quarts de Finale"
        elif round_name == "Quarts de Finale":
            round_name = "Demi-Finales"
        elif round_name == "Demi-Finales":
            round_name = "Finale"
        elif round_name == "Finale":
            current_round = [] 
            
    return knockout_results


# --- 5. Fonction Principale ---

def main():
    
    print("--- SIMULATION CAN 2025 DEMARRAGE ---")
    
    # 5.1. Charger le modèle et les données
    model, current_elo_ratings, mean_points = load_data()
    
    # 5.2. Simulation de la phase de groupes
    print("\n=============================================")
    print("         1. SIMULATION PHASE DE GROUPES      ")
    print("=============================================")
    # NOTE: qualified_teams contient désormais les équipes dans l'ordre du bracket R16
    df_standings, qualified_teams_ordered, r16_matchs_structured = simulate_group_phase(model, current_elo_ratings, mean_points)
    
    print("\n--- CLASSEMENT FINAL DES GROUPES (Pts, Diff. Buts) ---")
    for group_name in sorted(df_standings['Group'].unique()):
        print(f"\n--- GROUPE {group_name} ---")
        group_df = df_standings[df_standings['Group'] == group_name]
        # On cache la colonne Group car elle est dans le titre
        print(group_df[['team', 'Pts', 'GD', 'GF', 'GA']].to_string(index=False))
    
    print("\n--- ÉQUIPES QUALIFIÉES POUR LES HUITIÈMES (16) ---")
    print(f"L'ordre des équipes ci-dessous correspond à l'appariement du bracket R16:")
    print(qualified_teams_ordered)
    
    # 5.3. Simulation de la phase à élimination directe
    print("\n=============================================")
    print("        2. SIMULATION PHASE ELIMINATION      ")
    print("      (Contrainte: Équipes du même groupe séparées)  ")
    print("=============================================")
    knockout_results = simulate_knockout(r16_matchs_structured, model, current_elo_ratings, mean_points)
    
    # 5.4. Affichage du résultat final
    df_knockout = pd.DataFrame(knockout_results)
    final_winner = df_knockout.iloc[-1]['Vainqueur']
    
    print("\n=============================================")
    print(f"       🏆 VAINQUEUR DE LA CAN 2025 : {final_winner} 🏆")
    print("=============================================")


# --- Exécution ---
if __name__ == '__main__':
    # REMARQUE: Les graines aléatoires sont commentées pour que les résultats varient.
    # Pour obtenir des résultats reproductibles, décommentez ces lignes :
    # random.seed(42) 
    # np.random.seed(42)
    main()