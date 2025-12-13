# ==============================================================================
# scripts/elo_ranking.py
# Script pour calculer le classement Elo dynamique pour tous les matchs historiques.
# ==============================================================================

import pandas as pd
import numpy as np
import sys
import os

# --- 1. Constantes et Paramètres ELO ---
# K_FACTOR: Détermine la volatilité des scores (impact d'un seul match).
K_FACTOR = 30  

# HOME_FIELD_ADVANTAGE: Avantage de jouer à domicile (en points Elo).
HOME_FIELD_ADVANTAGE = 100 

# INITIAL_ELO: Score de départ pour toutes les équipes.
INITIAL_ELO = 1500 

# --- 2. Fonctions de Calcul ELO ---

def calculate_elo_probability(elo_A, elo_B):
    """Calcule la probabilité qu'une équipe A gagne contre une équipe B."""
    # Formule standard de probabilité ELO : 1 / (1 + 10^((Elo_B - Elo_A) / 400))
    return 1.0 / (1 + 10**((elo_B - elo_A) / 400.0))
    # 

def update_elo(elo_A, elo_B, score_A, score_B, K, HFA):
    """Met à jour les scores Elo après un match."""
    
    # 1. Ajuster l'Elo de l'équipe à domicile (A) avec l'avantage du terrain (HFA)
    elo_A_adjusted = elo_A + HFA

    # 2. Calculer la probabilité de victoire de l'équipe A
    expected_A = calculate_elo_probability(elo_A_adjusted, elo_B)
    
    # 3. Déterminer le résultat réel (S)
    if score_A > score_B:
        S_A = 1.0  # Victoire A
    elif score_A < score_B:
        S_A = 0.0  # Défaite A
    else:
        S_A = 0.5  # Match nul

    # 4. Calculer la modification de l'Elo
    # 
    change_A = K * (S_A - expected_A)
    change_B = -change_A
    
    # 5. Calculer le nouvel Elo
    new_elo_A = elo_A + change_A
    new_elo_B = elo_B + change_B
    
    return new_elo_A, new_elo_B, change_A

# --- 3. Processus Principal ELO ---

def run_elo_calculation():
    """Charge les données, boucle sur les matchs, et calcule les scores Elo."""
    
    print("--- Démarrage du Calcul ELO ---")
    
    # 3.1. Charger les données nettoyées
    # CHEMIN CORRIGÉ pour la LECTURE (relatif au CWD qui est PredictCAN/)
    try:
        # Assurez-vous que le nom du fichier est bien 'can_processed_data.csv'
        df = pd.read_csv('./data/processed/can_processed_data.csv')
    except FileNotFoundError:
        print("Erreur: Le fichier can_processed_data.csv est introuvable. Veuillez vérifier que le chemin './data/processed/' est correct.")
        sys.exit()
    
    # Vérifications et tri
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    
    # Initialiser le dictionnaire de scores ELO
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo_ratings = {team: INITIAL_ELO for team in all_teams}
    
    elo_home_before = []
    elo_away_before = []
    
    # 3.2. Boucler sur chaque match (le calcul ELO)
    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Récupérer les scores ELO actuels
        elo_A = elo_ratings[home_team]
        elo_B = elo_ratings[away_team]
        
        # Stocker les scores ELO avant le match
        elo_home_before.append(elo_A)
        elo_away_before.append(elo_B)
        
        # Mettre à jour les scores ELO
        new_elo_A, new_elo_B, change = update_elo(
            elo_A, 
            elo_B, 
            row['home_score'], 
            row['away_score'], 
            K_FACTOR, 
            HOME_FIELD_ADVANTAGE
        )
        
        # Mettre à jour le dictionnaire global
        elo_ratings[home_team] = new_elo_A
        elo_ratings[away_team] = new_elo_B
        
    # 3.3. Ajouter les colonnes ELO au DataFrame
    df['home_elo'] = elo_home_before
    df['away_elo'] = elo_away_before
    
    # 3.4. Sauvegarder le DataFrame enrichi
    # CHEMIN CORRIGÉ pour la SAUVEGARDE (relatif au CWD qui est PredictCAN/)
    df.to_csv('./data/processed/can_processed_data_with_elo.csv', index=False)
    
    print("\nCalcul ELO terminé. Fichier sauvegardé dans data/processed/can_processed_data_with_elo.csv")
    print(f"Scores Elo finaux des équipes de la CAN 2025:")
    
    # 3.5. Afficher les scores ELO finaux
    can_teams_list = [
        'Morocco', 'Mali', 'Zambia', 'Comoros', 'Egypt', 'South Africa', 
        'Angola', 'Zimbabwe', 'Nigeria', 'Tunisia', 'Uganda', 'Tanzania', 
        'Senegal', 'DR Congo', 'Benin', 'Botswana', 'Algeria', 'Burkina Faso', 
        'Equatorial Guinea', 'Sudan', 'Ivory Coast', 'Cameroon', 'Gabon', 'Mozambique'
    ]
    
    final_elo_scores = {team: elo_ratings.get(team, INITIAL_ELO) for team in can_teams_list}
    
    sorted_elo = sorted(final_elo_scores.items(), key=lambda item: item[1], reverse=True)
    for team, score in sorted_elo:
        print(f"  {team}: {score:.2f}")

# Exécution du script
if __name__ == '__main__':
    run_elo_calculation()