import sys
import os
import joblib
import pandas as pd
import numpy as np

# Ajouter le dossier scripts au path pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from simulate_tournament import load_data, predict_match_result, simulate_group_phase, main as run_simulation
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que le fichier scripts/simulate_tournament.py existe.")
    sys.exit(1)

def predict_custom_match(model, current_elo_ratings, mean_points):
    print("\n--- PRÉDICTION DE MATCH PERSONNALISÉ ---")
    team_A = input("Entrez le nom de l'équipe A (ex: Morocco): ").strip()
    team_B = input("Entrez le nom de l'équipe B (ex: Senegal): ").strip()
    
    # Vérification basique si les équipes existent dans les données ELO
    if team_A not in current_elo_ratings:
        print(f"Attention: L'équipe '{team_A}' n'est pas trouvée dans les données ELO. Utilisation de l'ELO par défaut.")
    if team_B not in current_elo_ratings:
        print(f"Attention: L'équipe '{team_B}' n'est pas trouvée dans les données ELO. Utilisation de l'ELO par défaut.")
        
    is_neutral_input = input("Le match est-il sur terrain neutre ? (o/n) [défaut: o]: ").strip().lower()
    is_neutral = 0 if is_neutral_input == 'n' else 1
    
    print(f"\nSimulation du match : {team_A} vs {team_B}")
    winner, proba = predict_match_result(model, team_A, team_B, current_elo_ratings, mean_points, is_neutral)
    
    print(f"\nRésultat prédit : {winner}")
    if winner == "Draw":
        print(f"Probabilité de match nul : {proba:.2%}")
    else:
        print(f"Probabilité de victoire : {proba:.2%}")

def main():
    print("=============================================")
    print("        PredictCAN - Menu Principal")
    print("=============================================")
    
    # Chargement des données une seule fois
    print("Chargement du modèle et des données...")
    model, current_elo_ratings, mean_points = load_data()
    print("Données chargées avec succès.\n")
    
    while True:
        print("\nChoisissez une option :")
        print("1. Simuler le tournoi complet (CAN 2025)")
        print("2. Prédire un match spécifique")
        print("3. Quitter")
        
        choice = input("\nVotre choix (1-3) : ")
        
        if choice == '1':
            run_simulation()
        elif choice == '2':
            predict_custom_match(model, current_elo_ratings, mean_points)
        elif choice == '3':
            print("Au revoir !")
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

if __name__ == "__main__":
    main()
