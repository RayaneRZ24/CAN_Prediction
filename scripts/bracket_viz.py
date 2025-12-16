import streamlit as st

def display_bracket(knockout_results):
    """
    Affiche le bracket dans Streamlit.
    """
    html = generate_bracket_html(knockout_results)
    st.components.v1.html(html, height=600, scrolling=True)

def generate_bracket_html(knockout_results):
    """
    Génère le code HTML/CSS pour afficher un bracket de tournoi.
    
    Args:
        knockout_results (list): Liste de dictionnaires contenant les résultats des matchs.
                                 Format attendu: [{'Tour': '...', 'Match': 'A vs B', 'Vainqueur': '...', 'Score': '...'}, ...]
    """
    
    # Organiser les données par tour
    rounds = {
        "Huitièmes de Finale": [],
        "Quarts de Finale": [],
        "Demi-Finales": [],
        "Finale": []
    }
    
    for match in knockout_results:
        if match['Tour'] in rounds:
            # Parser les équipes et le score
            teams = match['Match'].split(' vs ')
            team1 = teams[0]
            team2 = teams[1]
            score = match['Score']
            winner = match['Vainqueur']
            
            rounds[match['Tour']].append({
                "team1": team1,
                "team2": team2,
                "score": score,
                "winner": winner
            })

    # CSS pour le bracket
    css = """
    <style>
        .bracket-container {
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            align-items: stretch;
            padding: 20px;
            font-family: sans-serif;
            overflow-x: auto;
        }
        .round {
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            width: 200px;
            margin-right: 20px;
        }
        .match {
            border: 1px solid #ccc;
            border-radius: 5px;
            margin: 10px 0;
            background-color: #f9f9f9;
            padding: 5px;
            position: relative;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .team {
            padding: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .team.winner {
            font-weight: bold;
            background-color: #e6fffa;
            border-radius: 3px;
        }
        .score {
            font-weight: bold;
            margin-left: 10px;
        }
        .round-title {
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        
        /* Connecteurs (Lignes) - Approche simplifiée */
        .round:not(:last-child) .match::after {
            content: '';
            position: absolute;
            right: -20px;
            top: 50%;
            width: 20px;
            height: 1px;
            background-color: #ccc;
        }
        
        /* Dark mode support basic */
        @media (prefers-color-scheme: dark) {
            .match {
                background-color: #262730;
                border-color: #444;
                color: #fff;
            }
            .team.winner {
                background-color: #0e1117;
                color: #4caf50;
            }
            .round-title {
                color: #ddd;
            }
        }
    </style>
    """
    
    html = '<div class="bracket-container">'
    
    # Ordre des tours pour l'affichage
    round_order = ["Huitièmes de Finale", "Quarts de Finale", "Demi-Finales", "Finale"]
    
    for round_name in round_order:
        matches = rounds[round_name]
        html += f'<div class="round"><div class="round-title">{round_name}</div>'
        
        for match in matches:
            # Nettoyer le score pour l'affichage (enlever le texte a.p. ou t.a.b)
            clean_score = match['score'].split('(')[0].strip()
            
            # Déterminer les scores individuels si possible (format "X-Y")
            try:
                s1, s2 = clean_score.split('-')
            except:
                s1, s2 = "?", "?"
            
            # Identifier le vainqueur pour le style
            t1_class = "winner" if match['team1'] == match['winner'] else ""
            t2_class = "winner" if match['team2'] == match['winner'] else ""
            
            # Ajouter un indicateur visuel si prolongation/tab
            extra_info = ""
            if "(a.p.)" in match['score']:
                extra_info = '<div style="font-size:0.7em; text-align:center; color:#888;">(a.p.)</div>'
            elif "(t.a.b)" in match['score']:
                extra_info = '<div style="font-size:0.7em; text-align:center; color:#888;">(t.a.b)</div>'

            html += f"""
            <div class="match">
                <div class="team {t1_class}">
                    <span>{match['team1']}</span>
                    <span class="score">{s1}</span>
                </div>
                <div class="team {t2_class}">
                    <span>{match['team2']}</span>
                    <span class="score">{s2}</span>
                </div>
                {extra_info}
            </div>
            """
        html += '</div>'
    
    html += '</div>'
    
    return css + html
