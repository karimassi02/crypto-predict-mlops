# Crypto Predict MLOps

Systeme de prediction de tendances pour les cryptomonnaies, integrant collecte multi-sources, apprentissage automatique et pipeline MLOps complet.

> **Projet M2 Data Science** - Certification RNCP39586 "Ingenieur en science des donnees, specialisation apprentissage automatique"

---

## Objectif

Developper un systeme end-to-end de prediction des prix crypto :

1. **Collecte automatique** de donnees multi-sources (API, web scraping, bases de donnees)
2. **Stockage** dans une architecture SQL + NoSQL
3. **Pipeline ETL** orchestre avec Apache Airflow
4. **Feature engineering** avec indicateurs techniques et sentiment
5. **Modeles ML** (Scikit-learn, XGBoost, PyTorch LSTM)
6. **Optimisation** des hyperparametres (GridSearch + Optuna)
7. **Deploiement** via API FastAPI + CI/CD GitHub Actions
8. **Monitoring** avec detection de drift (Evidently AI)
9. **Dashboard** interactif Streamlit

---

## Stack Technique

| Categorie | Technologies |
|-----------|-------------|
| Langage | Python 3.12 |
| Collecte API | requests, CoinGecko API, Fear & Greed Index |
| Web Scraping | BeautifulSoup, requests |
| Base SQL | PostgreSQL |
| Base NoSQL | MongoDB |
| ETL / Orchestration | Apache Airflow |
| Transformation | pandas, numpy |
| Feature Engineering | pandas, ta (indicateurs techniques) |
| Feature Selection | scikit-learn (RFE, SelectKBest, PCA) |
| ML Classique | scikit-learn, XGBoost |
| Deep Learning | PyTorch (LSTM) |
| Optimisation Hyperparametres | GridSearchCV, Optuna |
| Tracking Experiences | MLflow |
| Sauvegarde Modeles | MLflow, joblib |
| API de Prediction | FastAPI |
| Dashboard | Streamlit, Plotly |
| Visualisation | Plotly, matplotlib, seaborn |
| Tests Statistiques | scipy.stats |
| CI/CD | GitHub Actions |
| Monitoring / Drift | Evidently AI |
| Containerisation | Docker, Docker Compose |
| Gestion de projet | GitHub Projects (Kanban) |
| Securite | python-dotenv, PostgreSQL roles/SSL |

---

## Avancement par Phase

| Phase | Description | Statut |
|-------|------------|--------|
| Phase 1 | Collecte de donnees (API, scraping, automatisation) | Termine |
| Phase 2 | Stockage (PostgreSQL + MongoDB) | Termine |
| Phase 3 | Transformation & ETL (Airflow) | Termine |
| Phase 4 | Securite des donnees | Termine |
| Phase 5 | Analyse & Visualisation (Dashboard) | Termine |
| Phase 6 | Modeles ML & deploiement (Bloc 5) | A faire |
| Phase 7 | Gestion de projet (documentation Bloc 3) | A faire |

---

## Blocs de Competences RNCP39586

### Bloc 1 : Collecter, transformer et securiser des donnees

| Competence | Description | Eliminatoire | Statut |
|------------|------------|:------------:|--------|
| C1.1.1 | Elaborer une strategie de collecte de donnees | | A faire |
| C1.1.2 | Mettre en oeuvre des techniques de collecte (API, BD, scraping) | **OUI** | A faire |
| C1.1.3 | Automatiser la collecte de donnees | | A faire |
| C1.2.1 | Elaborer la strategie de stockage des donnees | | A faire |
| C1.2.2 | Construire une base de donnees (SQL/NoSQL + Big Data) | **OUI** | A faire |
| C1.3.1 | Selectionner les technologies de traitement de donnees | | A faire |
| C1.3.2 | Transformer les donnees (nettoyage) | **OUI** | A faire |
| C1.3.3 | Developper un processus ETL | **OUI** | A faire |
| C1.4.1 | Definir la politique de securisation des donnees | | A faire |
| C1.4.2 | Concevoir une architecture securisee | | A faire |

### Bloc 2 : Analyser, organiser et valoriser des donnees

| Competence | Description | Eliminatoire | Statut |
|------------|------------|:------------:|--------|
| C2.1.1 | Analyser les besoins metier | | A faire |
| C2.1.2 | Definir les axes d'analyse et les metriques | **OUI** | A faire |
| C2.1.3 | Realiser des requetes et des calculs (dashboard, SQL, Python) | **OUI** | A faire |
| C2.1.4 | Elaborer des modeles statistiques et tests d'hypotheses | | A faire |
| C2.2.1 | Representer les donnees (visualisations) | **OUI** | A faire |
| C2.2.2 | Presenter des recommandations | | A faire |
| C2.3.1 | Former les utilisateurs | | A faire |
| C2.3.2 | Rediger la documentation technique | | A faire |

### Bloc 3 : Elaborer et piloter un projet data

| Competence | Description | Eliminatoire | Statut |
|------------|------------|:------------:|--------|
| C3.1.1 | Definir les objectifs et le perimetre du projet | | A faire |
| C3.1.2 | Dimensionner le projet (charge, budget) | **OUI** | A faire |
| C3.1.3 | Rediger la documentation projet | | A faire |
| C3.2.1 | Planifier l'execution du projet (Gantt, RACI) | **OUI** | A faire |
| C3.2.2 | Suivre l'avancement (outil de suivi, KPIs) | **OUI** | A faire |
| C3.3.1 | Evaluer les besoins en competences | | A faire |
| C3.3.2 | Piloter l'equipe projet | | A faire |
| C3.3.3 | Proceder aux arbitrages | | A faire |
| C3.4.1 | Mettre en place un systeme de veille | | A faire |
| C3.4.2 | Integrer les enjeux RSE, ethique, RGPD | | A faire |

### Bloc 5 : Concevoir et deployer des modeles d'apprentissage automatique (Specialite Data Scientist)

| Competence | Description | Eliminatoire | Statut |
|------------|------------|:------------:|--------|
| C5.1.1 | Analyser la problematique et le contexte | | A faire |
| C5.1.2 | Cadrer la strategie de resolution du probleme | | A faire |
| C5.1.3 | Selectionner les technologies et algorithmes | | A faire |
| C5.2.1 | Construire des variables (feature engineering) | **OUI** | A faire |
| C5.2.2 | Selectionner les variables (feature selection) | **OUI** | A faire |
| C5.2.3 | Entrainer un modele d'apprentissage automatique | **OUI** | A faire |
| C5.2.4 | Optimiser la performance (hyperparametres) | **OUI** | A faire |
| C5.3.1 | Sauvegarder le modele (serialisation, versioning) | | A faire |
| C5.3.2 | Deployer via API et CI/CD | | A faire |
| C5.3.3 | Superviser le systeme ML (monitoring, drift) | | A faire |
| C5.3.4 | Automatiser le cycle de vie ML | | A faire |

---

## Structure du Projet

```
crypto-predict-mlops/
|
|-- src/                        # Code source principal
|   |-- data/
|   |   |-- collectors/         # Collecteurs de donnees (API, scraping)
|   |   |-- storage/            # Connexions BD (PostgreSQL, MongoDB)
|   |   +-- etl/                # Pipelines ETL
|   |-- features/               # Feature engineering & selection
|   |-- models/                 # Entrainement, optimisation, evaluation
|   |-- api/                    # FastAPI + Dashboard Streamlit
|   |-- monitoring/             # Drift detection, alertes
|   +-- utils/                  # Configuration, helpers
|
|-- dags/                       # DAGs Apache Airflow
|-- tests/                      # Tests unitaires et integration
|-- notebooks/                  # Jupyter notebooks (EDA, analyse)
|-- data/
|   |-- raw/                    # Donnees brutes
|   +-- processed/              # Donnees transformees
|-- models/                     # Modeles sauvegardes
|-- config/                     # Fichiers de configuration
|-- docker/                     # Dockerfiles et docker-compose
|-- .github/workflows/          # CI/CD GitHub Actions
|-- RNCP ressources/            # Documents de reference certification
|-- requirements.txt
|-- .env.example
+-- README.md
```

---

## Auteur

**Karim Assi** - M2 Data Science / Expert IA - YNOV

---

## Licence

MIT
