








 














I.	Chapitre 1 : Présentation Générale du Projet
1.	Problématique
           Le marché immobilier marocain se caractérise par une forte hétérogénéité des prix, influencée par plusieurs facteurs tels que la localisation géographique (ville et quartier), la superficie du bien, le nombre de pièces, ainsi que le type de logement. Cette diversité rend l’estimation du prix réel d’un bien immobilier complexe et souvent imprécise pour les acheteurs comme pour les vendeurs.
Par ailleurs, les plateformes d’annonces en ligne, notamment Avito, constituent une source riche de données immobilières mises à jour en continu. Cependant, ces données sont principalement utilisées de manière descriptive et ne font pas l’objet d’une exploitation analytique avancée permettant d’en extraire de la valeur décisionnelle.
En l’absence d’un outil automatique, intelligent et fiable de prédiction des prix immobiliers, les acteurs du marché se basent généralement sur des estimations subjectives, des comparaisons approximatives ou l’expérience personnelle. Cette situation peut conduire à des décisions peu optimales, telles que la surévaluation ou la sous-évaluation des biens, ralentissant ainsi les transactions immobilières.
Face à ce constat, il devient nécessaire de concevoir une solution basée sur l’analyse de données et le Machine Learning, capable d’exploiter efficacement les données issues des annonces immobilières afin de fournir une estimation objective et précise des prix des biens immobiliers au Maroc.
Le projet adopte une approche Data Science complète allant de la collecte des données jusqu’au déploiement d’une application exploitable.

2.	Objectif du Projet
L'objectif principal est de créer un système de prédiction automatique des prix immobiliers au Maroc via les données Avito, pour aider vendeurs, acheteurs et agences avec des estimations rapides basées sur localisation, surface et caractéristiques.
Étapes clés du projet
•	Données Avito : Collecte automatisée d'annonces réelles (scraping).
•	Prétraitement : Nettoyage, encodage et normalisation des données.
•	EDA : Analyse statistique et visualisation.
•	Modèle ML : Entraînement et sélection (ex. régression).
•	API FastAPI : Service d'inférence pour prédictions.
•	Interface Streamlit : Outil web interactif.
•	Résultat : Estimation claire des prix pour décision
•	
3.	Méthodologie de travail
 
•	Figure 1 : Architecture globale du système










































II.	Chapitre 2 : Collecte et Préparation des Données
2.1	Présentation de la source de données (Avito)
Les données utilisées dans ce projet proviennent de la plateforme Avito, qui est l’un des principaux sites de petites annonces au Maroc. Avito permet aux particuliers et aux professionnels de publier des annonces immobilières concernant la vente ou la location de biens tels que les appartements, maisons, studios et terrains.
Cette plateforme constitue une source de données riche et pertinente pour l’analyse du marché immobilier marocain, car elle regroupe un grand nombre d’annonces couvrant différentes villes et régions du pays. Chaque annonce contient des informations essentielles telles que le prix, la localisation, la superficie, le nombre de pièces, le type de bien ainsi qu’une description textuelle.
Cependant, les données disponibles sur Avito ne sont pas directement exploitables pour un modèle de Machine Learning, ce qui nécessite une phase de collecte automatique suivie d’un important travail de nettoyage et de préparation.
 
Figure 2 : Exemple d’annonces immobilières sur la plateforme Avito





2.2	Méthodes de collecte des données
2.2.1	Scraping web
La collecte des données a été réalisée par la technique du web scraping, qui consiste à extraire automatiquement des informations à partir de pages web. Cette méthode permet de récupérer un grand volume de données de manière rapide et efficace.
Le processus de scraping comprend les étapes suivantes :
•	Accès aux pages d’annonces immobilières
•	Analyse de la structure HTML des pages
•	Extraction des informations pertinentes (prix, ville, surface, etc.)
•	Stockage des données extraites dans un fichier structuré
Le scraping a été effectué dans le respect d’un rythme raisonnable afin d’éviter toute surcharge du serveur.
2.2.2	Outils utilisés
La collecte des données a été réalisée à l’aide des outils suivants :
•	Python : langage principal du projet, utilisé pour l’automatisation et le traitement des données.
•	Requests : pour envoyer des requêtes HTTP aux pages web.
•	BeautifulSoup : pour analyser et parcourir le code HTML et extraire les informations.
•	Selenium : utilisé lorsque certaines pages sont dynamiques et nécessitent l’exécution de JavaScript.
Les données collectées sont stockées dans un fichier CSV nommé appartements-data-db.csv.
 
Figure 3:Processus de collecte des données par scraping web


2.3	Description du dataset brut
Le dataset brut, nommé appartements-data-db.csv, contient l’ensemble des annonces immobilières collectées depuis Avito sans aucun traitement préalable.
Principales variables du dataset brut :
•	Price : prix affiché du bien immobilier
•	city : ville du bien
•	surface_area : superficie en mètres carrés
•	nb_rooms : nombre de chambres
•	nb baths : nombre de salles de bain
•	équipement : type de bien (appartement, maison, etc.)
Ce dataset contient plusieurs imperfections telles que :
•	Des valeurs manquantes
•	Des doublons
•	Des formats non uniformes
•	Des valeurs aberrantes








2.4	Nettoyage des données
Le nettoyage des données est une étape essentielle afin de garantir la qualité des données utilisées pour l’apprentissage du modèle.
2.4.1	Traitement des valeurs manquantes
Les valeurs manquantes ont été traitées selon leur importance :
•	Imputation par la moyenne ou la médiane pour les variables numériques
•	Imputation par la valeur la plus fréquente pour les variables catégorielles
•	Suppression des lignes contenant trop de valeurs manquantes
2.4.2	Suppression des doublons
Certaines annonces apparaissent plusieurs fois sur la plateforme. Les doublons ont été détectés et supprimés afin d’éviter un biais dans l’apprentissage du modèle.
2.4.3	Normalisation des données
Les données numériques telles que le prix et la superficie ont été normalisées afin de réduire l’influence des différences d’échelle et d’améliorer les performances des algorithmes de Machine Learning.
Le dataset nettoyé est enregistré dans un fichier data/processed/data_cleaned.csv.
2.5	Prétraitement et encodage des variables
Afin de rendre les données compatibles avec les modèles de Machine Learning, une phase de prétraitement a été appliquée.
Les variables catégorielles telles que la ville, le quartier et le type de bien ont été transformées en variables numériques à l’aide des techniques suivantes :
•	One-Hot Encoding pour les variables avec peu de modalités
•	Label Encoding pour certaines variables ordinales
Les variables numériques ont été standardisées pour garantir une meilleure convergence des modèles.
Cette étape permet d’obtenir un dataset final prêt pour l’analyse exploratoire et la modélisation.
 
Figure 6:Processus de prétraitement et d’encodage des variables


Résumé du Chapitre
Ce chapitre a présenté la source des données, les méthodes de collecte, ainsi que les différentes étapes de nettoyage et de prétraitement. Ces étapes constituent la base fondamentale pour garantir la qualité et la fiabilité des résultats obtenus lors de la phase de modélisation.














































III.	Chapitre 3 : Analyse Exploratoire des Données (EDA)
3.1	Objectifs de l’analyse exploratoire
L’analyse exploratoire des données (Exploratory Data Analysis – EDA) constitue une étape essentielle avant la phase de modélisation. Elle permet de comprendre la structure des données, d’identifier les tendances principales du marché immobilier marocain et de détecter d’éventuelles anomalies ou incohérences.
 Les objectifs principaux de l’EDA dans ce projet sont :
•	Comprendre la distribution des prix immobiliers
•	Analyser l’impact des différentes variables sur le prix
•	Identifier les relations et corrélations entre les variables
•	Détecter les valeurs aberrantes susceptibles d’influencer le modèle
•	Fournir des informations utiles pour le choix des variables et des modèles de Machine Learning

3.2	Analyse statistique descriptive
L’analyse statistique descriptive permet de résumer les principales caractéristiques du dataset à l’aide d’indicateurs statistiques.
Pour les variables numériques telles que le prix, la superficie et le nombre de chambres, les indicateurs suivants ont été calculés :
•	Moyenne
•	Médiane
•	Écart-type
•	Minimum et maximum
•	Quartiles
Cette analyse met en évidence une grande dispersion des prix, reflétant la diversité du marché immobilier marocain. Certaines villes présentent des prix moyens nettement plus élevés que d’autres, ce qui confirme l’importance de la localisation dans la détermination du prix.
Figure 3.1 : Statistiques descriptives des variables numériques
Description de l’image : Tableau récapitulatif (DataFrame Pandas ou Excel) présentant les statistiques principales des variables numériques.



3.3	Visualisation des données
La visualisation des données permet de représenter graphiquement les informations afin de mieux interpréter les tendances et relations.
3.3.1	Distribution des prix
La distribution des prix immobiliers a été analysée à l’aide d’un histogramme. Cette visualisation montre une distribution asymétrique à droite, indiquant la présence de biens à prix très élevés.
Cette observation justifie l’application éventuelle de transformations ou de techniques de normalisation lors de la modélisation. 
Figure 7:Distribution des prix immobiliers

3.3.2	Analyse par ville
Une analyse comparative des prix moyens par ville a été réalisée à l’aide de graphiques en barres. Cette visualisation met en évidence des écarts significatifs entre les grandes villes telles que Casablanca, Rabat et Marrakech, et les villes de plus petite taille.
La ville apparaît ainsi comme l’un des facteurs les plus influents dans la variation des prix immobiliers. 
Figure 8:Prix moyens des biens immobiliers par ville


3.3.3	Corrélations entre variables
Une matrice de corrélation a été utilisée pour analyser les relations entre les différentes variables numériques du dataset. Les résultats montrent une corrélation positive entre le prix et la superficie, ce qui confirme une relation intuitive entre ces deux variables.
D’autres variables, telles que le nombre de chambres, présentent également une corrélation modérée avec le prix. 
Figure 9:Matrice de corrélation des variables numériques



3.4	Détection des valeurs aberrantes
La détection des valeurs aberrantes (outliers) est essentielle afin d’identifier les observations extrêmes pouvant fausser l’apprentissage du modèle.
Des boxplots ont été utilisés pour visualiser les valeurs extrêmes, notamment pour les variables prix et superficie. Certains biens présentent des prix anormalement élevés ou des superficies très importantes, ce qui peut correspondre à des biens de luxe ou à des erreurs de saisie.
Selon le cas, ces valeurs ont été soit conservées, soit traitées à l’aide de techniques de limitation ou de transformation. 
Figure 10:Détection des valeurs aberrantes à l’aide de boxplots

3.5	Interprétation des résultats
Les résultats de l’analyse exploratoire confirment que le prix immobilier est fortement influencé par la localisation géographique, la superficie et le type de bien. L’EDA a permis d’identifier les variables les plus pertinentes pour la phase de modélisation et de mettre en évidence la nécessité de traiter les valeurs aberrantes et les distributions asymétriques.
Ces analyses constituent une base solide pour la construction de modèles de Machine Learning performants et justifient les choix méthodologiques adoptés dans les chapitres suivants.

  Conclusion du Chapitre
Ce chapitre a permis de mieux comprendre les données immobilières collectées, d’identifier les tendances du marché et de préparer efficacement les données pour la phase de modélisation. L’EDA joue ainsi un rôle central dans la réussite du système de prédiction des prix immobiliers.
































IV.	Chapitre 4 : Modélisation et Prédiction
4.1	Préparation des données pour la modélisation
		Avant d’entraîner les modèles de Machine Learning, les données ont été préparées afin de garantir leur compatibilité avec les algorithmes utilisés. Cette étape comprend la sélection des variables explicatives pertinentes issues de l’analyse exploratoire, ainsi que la séparation entre les variables indépendantes (features) et la variable cible (prix immobilier).
		Les variables catégorielles encodées et les variables numériques normalisées constituent l’entrée finale du modèle. Cette préparation permet d’améliorer la convergence des algorithmes et de réduire l’impact des différences d’échelle entre les variables.
Figure 4.1 : Données préparées pour la phase de modélisation
Description de l’image : Schéma illustrant la transformation du dataset nettoyé en matrices X (features) et y (prix).

4.2	Séparation des données (Train / Test)
		Afin d’évaluer les performances des modèles de manière objective, le dataset a été divisé en deux ensembles distincts :
•	Ensemble d’entraînement (80%) : utilisé pour entraîner les modèles
•	Ensemble de test (20%) : utilisé pour évaluer les performances
		Cette séparation permet de mesurer la capacité de généralisation du modèle sur des données jamais vues.
Figure 4.2 : Séparation du dataset en ensembles d’entraînement et de test
Description de l’image : Diagramme montrant la division du dataset en 80% train et 20% test.

4.3	Modèles de Machine Learning utilisés
		Plusieurs modèles de Machine Learning ont été testés afin de comparer leurs performances et sélectionner le plus adapté au problème de prédiction des prix immobiliers.
4.3.1	Régression linéaire
		La régression linéaire a été utilisée comme modèle de base. Elle permet de modéliser la relation entre les variables explicatives et la variable cible de manière simple et interprétable. Ce modèle sert de référence pour comparer les performances des modèles plus complexes.

4.3.2	Ridge et Lasso Regression
		Les modèles Ridge et Lasso sont des variantes de la régression linéaire qui incluent une régularisation pour éviter le surapprentissage. Ridge utilise une régularisation L2 tandis que Lasso utilise une régularisation L1 permettant également la sélection automatique de variables.

4.3.3	Arbre de décision
		Le modèle d'arbre de décision construit une structure arborescente pour prendre des décisions basées sur les caractéristiques des données. Il est capable de capturer des relations non linéaires complexes.

4.3.4	Random Forest
		Le modèle Random Forest repose sur un ensemble d'arbres de décision. Il est capable de capturer des relations non linéaires et de gérer efficacement les interactions entre les variables. Ce modèle est généralement performant pour les problèmes de prédiction sur des données structurées.

4.3.5	Gradient Boosting
		Le modèle Gradient Boosting construit séquentiellement des arbres de décision faibles, chaque nouvel arbre corrigeant les erreurs du précédent. C'est un modèle très performant pour les tâches de régression.
Figure 4.3 : Comparaison des modèles de Machine Learning utilisés
Description de l’image : Tableau ou graphique comparant les différents modèles testés.

4.4	Entraînement des modèles
		Chaque modèle a été entraîné sur l’ensemble d’entraînement en utilisant des hyperparamètres par défaut, puis optimisés à l’aide de techniques telles que la validation croisée et la recherche de paramètres (GridSearch ou RandomizedSearch).
		Cette phase vise à améliorer les performances des modèles tout en évitant le sur-apprentissage (overfitting).
Figure 4.4 : Processus d’entraînement des modèles
Description de l’image : Schéma illustrant l’entraînement et l’optimisation des modèles.

4.5	Évaluation des performances
		Les performances des modèles ont été évaluées sur l’ensemble de test à l’aide de plusieurs métriques.
4.5.1	RMSE (Root Mean Squared Error)
		Le RMSE mesure l’erreur moyenne quadratique entre les valeurs réelles et les valeurs prédites. Une valeur faible indique une meilleure précision du modèle.
4.5.2	MAE (Mean Absolute Error)
		Le MAE représente la moyenne des erreurs absolues. Il est moins sensible aux valeurs aberrantes que le RMSE.
4.5.3	R² (Coefficient de détermination)
		Le coefficient R² indique la proportion de la variance expliquée par le modèle. Une valeur proche de 1 signifie un bon ajustement.
 Figure 4.5 : Évaluation des performances des modèles
Description de l’image : Tableau ou graphique comparant les métriques RMSE, MAE et R² pour chaque modèle.

4.6	Choix du modèle final
		Après analyse comparative des performances, le modèle offrant le meilleur compromis entre précision, robustesse et capacité de généralisation a été sélectionné comme modèle final. Le modèle Gradient Boosting s'est révélé être le plus performant dans la majorité des cas, avec des valeurs de RMSE et de MAE inférieures et un score R² plus élevé.
		Ce modèle a été retenu pour la phase de déploiement et sauvegardé afin d’être utilisé par l’API de prédiction.
Figure 4.6 : Sélection du modèle final
Description de l’image : Schéma mettant en évidence le modèle choisi parmi les modèles testés.

		Conclusion du Chapitre
		Ce chapitre a présenté les différentes étapes de la modélisation et de la prédiction des prix immobiliers. Les expérimentations réalisées ont permis d’identifier le modèle le plus adapté, garantissant des performances satisfaisantes et une bonne capacité de généralisation.






























	














V.	Chapitre 5 : Implémentation en Python
5.1	Architecture du projet
	L'architecture du projet a été conçue de manière modulaire afin d'assurer une bonne lisibilité, une facilité de maintenance et une évolutivité du système. Chaque composant du projet est organisé dans un dossier spécifique correspondant à une étape du pipeline de traitement des données et de prédiction.
	Cette architecture permet de séparer clairement :
•	la collecte et la préparation des données (data/),
•	la phase d'analyse et de modélisation (notebooks/),
•	le déploiement du modèle via une API (backend/),
•	l'interface utilisateur (frontend/),
•	les tests et la validation (tests/).

Structure générale du projet :
```
saleshouses/
├── backend/           # API FastAPI
│   ├── main.py       # Point d'entrée de l'API
│   └── models/       # Modèles sauvegardés
├── frontend/         # Interface Streamlit
│   └── app.py       # Application web
├── notebooks/        # Pipeline ML complet
│   └── script.py    # Script de modélisation
├── data/            # Données brutes et traitées
│   ├── appartements-data-db.csv
│   └── processed/
├── models/          # Artefacts du modèle
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── tests/           # Tests unitaires
├── reports/         # Métriques et rapports
├── visualizations/  # Graphiques générés
└── pyproject.toml   # Configuration du projet
```
 
5.2	Organisation des scripts Python
	Les scripts Python sont organisés selon leurs fonctionnalités afin de faciliter le développement et les tests.
•	notebooks/script.py : contient le pipeline complet de machine learning depuis le chargement des données jusqu'à la sauvegarde du modèle entraîné.
•	backend/main.py : point d'entrée de l'API FastAPI pour la prédiction en temps réel.
•	frontend/app.py : interface utilisateur Streamlit permettant d'interagir avec le modèle via l'API.
•	tests/ : dossier contenant l'ensemble des tests unitaires pour valider les fonctionnalités.
	Cette organisation garantit une séparation claire des responsabilités et améliore la maintenabilité du code.
5.3	Bibliothèques utilisées
	Le projet repose sur plusieurs bibliothèques Python, choisies pour leur robustesse et leur popularité dans le domaine de la science des données et du Machine Learning :
•	Pandas : manipulation et analyse des données
•	NumPy : calculs numériques et algèbre linéaire
•	Matplotlib / Seaborn : visualisation des données et création de graphiques
•	Scikit-learn : modélisation, entraînement, évaluation et prétraitement des données
•	Joblib : sauvegarde et chargement efficace des modèles
•	FastAPI : création de l'API REST pour la prédiction
•	Streamlit : développement de l'interface web interactive
•	Pytest : framework de tests unitaires et d'intégration
•	Pydantic : validation des données et sérialisation
•	Uvicorn : serveur ASGI pour déployer l'API FastAPI



5.4	Implémentation du modèle
	L’implémentation du modèle de prédiction a été réalisée à l’aide de la bibliothèque Scikit-learn. Après la phase d’entraînement et de sélection du meilleur modèle, celui-ci est encapsulé dans un pipeline intégrant le prétraitement des données et la prédiction.
Le modèle reçoit en entrée les caractéristiques du bien immobilier (ville, surface, nombre de chambres, etc.) et retourne une estimation du prix.
Cette implémentation garantit :
•	une prédiction cohérente avec les données d’entraînement,
•	une intégration simple avec l’API FastAPI,
•	une réutilisation facile du modèle dans différents contextes.

Figure 5.2 : Pipeline d’implémentation du modèle de prédiction
Description de l’image : Schéma montrant l’enchaînement : Entrée utilisateur → Prétraitement → Modèle → Prix prédit.
5.5	Sauvegarde et chargement du modèle
	Afin de réutiliser le modèle entraîné sans avoir à le recalculer à chaque exécution, celui-ci est sauvegardé sous forme de fichier à l’aide de la bibliothèque Joblib.
La sauvegarde du modèle permet :
•	une utilisation rapide lors de l’inférence,
•	un déploiement simplifié,
•	une meilleure gestion des ressources.
Lors du lancement de l’API, le modèle est automatiquement chargé en mémoire et utilisé pour effectuer les prédictions en temps réel.
Figure 5.3 : Processus de sauvegarde et de chargement du modèle
Description de l’image : Diagramme illustrant la sauvegarde du modèle entraîné et son chargement par l’API.
Conclusion du Chapitre
	Ce chapitre a présenté l’implémentation complète du projet en Python, depuis l’architecture du code jusqu’à la sauvegarde du modèle. L’approche modulaire adoptée facilite la compréhension, la maintenance et l’évolution future du système de prédiction des prix immobiliers.



	


	




VI.	Chapitre 6 : Déploiement et Interface Utilisateur
6.1	Présentation de l’API FastAPI
Afin de rendre le modèle de prédiction accessible et exploitable par des applications externes, une API REST a été développée à l’aide du framework FastAPI. Ce choix est motivé par la rapidité d’exécution, la simplicité de mise en œuvre et la génération automatique de documentation interactive.
L’API joue le rôle d’intermédiaire entre le modèle de Machine Learning et l’interface utilisateur. Elle reçoit les données du bien immobilier, les traite, puis retourne le prix estimé sous forme de réponse JSON.
FastAPI offre également :
•	une validation automatique des données d’entrée,
•	une gestion efficace des requêtes HTTP,
•	une documentation interactive accessible via Swagger UI.
Figure 6.1 : Architecture de l’API FastAPI
Description de l’image : Schéma illustrant la communication entre l’utilisateur, l’API FastAPI et le modèle de prédiction.

6.2	Endpoints et fonctionnement
L’API est organisée autour de plusieurs endpoints, chacun ayant une fonction spécifique. Les principaux endpoints mis en place sont :
•	GET / : permet de vérifier que l’API est active.
•	POST /predict : reçoit les caractéristiques du bien immobilier et retourne le prix prédit.
Le fonctionnement de l’endpoint de prédiction se déroule en plusieurs étapes :
1.	Réception des données envoyées par l’utilisateur.
2.	Validation et prétraitement des entrées.
3.	Chargement du modèle entraîné.
4.	Calcul de la prédiction.
5.	Retour du résultat au format JSON.
Cette approche garantit une réponse rapide et fiable, tout en assurant la cohérence des prédictions.
Figure 6.2 : Flux de fonctionnement d’un endpoint de prédiction
Description de l’image : Diagramme montrant les étapes depuis l’envoi de la requête jusqu’à la réponse.

6.3	Interface Web avec Streamlit
Pour faciliter l’utilisation du système par des utilisateurs non techniques, une interface web intuitive a été développée à l’aide de Streamlit. Cette interface permet de saisir facilement les caractéristiques du bien immobilier via des champs interactifs.
Les fonctionnalités principales de l’interface sont :
•	saisie des informations du logement (ville, surface, nombre de pièces, etc.),
•	envoi automatique des données à l’API,
•	affichage clair et instantané du prix estimé.
Streamlit a été choisi pour sa simplicité, sa rapidité de développement et son intégration naturelle avec Python.
Figure 6.3 : Interface utilisateur Streamlit de prédiction
Description de l’image : Capture d’écran de l’interface Streamlit montrant le formulaire de saisie et le résultat de la prédiction.

6.4	Interaction utilisateur – API
L’interaction entre l’utilisateur et le modèle se fait de manière transparente grâce à l’API. Lorsque l’utilisateur renseigne les informations sur l’interface Streamlit et clique sur le bouton de prédiction :
1.	Les données sont envoyées à l’API via une requête HTTP.
2.	L’API traite les données et interroge le modèle.
3.	Le prix estimé est retourné à l’interface.
4.	Le résultat est affiché à l’utilisateur de manière lisible.
Cette interaction garantit une expérience utilisateur fluide et rapide, sans exposition des détails techniques du modèle.
Figure 6.4 : Interaction entre l’utilisateur, Streamlit et l’API
Description de l’image : Schéma illustrant le flux de données entre l’interface web et l’API FastAPI.

6.5	Déploiement de l’application
Le déploiement de l’application a pour objectif de rendre le système accessible depuis n’importe quel environnement. L’application peut être déployée localement ou sur un serveur distant.
Les principales étapes de déploiement sont :
•	installation des dépendances via le fichier pyproject.toml,
•	lancement de l’API FastAPI,
•	démarrage de l’interface Streamlit,
•	configuration de l’environnement d’exécution.
Ce déploiement permet une utilisation pratique du système et ouvre la voie à une mise en production future à plus grande échelle.
Figure 6.5 : Schéma de déploiement de l’application
Description de l’image : Diagramme montrant le déploiement local ou sur serveur, incluant API, modèle et interface utilisateur.

 Conclusion du Chapitre
Ce chapitre a présenté le déploiement du système de prédiction ainsi que la conception de l’interface utilisateur. L’intégration de FastAPI et Streamlit permet de proposer une solution complète, interactive et facilement accessible, répondant aux objectifs du projet.




























VII.	Chapitre 7 : Résultats, Discussion et Limites
7.1	Résultats obtenus
À l’issue de l’entraînement et de l’évaluation des différents modèles de Machine Learning, des résultats satisfaisants ont été obtenus pour la prédiction des prix immobiliers au Maroc. Les performances ont été mesurées à l’aide de plusieurs métriques standards, notamment le RMSE, le MAE et le coefficient de détermination R².
Le modèle Gradient Boosting s'est distingué par ses performances supérieures par rapport aux autres modèles testés, notamment la régression linéaire et Random Forest. Il a permis de capturer les relations non linéaires entre les variables explicatives et le prix immobilier, conduisant à des prédictions plus précises.
Les résultats montrent que le modèle est capable de fournir des estimations cohérentes et proches des prix réels observés sur la plateforme Avito, ce qui confirme la pertinence de l'approche adoptée.
Figure 7.1 : Comparaison des performances des modèles
Description de l'image : Tableau comparant les valeurs RMSE, MAE et R² pour les 6 modèles testés : Linear Regression (R²=0.819), Ridge (R²=0.805), Lasso (R²=0.819), Decision Tree (R²=0.703), Random Forest (R²=0.814), Gradient Boosting (R²=0.827).

7.2	Analyse et interprétation
L’analyse des résultats met en évidence l’influence significative de certaines variables sur le prix immobilier. Parmi les facteurs les plus déterminants figurent la localisation géographique, la superficie du bien et le type de logement.
Les résultats montrent également que :
•	les biens situés dans les grandes villes présentent des prix plus élevés,
•	une augmentation de la superficie entraîne généralement une hausse du prix,
•	certains quartiers ont un impact important sur la valorisation du bien.
Le bon comportement du modèle Gradient Boosting s'explique par sa capacité à construire séquentiellement des arbres de décision, chaque nouvel arbre corrigeant les erreurs du précédent. Cette approche permet de capturer des relations complexes entre variables et de réduire le risque de surapprentissage grâce à l'utilisation d'un ensemble de modèles faibles.
Figure 7.2 : Importance des variables du modèle final
Description de l’image : Graphique en barres montrant l’importance relative des caractéristiques dans la prédiction du prix.

7.3	Limites du projet
Malgré les résultats encourageants, le projet présente certaines limites qu’il convient de souligner :
•	Les données utilisées proviennent d’annonces en ligne, qui peuvent contenir des erreurs ou des informations incomplètes.
•	Le dataset ne couvre pas l’ensemble du marché immobilier marocain, ce qui peut limiter la généralisation des résultats.
•	Certaines variables importantes, telles que l’état du bien ou la proximité des services, ne sont pas toujours disponibles.
•	Le modèle ne prend pas en compte l’évolution temporelle des prix du marché.
Ces limitations peuvent influencer la précision des prédictions et doivent être prises en considération lors de l’interprétation des résultats.

7.4	Difficultés rencontrées
Plusieurs difficultés ont été rencontrées tout au long de la réalisation du projet :
•	Complexité du prétraitement des données immobilières avec de nombreuses valeurs manquantes et aberrantes
•	Gestion des variables catégorielles avec de nombreux équipements et villes différentes
•	Optimisation des hyperparamètres pour obtenir les meilleures performances
•	Implémentation correcte de l'API FastAPI avec gestion des erreurs et validation des données
•	Création d'une interface Streamlit réactive et ergonomique
•	Mise en place d'une architecture de tests unitaires complète
•	Déploiement et configuration de l'environnement de développement
Ces défis ont toutefois permis de renforcer les compétences techniques acquises et de mieux comprendre les contraintes liées à un projet réel de Data Science.

Conclusion du Chapitre
Ce chapitre a permis de présenter les résultats obtenus, d’en analyser la signification, ainsi que de discuter les limites et les difficultés du projet. Malgré certaines contraintes, le système développé démontre l’efficacité de l’utilisation du Machine Learning pour la prédiction des prix immobiliers et constitue une base solide pour des améliorations futures.


















Conclusion Générale et Perspectives
Conclusion Générale

Ce projet avait pour objectif principal de concevoir un système intelligent capable de prédire les prix immobiliers au Maroc à partir des données collectées. À travers une approche complète allant de la collecte des données jusqu'au déploiement d'une application web, le projet a permis de mettre en pratique les concepts fondamentaux de la programmation Python, de la data science et du machine learning.
Les différentes étapes du projet, incluant le nettoyage des données, l'analyse exploratoire, la modélisation avec 6 algorithmes différents, la création d'une API FastAPI et le développement d'une interface Streamlit, ont été réalisées avec succès. Les résultats obtenus montrent que les modèles de machine learning, en particulier le modèle Gradient Boosting, offrent des performances satisfaisantes pour l'estimation des prix immobiliers avec un R² de 0.827.
L'intégration de l'API FastAPI et de l'interface Streamlit a permis de proposer une solution complète, interactive et facilement accessible, répondant aux objectifs du projet initial.
Ce projet illustre ainsi l’intérêt de l’exploitation intelligente des données issues des plateformes d’annonces en ligne pour améliorer la transparence du marché immobilier et faciliter la prise de décision des utilisateurs.

















Bilan du projet
Le projet a permis d’atteindre les objectifs fixés initialement, notamment :
•	La mise en place d'un pipeline complet de machine learning pour la prédiction immobilière
•	La construction d’un dataset propre et exploitable à partir des données brutes
•	La réalisation d’une analyse exploratoire approfondie avec visualisations
•	L’entraînement et la comparaison de six modèles de prédiction différents
•	L'obtention de performances satisfaisantes avec un R² de 0.827 pour le meilleur modèle
•	La création d'une API FastAPI robuste avec validation des données
•	Le développement d’une interface web Streamlit moderne et intuitive
•	La mise en place d'une suite complète de tests unitaires
Le pipeline complet mis en place démontre la faisabilité d’un système de prédiction de prix immobilier basé sur des données réelles et actualisées.


















Apports du projet
Ce projet a apporté plusieurs bénéfices sur les plans académique et technique :
•	Renforcement des compétences en programmation Python et data science
•	Maîtrise du pipeline complet de machine learning (préparation → modélisation → déploiement)
•	Application pratique des algorithmes de régression et d'ensemble learning
•	Utilisation avancée des frameworks FastAPI et Streamlit
•	Implémentation de tests unitaires complets avec pytest
•	Approche complète d'un projet Data Science de bout en bout
•	Développement d'une méthodologie rigoureuse de gestion de projet
•	Expérience concrète dans la manipulation de données réelles complexes
Il a également permis d'acquérir une expérience précieuse dans l'architecture de systèmes de production et la validation de code.



















Perspectives d'amélioration
Plusieurs axes d’amélioration peuvent être envisagés pour enrichir ce projet :
•	Optimisation des hyperparamètres avec recherche en grille plus extensive
•	Intégration de données géospatiales (coordonnées GPS) pour améliorer la précision
•	Utilisation de modèles plus avancés tels que XGBoost ou LightGBM
•	Prise en compte de l'évolution temporelle des prix (séries temporelles)
•	Enrichissement du dataset avec des sources externes (indices économiques, données démographiques)
•	Déploiement sur une infrastructure cloud (AWS, GCP) avec monitoring continu
•	Amélioration de l'interface utilisateur avec des visualisations plus interactives
•	API de feedback utilisateur pour améliorer continuellement le modèle
Ces perspectives ouvrent la voie à un système encore plus robuste et performant.




















Bibliographie
1.	Géron, A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media.
2.	Hastie, T., Tibshirani, R., Friedman, J. The Elements of Statistical Learning. Springer.
3.	Documentation officielle de Scikit-learn : https://scikit-learn.org
4.	Documentation officielle de FastAPI : https://fastapi.tiangolo.com
5.	Documentation officielle de Streamlit : https://streamlit.io
6.	Documentation officielle de Pytest : https://pytest.org
7.	Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
