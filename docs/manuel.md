# EmoVision - Manuel d'utilisation

## 1. Description

EmoVision est une application de reconnaissance automatique des émotions
faciales en temps réel. À partir d'une image fixe ou du flux d'une
webcam, elle détecte le visage présent et classifie l'expression parmi
sept émotions : colère, dégoût, peur, joie, neutre, tristesse, surprise.

L'application affiche aussi un histogramme des probabilités pour chaque
émotion, et permet de sauvegarder un instantané annoté.

## 2. Configuration requise

- Windows 10 ou 11
- Python 3.12
- Une webcam (intégrée ou USB)
- 4 Go de RAM minimum
- Environ 500 Mo d'espace disque pour les dépendances

## 3. Installation

### 3.1 Cloner le dépôt

    git clone https://github.com/renvanor3/EmoVision.git
    cd EmoVision

### 3.2 Créer un environnement virtuel et installer les dépendances

    python -m venv .venv
    .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt

L'installation prend quelques minutes, TensorFlow et OpenCV étant
volumineux.

### 3.3 Récupérer le modèle entraîné

Le fichier `models/final.keras` est nécessaire pour faire tourner
l'application. Deux options :

- Utiliser le modèle pré-entraîné fourni avec le dépôt
- Le ré-entraîner soi-même avec `python train/train_model.py` (environ
  8 heures sur CPU, nécessite le dataset FER-2013 décompressé dans
  `data/`)

## 4. Lancer l'application

    python gui/EmoVisionApp.py

La fenêtre principale s'ouvre. Cliquer sur "Démarrer la webcam" pour
voir la reconnaissance en temps réel.

## 5. Interface

Trois boutons en haut :

- **Charger une image** : analyse une image depuis le disque (JPG, PNG,
  BMP).
- **Démarrer la webcam** : active le flux temps réel. Le bouton devient
  "Arrêter la webcam" pendant la capture.
- **Capturer un instantané** : sauvegarde l'image annotée courante
  dans `snapshots/` avec un nom horodaté.

Zone centrale : image à gauche avec rectangle vert autour du visage et
nom de l'émotion, histogramme des sept probabilités à droite.

Zone basse : message texte indiquant l'émotion détectée et la
confiance, ou un message d'erreur (par exemple "Aucun visage détecté").

## 6. Les sept émotions

| Anglais   | Français   |
|-----------|------------|
| angry     | colère     |
| disgust   | dégoût     |
| fear      | peur       |
| happy     | joie       |
| neutral   | neutre     |
| sad       | tristesse  |
| surprise  | surprise   |

## 7. Lecture de l'histogramme

La barre verte correspond à l'émotion prédite. Les barres bleues
montrent la confiance du modèle sur les autres classes.

- Une barre au dessus de 70\% indique une prédiction confiante.
- Deux barres proches autour de 40\% indiquent que le modèle hésite
  entre deux émotions, souvent à cause d'une expression ambiguë.

Confusions fréquentes : fear avec surprise, sad avec neutral, angry
avec disgust.

## 8. Limitations connues

- Détection efficace uniquement sur visages frontaux. Le profil et les
  fortes inclinaisons font échouer le détecteur Haar.
- Précision plafond autour de 65\%, limitée par la qualité du dataset
  FER-2013 (48x48, annotations parfois douteuses).
- Mode mono-visage : seul le plus grand visage est analysé si
  plusieurs sont présents.
- Les accessoires comme un masque ou des lunettes de soleil épaisses
  bloquent la détection.

## 9. Auteurs

Projet réalisé dans l'UE Projet, Double Licence Mathématiques-Informatique
L3, Aix-Marseille Université.

- **Bardak Ali Sait** : préparation des données, modèle CNN, entraînement
- **Bouslah Rayane** : interface graphique, intégration webcam
- **Encadrante** : Line Jakubiec-Jamet

Dataset FER-2013 fourni par Pierre-Luc Carrier et Aaron Courville
(ICML 2013), disponible sur Kaggle.