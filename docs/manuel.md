# EmoVision - Manuel d'utilisation

## 1. Description

EmoVision est une application de reconnaissance automatique des émotions
faciales en temps réel. À partir d'une image fixe ou du flux d'une webcam,
elle détecte le visage présent et classifie l'expression parmi sept
émotions de base : colère, dégoût, peur, joie, neutre, tristesse, surprise.

L'application affiche également un histogramme indiquant la probabilité
de chaque émotion selon le modèle, et permet de sauvegarder un instantané
annoté.

## 2. Configuration requise

### Matériel
- Un ordinateur avec processeur récent (Intel Core i5 ou équivalent)
- Au moins 4 Go de RAM disponibles
- Une webcam intégrée ou externe pour le mode temps réel

### Logiciel
- Système d'exploitation : Windows 10/11 ou Linux Ubuntu 22.04+
- Python 3.12 (testé sur 3.12.10)
- Pip à jour

## 3. Installation

### 3.1 Récupérer le code

Cloner le dépôt depuis GitHub :

    git clone https://github.com/renvanor3/EmoVision.git
    cd EmoVision

### 3.2 Créer un environnement virtuel

Sous Windows :

    python -m venv .venv
    .venv\Scripts\activate

Sous Linux :

    python3 -m venv .venv
    source .venv/bin/activate

### 3.3 Installer les dépendances

    pip install --upgrade pip
    pip install -r requirements.txt

L'installation prend quelques minutes (TensorFlow et OpenCV pèsent
plusieurs centaines de Mo).

### 3.4 Récupérer le dataset (optionnel)

Le dataset FER-2013 n'est nécessaire que pour ré-entraîner le modèle.
Pour utiliser l'application avec le modèle pré-entraîné fourni,
cette étape est facultative.

Télécharger l'archive depuis Kaggle :
https://www.kaggle.com/datasets/msambare/fer2013

Décompresser le contenu dans le dossier `data/` du projet. La structure
finale doit être :

    EmoVision/data/train/<émotion>/<images>.png
    EmoVision/data/test/<émotion>/<images>.png

### 3.5 Récupérer le modèle entraîné

Le modèle `final.keras` doit être présent dans le dossier `models/`.
S'il n'y est pas, deux options :

- Télécharger le modèle pré-entraîné fourni séparément (lien partagé
  par les auteurs)
- L'entraîner depuis zéro avec la commande :

      python train/train_model.py

L'entraînement prend plusieurs heures sur CPU.

## 4. Démarrage rapide

Une fois l'installation terminée :

    python gui/app.py

La fenêtre principale s'ouvre. Cliquer sur "Démarrer la webcam" pour
voir immédiatement la reconnaissance en temps réel.

## 5. Description de l'interface

La fenêtre est divisée en trois zones :

### Barre supérieure
Trois boutons :
- **Charger une image** : ouvre une boîte de dialogue pour sélectionner
  une image depuis le disque
- **Démarrer la webcam** : active le flux temps réel (le texte du bouton
  devient "Arrêter la webcam" quand le mode est actif)
- **Capturer un instantané** : sauvegarde l'image actuellement affichée

### Zone d'affichage (centre)
- À gauche : l'image en cours, avec un rectangle vert autour du visage
  détecté et le nom de l'émotion en surimpression
- À droite : un histogramme des sept probabilités, la barre verte
  correspondant à l'émotion prédite

### Barre inférieure
Affiche un message texte avec l'émotion détectée et le pourcentage de
confiance du modèle, ou un message d'erreur le cas échéant.

## 6. Utilisation détaillée

### 6.1 Analyser une image fixe

1. Cliquer sur "Charger une image"
2. Naviguer jusqu'à la photo souhaitée
3. Choisir un fichier au format JPG, JPEG, PNG ou BMP
4. Valider avec "Ouvrir"

L'image s'affiche redimensionnée pour tenir dans la zone, avec
l'annotation et l'histogramme mis à jour.

Si plusieurs visages sont présents, seul le plus grand est analysé,
conformément aux spécifications du projet (mode mono-visage).

### 6.2 Reconnaître en temps réel via webcam

1. S'assurer qu'aucune autre application n'utilise la webcam
2. Cliquer sur "Démarrer la webcam"
3. Autoriser l'accès si Windows ou Linux le demande
4. Se placer face à la caméra, dans des conditions d'éclairage
   correctes
5. Observer la prédiction qui se met à jour en continu

Pour arrêter, cliquer sur "Arrêter la webcam". La caméra est libérée
proprement et redevient disponible pour d'autres applications.

### 6.3 Capturer un instantané

1. Afficher une image (chargée ou via webcam) avec un visage détecté
2. Cliquer sur "Capturer un instantané"

Le fichier PNG est sauvegardé automatiquement dans le dossier
`snapshots/` à la racine du projet. Le nom contient un horodatage
pour éviter les collisions :

    snapshots/snapshot_20260515_143022.png

L'image enregistrée inclut le rectangle vert et le label de l'émotion.

## 7. Comprendre les résultats

### 7.1 Les sept émotions

| Anglais   | Français   | Description                              |
|-----------|------------|------------------------------------------|
| angry     | colère     | Sourcils froncés, mâchoire serrée        |
| disgust   | dégoût     | Nez plissé, lèvre supérieure relevée     |
| fear      | peur       | Yeux écarquillés, bouche entrouverte     |
| happy     | joie       | Sourire, coins de la bouche relevés      |
| neutral   | neutre     | Expression au repos, sans émotion forte  |
| sad       | tristesse  | Coins de la bouche tombants, regard bas  |
| surprise  | surprise   | Sourcils hauts, bouche grande ouverte    |

### 7.2 Interpréter l'histogramme

La barre verte indique l'émotion prédite. Les autres barres en bleu
montrent les "concurrentes" du modèle :

- Si une barre dépasse 70%, le modèle est confiant
- Si deux barres sont proches autour de 40%, le modèle hésite
- Cette hésitation reflète souvent une expression ambiguë

Les confusions fréquentes du modèle sont :
- fear / surprise (yeux écarquillés communs)
- sad / neutral (faible amplitude d'expression)
- angry / disgust (proximité des unités d'action faciale)

## 8. Dépannage

### La webcam ne s'ouvre pas
- Vérifier qu'aucune autre application ne l'utilise (Teams, Zoom,
  Discord, OBS)
- Vérifier les permissions caméra dans Paramètres > Confidentialité
  (Windows) ou Paramètres > Vie privée (Linux)
- Sur certains portables, vérifier la touche physique de désactivation
  de la caméra

### Aucun visage n'est détecté
- Se rapprocher de la caméra ou augmenter la taille du visage à l'image
- Améliorer l'éclairage (éviter le contre-jour)
- Se placer de face, sans lunettes de soleil
- Diminuer le paramètre `min_size` dans `predict/face_detector.py`

### L'application rame ou les FPS sont trop bas
- Augmenter `WEBCAM_DELAY_MS` dans `gui/app.py` (50 ou 100 ms)
- Fermer les autres applications gourmandes en CPU
- Vérifier qu'aucun antivirus ne scanne en arrière-plan

### Erreur "ModuleNotFoundError"
- S'assurer que l'environnement virtuel est activé
- Relancer `pip install -r requirements.txt`
- Si l'erreur concerne `predict` ou `train` : marquer le dossier racine
  comme Sources Root dans PyCharm, ou définir PYTHONPATH

### Erreur "FileNotFoundError" sur final.keras
- Vérifier que le modèle est bien dans `models/final.keras`
- Sinon, mettre à jour `MODEL_PATH` dans `predict/predictor.py` pour
  pointer vers un modèle existant (ex. `baseline.keras`)

## 9. Limitations connues

- Détection limitée aux visages frontaux, peu efficace en profil
- Précision plafonnée à environ 65% du fait du dataset FER-2013
  (faible résolution 48x48, certaines annotations discutables)
- Mode mono-visage : un seul visage analysé par image, même si
  plusieurs sont présents
- Pas de gestion des accessoires importants (masques, lunettes de
  soleil épaisses) qui peuvent bloquer la détection

## 10. Auteurs et contexte

Projet réalisé dans le cadre de l'UE Projet, Double Licence
Mathématiques-Informatique L3, Aix-Marseille Université.

- **Bardak Ali Sait** : préparation des données, modèle CNN, entraînement
- **Bouslah Rayane** : interface graphique, intégration webcam
- **Encadrante** : Line Jakubiec-Jamet

## 11. Licence et dataset

Le code est distribué à des fins pédagogiques.

Le dataset FER-2013 est fourni par Pierre-Luc Carrier et Aaron
Courville pour le ICML 2013 Challenges in Representation Learning,
disponible publiquement sur Kaggle.