# API d'Analyse de Sentiments

API Flask qui attribue à chaque tweet un score de sentiment entre **-1** (négatif) et **1**
(positif), à partir d'un modèle de régression logistique entraîné sur des tweets annotés
stockés en base MySQL.

## Installation

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

## Base de données

MySQL doit tourner (ex: via XAMPP). Puis :

```bash
mysql -u root < schema.sql        # cree la base et la table tweets
python seed_data.py               # charge les tweets annotes de depart
```

## Entraîner le modèle

```bash
python train_model.py
```

Charge les tweets depuis MySQL, entraîne 2 modèles (`positive` / `negative`), affiche les
rapports de classification, sauvegarde les matrices de confusion dans `reports/` et les
modèles dans `models/`.

## Lancer l'API

```bash
python app.py
```

### `POST /api/sentiment`

```bash
curl -X POST http://127.0.0.1:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '["Ce produit est incroyable, je le recommande", "Service catastrophique, a fuir"]'
```

Réponse :
```json
{
  "Ce produit est incroyable, je le recommande": 0.49,
  "Service catastrophique, a fuir": -0.33
}
```

## Réentraînement automatique

`retrain_cron.py` relance l'entraînement à partir des données les plus récentes. À
programmer une fois par semaine avec le Planificateur de tâches Windows ou un cronjob :

```bash
python retrain_cron.py
```