import logging
import os
from datetime import datetime

from train_model import main as train_main

REPORTS_DIR = "reports"


def run():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=f"{REPORTS_DIR}/retrain.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info("Début du réentraînement planifié")
    try:
        train_main()
        logging.info("Réentraînement terminé avec succès")
    except Exception as exc:
        logging.exception("Échec du réentraînement : %s", exc)
        raise


if __name__ == "__main__":
    run()
