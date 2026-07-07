import os

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DATABASE", "sentiment_db"),
    )


def load_tweets() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql("SELECT id, text, positive, negative FROM tweets", conn)
    finally:
        conn.close()
    return df


def insert_tweet(text: str, positive: int, negative: int) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            (text, positive, negative),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
