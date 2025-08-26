import psycopg2
import streamlit as st
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_conn():
    """
    Returns a psycopg2 connection using credentials from Streamlit secrets.
    """
    try:
        # Load database credentials from Streamlit Secrets
        db = st.secrets["DB"]

        conn = psycopg2.connect(
            host=db["DB_HOST"],
            user=db["DB_USER"],
            password=db["DB_PASSWORD"],
            dbname=db["DB_DATABASE"],
            port=db["DB_PORT"]
        )
        return conn

    except Exception as e:
        logger.exception("Failed to connect to DB")
        return None


def init_db():
    """
    Initializes the database by creating the required tables if they do not exist.
    """
    conn = get_db_conn()
    if not conn:
        logger.error("Database connection could not be established. Exiting init_db.")
        return

    try:
        with conn.cursor() as cursor:
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    mfa_secret VARCHAR(255),
                    role VARCHAR(50) NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create user_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    created_at BIGINT NOT NULL
                )
            """)

        conn.commit()
        logger.info("Database initialized successfully.")

    except Exception as e:
        logger.exception("Error while initializing DB")
        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()
