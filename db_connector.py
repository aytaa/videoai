import os
import mysql.connector
from mysql.connector import pooling

# Ortam değişkenleri ile veritabanı yapılandırması
db_config = {
    'host': os.environ.get("DB_HOST", "localhost"),
    'port': int(os.environ.get("DB_PORT", 3306)),
    'user': os.environ.get("DB_USER", "root"),
    'password': os.environ.get("DB_PASSWORD", "Odak1098"),
    'database': os.environ.get("DB_NAME", "mydatabase")
}

connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    **db_config
)


def get_connection():
    try:
        return connection_pool.get_connection()
    except mysql.connector.Error as err:
        print(f"Veritabanı bağlantı hatası: {err}")
        return None
