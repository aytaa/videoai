# db_connector.py
import mysql.connector
from mysql.connector import pooling

# Doğrudan sabit değerlerle veritabanı yapılandırması (Docker ortamında 'db' kullanılıyor)
db_config = {
    'host': "db",         # "localhost" yerine MySQL container servis adı: "db"
    'port': 3306,
    'user': "root",
    'password': "example",
    'database': "mydatabase"
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
