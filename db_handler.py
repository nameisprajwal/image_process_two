import sqlite3
from datetime import datetime
import json

class DatabaseHandler:
    def __init__(self, db_path="retail_data.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            scan_date TIMESTAMP,
            total_products INTEGER
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            product_name TEXT,
            count INTEGER,
            FOREIGN KEY (scan_id) REFERENCES scans (id)
        )
        ''')

        conn.commit()
        conn.close()

    def save_results(self, image_path, results):
        """Save scan results to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert scan record
            cursor.execute('''
            INSERT INTO scans (image_path, scan_date, total_products)
            VALUES (?, ?, ?)
            ''', (image_path, datetime.now(), len(results["products"])))
            
            scan_id = cursor.lastrowid

            # Insert product records
            for product in results["products"]:
                cursor.execute('''
                INSERT INTO products (scan_id, product_name, count)
                VALUES (?, ?, ?)
                ''', (scan_id, product["name"], product["count"]))

            conn.commit()
            print(f"Results saved to database: {self.db_path}")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            conn.rollback()
        
        finally:
            conn.close()

    def get_scan_history(self):
        """Retrieve scan history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT s.id, s.image_path, s.scan_date, s.total_products,
               GROUP_CONCAT(p.product_name || ':' || p.count) as products
        FROM scans s
        LEFT JOIN products p ON s.id = p.scan_id
        GROUP BY s.id
        ORDER BY s.scan_date DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        return results 