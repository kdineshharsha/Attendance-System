import sqlite3
import json
from API_manager import APIManager
import numpy as np


class CacheManager:
    def __init__(self, db_name="cache.db"):
        self.db_name = db_name
        self.api = APIManager()
        self.create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def create_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users_cache (
            emp_id TEXT PRIMARY KEY,
            name TEXT,
            designation TEXT,
            embedding TEXT
        )
    """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS offline_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT,
            timestamp TEXT
        )
        """
        )

        conn.commit()
        conn.close()
        print("✅ Local Cache Database Initialized!")

    def sync_users_to_local_db(self):
        print("🔄 Starting Data Sync from Cloud to Local DB...")

        response = self.api.get_users_for_table()

        if response.get("success") == True:
            users_list = response.get("data", [])

            if not users_list:
                print("⚠️ No users found in the cloud database.")
                return {"success": True, "message": "No users to sync."}

            try:
                conn = self.get_connection()
                cursor = conn.cursor()

                cursor.execute("DELETE FROM users_cache")

                sync_count = 0

                for user in users_list:
                    emp_id = user.get("emp_id")
                    name = user.get("name")
                    designation = user.get("designation")
                    face_data = user.get("face_embeddings")

                    if emp_id:
                        embedding = json.dumps(face_data)

                        cursor.execute(
                            """
                            INSERT INTO users_cache (emp_id, name, designation, embedding)
                            VALUES (?, ?, ?, ?)
                        """,
                            (emp_id, name, designation, embedding),
                        )

                        sync_count += 1

                conn.commit()
                conn.close()

                print(f"✅ Successfully synced {sync_count} users to Local Cache!")
                return {
                    "success": True,
                    "message": f"Synced {sync_count} users successfully.",
                }

            except sqlite3.Error as e:
                print(f"❌ SQLite Error during sync: {e}")
                return {"success": False, "message": "Local Database Error."}

        else:
            error_msg = response.get("message", "Unknown API Error")
            print(f"❌ Failed to fetch users from Cloud: {error_msg}")
            return {"success": False, "message": error_msg}

    def load_users(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT emp_id, name, designation, embedding FROM users_cache")
        rows = cursor.fetchall()

        users_data = []
        for row in rows:
            emp_id, name, designation, face_encoding_str = row
            face_encoding = json.loads(face_encoding_str)
            face_encoding = np.array(face_encoding)
            users_data.append(
                {
                    "emp_id": emp_id,
                    "name": name,
                    "designation": designation,
                    "embedding": face_encoding,
                }
            )

        conn.close()
        print(f"✅ Loaded {len(users_data)} users from Local Cache.")
        return users_data

    def mark_attendance(self, emp_id, timestamp):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO offline_attendance (emp_id, timestamp) VALUES (?, ?)",
                (emp_id, timestamp),
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"❌ Error saving offline attendance: {e}")
            return False

    def get_pending_attendance(self):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, emp_id, timestamp FROM offline_attendance")
            rows = cursor.fetchall()
            conn.close()

            pending_records = []
            for row in rows:
                pending_records.append(
                    {"id": row[0], "emp_id": row[1], "timestamp": row[2]}
                )
            print(pending_records)
            return pending_records
        except sqlite3.Error as e:
            print(f"❌ Error fetching pending attendance: {e}")
            return []

    def delete_pending_attendance(self, record_id):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM offline_attendance WHERE id = ?", (record_id,))
            conn.commit()
            conn.close()
            return True
        except sqlite3.Error as e:
            print(f"❌ Error deleting pending attendance: {e}")
            return False


if __name__ == "__main__":
    cache_db = CacheManager()
