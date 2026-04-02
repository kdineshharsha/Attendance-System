from calendar import c
from queue import PriorityQueue
import re
import sqlite3
import json
import stat
from turtle import st
import numpy as np
from datetime import datetime


class DBManager:
    def __init__(self, db_name="face_database.db"):
        self.db_name = db_name
        self.create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def create_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY ,
                name TEXT NOT NULL,
                age INTEGER,
                details TEXT,
                embedding TEXT NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                in_time TEXT NOT NULL,
                out_time TEXT,
                early_leave BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )

    def add_user(self, id, name, age, details, embedding):
        conn = self.get_connection()
        cursor = conn.cursor()
        embedding_list = embedding.tolist()
        embedding_json = json.dumps(embedding_list)

        # Add a comment for the face_recognition insted of deepface
        # embedding_json = json.dumps(embedding)
        print(embedding_json)

        cursor.execute(
            """
            INSERT INTO users (id, name, age, details, embedding)
            VALUES (?,?, ?, ?, ?)
        """,
            (id, name, age, details, embedding_json),
        )

        conn.commit()
        conn.close()
        print(f"[DB] {name} added!")
        return True

    def load_users(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, details, embedding FROM users")
        rows = cursor.fetchall()
        users_data = []
        for row in rows:
            user = {
                "id": row[0],
                "name": row[1],
                "age": row[2],
                "details": row[3],
                "embedding": np.array(json.loads(row[4])),
                # "embedding": json.loads(row[4]),
            }
            users_data.append(user)
        return users_data

    def grant_early_leave(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        current_date = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            "UPDATE attendance SET early_leave = 1 WHERE user_id = ? AND date = ?",
            (user_id, current_date),
        )
        conn.commit()
        conn.close()
        print(f"[DB] Early leave granted for user {user_id} on {current_date}")

    def mark_attendance(self, user_id, manual_out=False):
        conn = self.get_connection()
        cursor = conn.cursor()

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")

        cursor.execute(
            "SELECT id, in_time, out_time,early_leave FROM attendance WHERE user_id = ? AND date = ?",
            (user_id, current_date),
        )
        record = cursor.fetchone()

        if record is None:
            cursor.execute(
                "INSERT INTO attendance (user_id, date, in_time) VALUES (?, ?, ?)",
                (user_id, current_date, current_time),
            )
            print(f"[DB] Attendance marked for user {user_id} at {current_time}")
            status = "IN"

        else:
            record_id, in_time, out_time, early_leave = record

            if out_time is None:

                past_12_30 = now.hour > 12 or (now.hour == 12 and now.minute >= 30)
                past_5_00 = now.hour > 17 or (now.hour == 17 and now.minute >= 0)
                if past_12_30 and early_leave == 1:
                    cursor.execute(
                        "UPDATE attendance SET out_time = ? WHERE id = ?",
                        (current_time, record_id),
                    )
                    print(f"[DB] User {user_id} marked as left early at {current_time}")
                    status = "OUT (EARLY)"

                elif past_5_00:
                    cursor.execute(
                        "UPDATE attendance SET out_time = ? WHERE id = ?",
                        (current_time, record_id),
                    )
                    print(f"[DB] User {user_id} marked as left at {current_time}")
                    status = "OUT"
                else:
                    print(f"[⚠️ WARNING] {user_id} You can't Leave before 5.")
                    status = "ALREADY_IN"
            else:
                print(
                    f"[⚠️ WARNING] {user_id} You have already marked out at {out_time}."
                )
                status = "ALREADY OUT"
        conn.commit()
        conn.close()
        return status

    def get_users_for_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT u.id, u.name, u.age, u.details, a.in_time, a.out_time, a.early_leave
            FROM users u
            LEFT JOIN attendance a ON u.id = a.user_id AND a.date = ?
        """,
            (today,),
        )

        rows = cursor.fetchall()
        users_data = []
        for row in rows:
            user = {
                "id": row[0],
                "name": row[1],
                "age": row[2],
                "details": row[3],
                "early_leave": bool(row[4]),
            }
            users_data.append(user)
        print(users_data)
        return users_data

    def get_attendance_for_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT  a.user_id,u.name , a.in_time, a.out_time, a.early_leave
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.date DESC, a.in_time DESC
        """
        )
        rows = cursor.fetchall()
        # print(rows)
        attendance_data = []
        for row in rows:
            record = {
                # "date": row[0],
                "user_id": row[0],
                "name": row[1],
                "in_time": row[2],
                "out_time": row[3],
                "early_leave": bool(row[4]),
            }
            attendance_data.append(record)
        # print(f"{attendance_data}")
        return attendance_data


if __name__ == "__main__":
    db = DBManager()
    users = db.load_users()
    print(users)
