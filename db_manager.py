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
                email TEXT,
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
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """
        )

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS leaves(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                leave_type TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shift_start TEXT DEFAULT "08:00:00",
                shift_end TEXT DEFAULT "17:00:00",
                grace_period INTEGER DEFAULT 5,
                min_ot INTEGER DEFAULT 30
            )
        """
        )

        cursor.execute("SELECT COUNT(*) FROM settings")

        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO settings (shift_start, shift_end, grace_period, min_ot) VALUES ('08:00:00', '17:00:00', 5, 30)"
            )

        conn.commit()

    def add_user(self, id, name, age, email, details, embedding):
        conn = self.get_connection()
        cursor = conn.cursor()
        embedding_list = embedding.tolist()
        embedding_json = json.dumps(embedding_list)

        cursor.execute(
            """
            INSERT INTO users (id, name, age,email,details, embedding)
            VALUES (?,?, ?, ?, ?,?)
        """,
            (id, name, age, email, details, embedding_json),
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

    def update_user(self, user_id, name, age, email, details):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE users
                SET name = ?, age = ?, email = ?, details = ?
                WHERE id = ?
            """,
                (name, age, email, details, user_id),
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating user: {e}")
            return False

    def add_leave(self, user_id, date, leave_type, reason):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM leaves WHERE user_id = ? AND date = ?", (user_id, date)
        )
        if cursor.fetchone():
            conn.close()
            return False

        cursor.execute(
            """
            INSERT INTO leaves (user_id, date, leave_type, reason)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, date, leave_type, reason),
        )
        conn.commit()
        conn.close()
        print(f"[DB] Leave added for user {user_id} on {date}")
        return True

    def get_all_leaves(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT l.id, l.user_id, u.name, l.date, l.leave_type, l.reason
            FROM leaves l
            JOIN users u ON l.user_id = u.id
            ORDER BY l.date DESC
        """
        )
        rows = cursor.fetchall()
        leaves_data = []
        for row in rows:
            leave = {
                "id": row[0],
                "user_id": row[1],
                "name": row[2],
                "date": row[3],
                "leave_type": row[4],
                "reason": row[5],
            }
            leaves_data.append(leave)
        conn.close()
        return leaves_data

    def revoke_leave(self, leave_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM leaves WHERE id = ?", (leave_id,))
        conn.commit()
        conn.close()
        print(f"[DB] Leave {leave_id} revoked!")
        return True

    def mark_attendance(self, user_id, manual_out=False):
        conn = self.get_connection()
        cursor = conn.cursor()

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")

        cursor.execute(
            "SELECT id, in_time, out_time FROM attendance WHERE user_id = ? AND date = ?",
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
            record_id, in_time, out_time = record

            if out_time is None:

                cursor.execute(
                    "SELECT leave_type FROM leaves WHERE user_id = ? AND date = ?",
                    (user_id, current_date),
                )
                leave_record = cursor.fetchone()
                has_early_leave = False
                if leave_record and leave_record[0] == "Early Leave":
                    has_early_leave = True

                past_12_00 = now.hour > 12 or (now.hour == 12 and now.minute >= 30)
                # before_12_00 = now.hour < 12 or (now.hour == 12 and now.minute < 30)
                past_5_00 = now.hour > 17 or (now.hour == 17 and now.minute >= 0)
                if past_12_00 and has_early_leave == 1:
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
            SELECT u.id, u.name, u.age,u.email, u.details, a.in_time, a.out_time
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
                "email": row[3],
                "details": row[4],
                "early_leave": bool(row[5]),
            }
            users_data.append(user)
        return users_data

    def get_attendance_for_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT a.user_id, u.name, a.in_time, a.out_time, l.leave_type
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            LEFT JOIN leaves l ON a.user_id = l.user_id AND a.date = l.date
            WHERE a.date = ?
            ORDER BY a.in_time DESC
        """,
            (today,),
        )

        rows = cursor.fetchall()
        attendance_data = []
        for row in rows:
            record = {
                "user_id": row[0],
                "name": row[1],
                "in_time": row[2],
                "out_time": row[3],
                "leave_type": row[4],
            }
            attendance_data.append(record)
        conn.close()
        return attendance_data

    def get_attendance_by_date_range(self, start_date, end_date):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT a.user_id, u.name, a.date, a.in_time, a.out_time, l.leave_type
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            LEFT JOIN leaves l ON a.user_id = l.user_id AND a.date = l.date
            WHERE a.date BETWEEN ? AND ?
            ORDER BY a.date ASC, a.in_time ASC
        """,
            (start_date, end_date),
        )

        rows = cursor.fetchall()
        attendance_data = []
        for row in rows:
            record = {
                "user_id": row[0],
                "name": row[1],
                "date": row[2],
                "in_time": row[3],
                "out_time": row[4],
                "leave_type": row[5],
            }
            attendance_data.append(record)
        conn.close()
        return attendance_data

    def get_leaves_by_date_range(self, start_date, end_date):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT l.id, l.user_id, u.name, l.date, l.leave_type, l.reason
            FROM leaves l
            JOIN users u ON l.user_id = u.id
            WHERE l.date BETWEEN ? AND ?
            ORDER BY l.date ASC
        """,
            (start_date, end_date),
        )

        rows = cursor.fetchall()
        leaves_data = []
        for row in rows:
            leaves_data.append(
                {
                    "leave_id": str(row[0]),
                    "user_id": row[1],
                    "name": row[2],
                    "date": row[3],
                    "leave_type": row[4],
                    "reason": row[5],
                }
            )

        conn.close()
        return leaves_data

    def get_settings(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT shift_start, shift_end, grace_period, min_ot FROM settings WHERE id = 1"
        )
        row = cursor.fetchone()
        if row:
            return {"start": row[0], "end": row[1], "grace": row[2], "min_ot": row[3]}
        return {"start": "08:00:00", "end": "17:00:00", "grace": 5, "min_ot": 30}

    def update_settings(self, start, end, grace, min_ot):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE settings
            SET shift_start = ?, shift_end = ?, grace_period = ?, min_ot = ?
            WHERE id = 1
        """,
            (start, end, grace, min_ot),
        )
        conn.commit()


if __name__ == "__main__":
    db = DBManager()
    users = db.load_users()
    print(users)
