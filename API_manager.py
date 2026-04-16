from os import name
from re import A
import re
from urllib import response

import requests


class APIManager:
    def __init__(
        self,
        base_url="http://localhost:5000/api",
    ):
        self.base_url = base_url

    def add_user(self, emp_id, name, email, designation, basic_salary, face_embedding):
        face_embedding = (
            face_embedding.tolist()
            if hasattr(face_embedding, "tolist")
            else face_embedding
        )
        url = f"{self.base_url}/users/register"
        payload = {
            "emp_id": emp_id,
            "name": name,
            "email": email,
            "designation": designation,
            "basic_salary": basic_salary,
            "face_embedding": face_embedding,
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": result.get("message", "User added successfully"),
                    "data": result.get("user"),
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid input data"),
                }
            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_users_for_table(self):
        url = f"{self.base_url}/users"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": (
                        response.message
                        if hasattr(response, "message")
                        else "Data fetched successfully"
                    ),
                    "data": result.get("data", []),
                }
            else:
                return {
                    "success": False,
                    "message": (
                        response.message
                        if hasattr(response, "message")
                        else f"Server returned {response.status_code}"
                    ),
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def mark_attendance(self, emp_id, timestamp):

        url = f"{self.base_url}/attendance/mark"
        payload = {"emp_id": emp_id, "timestamp": timestamp}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "status": result.get("status"),
                    "message": result.get("message", "Success"),
                    "data": result.get("attendance"),
                }

            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "All fields are required"),
                }

            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_daily_attendance(
        self,
    ):
        url = f"{self.base_url}/attendance/today"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Data fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return []

    def update_user(
        self,
        emp_id,
        name,
        email,
        designation,
        basic_salary,
    ):
        url = f"{self.base_url}/users/update/{emp_id}"

        try:
            response = requests.patch(
                url,
                json={
                    "name": name,
                    "email": email,
                    "designation": designation,
                    "basic_salary": basic_salary,
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": result.get("message", "User updated successfully"),
                    "data": result.get("user"),
                }

            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid input data"),
                }

            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_dashboard_summary(self):
        url = f"{self.base_url}/attendance/summary"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Summary fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": {},
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": {},
            }

    def add_leave(self, emp_id, date, leave_type, reason):
        url = f"{self.base_url}/leaves/apply"

        try:
            response = requests.post(
                url,
                json={
                    "emp_id": emp_id,
                    "date": date,
                    "leave_type": leave_type,
                    "reason": reason,
                    "status": "Approved",
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )

            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": result.get("message", "Leave applied successfully"),
                    "data": result.get("leave"),
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid input data"),
                }
            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_all_leaves(self):
        url = f"{self.base_url}/leaves"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Leaves fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def update_leave_status(self, leave_id, status):
        url = f"{self.base_url}/leaves/status/"
        payload = {"leave_id": leave_id, "status": status}

        try:
            response = requests.put(
                url,
                timeout=5,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            result = response.json()

            if response.status_code in [200, 204]:
                return {
                    "success": True,
                    "message": result.get("message", "Leave revoked successfully"),
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid leave ID"),
                }
            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_attendance_by_date_range(self, start_date, end_date):
        url = f"{self.base_url}/attendance/report?start_date={start_date}&end_date={end_date}"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Data fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def get_leaves_by_date_range(self, start_date, end_date):
        url = (
            f"{self.base_url}/leaves/report?start_date={start_date}&end_date={end_date}"
        )
        try:
            response = requests.get(url, timeout=10)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Data fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def get_detailed_attendance_report(self, start_date, end_date):
        url = f"{self.base_url}/attendance/detailed-attendance?emp_id&start_date={start_date}&end_date={end_date}"
        try:
            response = requests.get(url, timeout=10)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Data fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def get_payroll_summary_report(self, month):
        url = f"{self.base_url}/payroll/report?month={month}"
        try:
            response = requests.get(url, timeout=10)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Data fetched successfully",
                    "data": result.get("data", []),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": [],
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": [],
            }

    def generate_bulk_payroll(
        self, month, from_date, to_date, standard_working_days, actual_open_days
    ):
        url = f"{self.base_url}/payroll/generate"

        payload = {
            "month": month,
            "from_date": from_date,
            "to_date": to_date,
            "standard_working_days": standard_working_days,
            "actual_open_days": actual_open_days,
        }

        try:
            response = requests.post(url, json=payload)
            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": result.get("message", "Payroll generated successfully"),
                    "data": result.get("data", []),
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid input data"),
                }
            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }

    def get_settings(self):
        url = f"{self.base_url}/settings"
        try:
            response = requests.get(url, timeout=5)
            result = response.json()

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Settings fetched successfully",
                    "data": result.get("data", {}),
                }

            else:
                return {
                    "success": False,
                    "message": f"Server Error: {response.status_code}",
                    "data": {},
                }

        except requests.exceptions.RequestException as e:
            print(f"Connection Error: {e}")
            return {
                "success": False,
                "message": "Cannot connect to Node.js server. Is it running?",
                "data": {},
            }

    def update_settings(self, payload):
        url = f"{self.base_url}/settings/"

        try:
            response = requests.put(url, json=payload, timeout=5)
            result = response.json()

            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": result.get("message", "Settings updated successfully"),
                }
            elif response.status_code == 400:
                return {
                    "success": False,
                    "error_type": "ClientError",
                    "message": result.get("message", "Invalid input data"),
                }
            else:
                return {
                    "success": False,
                    "error_type": "ServerError",
                    "message": result.get(
                        "message", f"Server returned {response.status_code}"
                    ),
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_type": "ConnectionError",
                "message": "Cannot connect to Node.js server. Is it running?",
            }


if __name__ == "__main__":
    db = APIManager()
    print(db.get_daily_attendance())
