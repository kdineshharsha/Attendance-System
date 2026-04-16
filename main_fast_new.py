from re import A

import pandas as pd
import os
import face_recognition
import multiprocessing as mp
import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QThread, Signal, QDate, QTime, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QFileDialog,
    QTableWidgetItem,
    QCompleter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QBarSet,
    QBarSeries,
    QBarCategoryAxis,
    QValueAxis,
    QPieSeries,
)
from scipy.datasets import face
from db_manager import DBManager
from PySide6.QtGui import QPixmap, QImage, QColor, QFont, QPainter
import cv2
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from API_manager import APIManager
from cache_manager import CacheManager


def ai_scan_worker(input_queue, output_queue, users_data):
    while True:
        frame = input_queue.get()
        if frame is None:
            break

        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_small_frame, model="cnn"
            )
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            if face_encodings:
                current_embedding = face_encodings[0]
                best_match = None
                best_distance = 1.0

                for user in users_data:
                    if len(user["embedding"]) == 0:
                        continue
                    dist = face_recognition.face_distance(
                        [user["embedding"]], current_embedding
                    )

                    if dist < best_distance:
                        best_distance = dist
                        best_match = user

                if best_distance < 0.6 and best_match:
                    output_queue.put(("Match", best_match))
                else:
                    output_queue.put(("No_Match", None))
            else:
                output_queue.put(("No_Match", None))

        except ValueError:
            pass
        except Exception as e:
            output_queue.put(("Error", str(e)))


class RegistrationThread(QThread):
    success_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, reg_img_path, id, name, email, designation, basic_salary):
        super().__init__()
        self.reg_img_path = reg_img_path
        self.emp_id = id
        self.name = name
        self.basic_salary = basic_salary
        self.email = email
        self.designation = designation

        # self.db = DBManager()
        self.api = APIManager()
        self.cache_db = CacheManager()

    def run(self):
        try:
            target_image = face_recognition.load_image_file(self.reg_img_path)
            embedding = face_recognition.face_encodings(target_image, model="cnn")[0]

            face_embedding = embedding
            self.api.add_user(
                self.emp_id,
                self.name,
                self.email,
                self.designation,
                self.basic_salary,
                face_embedding,
            )

            self.success_signal.emit(self.name)
        except ValueError:
            self.error_signal.emit("Face Not Detected")
        except Exception as e:
            self.error_signal.emit(str(e))


class LiveScannerThread(QThread):
    frame_update = Signal(QImage)
    match_found = Signal(dict, QImage)
    no_match = Signal()
    error_signal = Signal(str)

    def __init__(self, users_data):
        super().__init__()

        self.users_data = users_data
        self.running = True
        self.is_processing = False

        self.input_queue = mp.Queue(maxsize=1)
        self.output_queue = mp.Queue()
        self.ai_process = mp.Process(
            target=ai_scan_worker,
            args=(self.input_queue, self.output_queue, self.users_data),
        )
        self.ai_process.daemon = True
        self.ai_process.start()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("Cannot access webcam")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_update.emit(qt_img)

            if self.input_queue.empty():
                self.input_queue.put(frame.copy())

            while not self.output_queue.empty():
                status, match_data = self.output_queue.get()
                if status == "Match":
                    self.match_found.emit(match_data, qt_img)
                elif status == "No_Match":
                    self.no_match.emit()
        cap.release()

    def stop(self):
        self.running = False
        if hasattr(self, "input_queue"):
            try:
                self.input_queue.put(None)
            except:
                pass
        if hasattr(self, "ai_process") and self.ai_process.is_alive():
            self.ai_process.join(timeout=1)

        self.quit()
        self.wait()


class EmailSenderThread(QThread):
    finished_signal = Signal(str, bool)

    def __init__(self, to_email, subject, body):
        super().__init__()
        self.to_email = to_email
        self.subject = subject
        self.body = body

        self.sender_email = os.getenv("ADMIN_EMAIL")
        self.app_password = os.getenv("APP_PASSWORD")

        if not self.sender_email or not self.app_password:
            self.finished_signal.emit(
                "⚠️ WARNING: Email credentials missing in .env file!", False
            )
            return

    def run(self):

        if not self.to_email or self.to_email == "--":
            self.finished_signal.emit("No valid email provided", False)
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = f"FaceRec Admin <{self.sender_email}>"
            msg["To"] = self.to_email
            msg["Subject"] = self.subject
            msg.attach(MIMEText(self.body, "plain"))

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.sender_email, self.app_password)

            server.send_message(msg)
            server.quit()

            print(f"✅ Email sent successfully to: {self.to_email}")
            self.finished_signal.emit(self.to_email, True)

        except Exception as e:
            print(f"❌ Failed to send email to {self.to_email}. Error: {str(e)}")
            self.finished_signal.emit(self.to_email, False)


class PayrollGeneratorThread(QThread):
    success_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, api, payload):
        super().__init__()
        self.api = APIManager()
        self.payload = payload

    def run(self):
        try:

            response = self.api.generate_bulk_payroll(self.payload)

            if response.get("success"):
                self.success_signal.emit(response)
            else:
                self.error_signal.emit(response.get("message", "Unknown Server Error"))
        except Exception as e:
            self.error_signal.emit(str(e))


class AttendanceSystem:
    def __init__(self):
        loader = QUiLoader()
        ui_file = QFile("design.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file.fileName()}: {ui_file.errorString()}")
            sys.exit(-1)
        self.ui = loader.load(ui_file)
        ui_file.close()
        self.reg_img_path = ""
        self.db = DBManager()
        self.api = APIManager()
        self.cache_db = CacheManager()
        self.cache_db.sync_users_to_local_db()

        self.cooldown_dict = {}
        self.cooldown_time = 60
        self.sync_timer = QTimer()

        load_dotenv()
        self.ui.btn_reg_upload.clicked.connect(self.upload_photo)
        self.ui.btn_reg_capture.clicked.connect(self.capture_photo)
        self.ui.btn_reg_save.clicked.connect(self.save_database)
        self.ui.btn_scan_start.clicked.connect(self.start_camera)
        self.ui.btn_scan_stop.clicked.connect(self.stop_camera)
        self.ui.btn_user_edit.clicked.connect(self.handle_edit_user)
        self.ui.btn_leave_save.clicked.connect(self.handle_add_leave)
        self.ui.btn_refresh_att.clicked.connect(self.load_all_attendance)
        self.ui.btn_refresh_leaves.clicked.connect(self.load_all_leaves)
        self.ui.btn_approve_leave.clicked.connect(
            lambda: self.handle_leave_action("Approved")
        )
        self.ui.btn_reject_leave.clicked.connect(
            lambda: self.handle_leave_action("Rejected")
        )
        self.ui.btn_generate_report.clicked.connect(self.handle_generate_report)
        self.ui.btn_export_xl.clicked.connect(self.export_to_excel)
        self.ui.btn_save_settings.clicked.connect(self.save_settings)

        self.ui.btn_nav_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.ui.btn_nav_scan.clicked.connect(lambda: self.switch_page(1))
        self.ui.btn_nav_attendance.clicked.connect(lambda: self.switch_page(2))
        self.ui.btn_nav_users.clicked.connect(lambda: self.switch_page(3))
        self.ui.btn_nav_payroll.clicked.connect(lambda: self.switch_page(4))
        self.ui.btn_nav_reports.clicked.connect(lambda: self.switch_page(5))
        self.ui.btn_nav_settings.clicked.connect(lambda: self.switch_page(6))
        self.current_settings = {}
        self.load_settings_to_ui()

        self.switch_page(0)

        current_date = QDate.currentDate()
        self.ui.dateEdit_from.setDate(
            QDate(current_date.year(), current_date.month(), 1)
        )
        self.ui.dateEdit_to.setDate(current_date)
        self.ui.btn_run_payroll.clicked.connect(self.handle_run_payroll)

        current_date = QDate.currentDate()
        self.ui.dateEdit_pr_month.setDate(current_date)
        self.ui.dateEdit_pr_from.setDate(
            QDate(current_date.year(), current_date.month(), 1)
        )
        self.ui.dateEdit_pr_to.setDate(current_date)
        self.update_dashboard()
        self.load_all_users()
        self.load_all_attendance()
        self.load_all_leaves()

        self.ui.dateEdit_leave.setDate(QDate.currentDate())
        self.load_users_to_leave_dropdown()
        self.last_matched_id = None
        self.no_match_frames = 0
        self.sync_timer.timeout.connect(self.sync_attendance_to_cloud)
        self.sync_timer.start(10000)

    def switch_page(self, index):

        self.ui.stackedWidget.setCurrentIndex(index)

        default_style = "text-align: left; padding: 12px 20px; font-size: 14px; border-radius: 8px; background-color: transparent; color: #a6adc8;"
        active_style = "text-align: left; padding: 12px 20px; font-size: 14px; border-radius: 8px; background-color: #89b4fa; color: #11111b; font-weight: bold;"

        buttons = [
            self.ui.btn_nav_dashboard,
            self.ui.btn_nav_scan,
            self.ui.btn_nav_attendance,
            self.ui.btn_nav_users,
            self.ui.btn_nav_payroll,
            self.ui.btn_nav_reports,
            self.ui.btn_nav_settings,
        ]

        for i, btn in enumerate(buttons):
            if i == index:
                btn.setStyleSheet(active_style)
            else:
                btn.setStyleSheet(default_style)

    def upload_photo(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Photo", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.reg_img_path = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.ui.lbl_reg_preview.width(),
                self.ui.lbl_reg_preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.ui.lbl_reg_preview.setPixmap(scaled_pixmap)

    def capture_photo(self):
        print("Capture Photo button clicked")

    def save_database(self):
        emp_id = self.ui.txt_reg_id.text().strip()
        name = self.ui.txt_reg_name.text().strip()
        designation = self.ui.txt_reg_designation.text().strip()
        basic_salary = self.ui.txt_reg_salary.text().strip()
        email = self.ui.txt_reg_email.text().strip()
        if not emp_id or not name:
            QMessageBox.warning(
                self.ui, "Input Error", "ID and Name are required fields."
            )
            return
        if self.ui.btn_reg_save.text() == "🔄️ Update User":
            self.api.update_user(emp_id, name, email, designation, basic_salary)
            self.on_update_user_success(name)
            self.load_all_users()

            return
        self.ui.btn_reg_save.setEnabled(False)
        self.ui.btn_reg_save.setText("Saving...")

        self.reg_thread = RegistrationThread(
            self.reg_img_path,
            emp_id,
            name,
            email,
            designation,
            basic_salary,
        )
        self.reg_thread.success_signal.connect(self.on_reg_success)
        self.reg_thread.error_signal.connect(self.on_reg_error)
        self.reg_thread.start()

    def update_dashboard(self):
        today_stats = self._calculate_today_stats()
        self._update_summary_cards(today_stats)
        self._check_camera_status()
        self._update_charts(today_stats)

    def create_bar_chart(self, days, present_list, late_list, absent_list):
        set_present = QBarSet("Present")
        set_late = QBarSet("Late")
        set_absent = QBarSet("Absent")

        set_present.append(present_list)
        set_late.append(late_list)
        set_absent.append(absent_list)

        set_present.setColor(QColor("#a6e3a1"))
        set_late.setColor(QColor("#797bff"))
        set_absent.setColor(QColor("#f38ba8"))

        series = QBarSeries()
        series.append(set_present)
        series.append(set_late)
        series.append(set_absent)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Last 5 Days Attendance")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundBrush(QColor("#313244"))
        chart.setTitleBrush(QColor("#cdd6f4"))
        chart.legend().setLabelColor(QColor("#cdd6f4"))

        axis_x = QBarCategoryAxis()
        axis_x.append(days)
        axis_x.setLabelsColor(QColor("#cdd6f4"))
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        max_users = len(self.db.load_users())
        if max_users == 0:
            max_users = 10

        axis_y = QValueAxis()
        axis_y.setRange(0, max_users)
        axis_y.setLabelsColor(QColor("#cdd6f4"))
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background: transparent;")
        return chart_view

    def create_pie_chart(self, present, late, leave, absent):
        series = QPieSeries()

        slice_present = series.append(f"Present ({present})", present)
        slice_late = series.append(f"Late ({late})", late)
        slice_leave = series.append(f"On Leave ({leave})", leave)
        slice_absent = series.append(f"Absent ({absent})", absent)

        slice_present.setBrush(QColor("#a6e3a1"))
        slice_late.setBrush(QColor("#797bff"))
        slice_leave.setBrush(QColor("#89b4fa"))
        slice_absent.setBrush(QColor("#f38ba8"))

        if present > 0:
            slice_present.setExploded(True)
            slice_present.setLabelVisible(True)
            slice_present.setLabelColor(QColor("#cdd6f4"))

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Today's Status Breakdown")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundBrush(QColor("#313244"))
        chart.setTitleBrush(QColor("#cdd6f4"))
        chart.legend().setLabelColor(QColor("#cdd6f4"))
        chart.legend().setAlignment(Qt.AlignBottom)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background: transparent;")
        return chart_view

    def load_all_users(self):

        users_data = self.api.get_users_for_table().get("data", [])
        self.ui.table_users.setColumnCount(5)
        self.ui.table_users.setHorizontalHeaderLabels(
            [
                "ID",
                "Name",
                "Email",
                "Designation",
                "basic_salary",
            ]
        )
        self.ui.table_users.setRowCount(len(users_data))
        self.ui.table_users.setColumnWidth(0, 100)
        self.ui.table_users.setColumnWidth(1, 150)
        self.ui.table_users.setColumnWidth(2, 200)
        self.ui.table_users.setColumnWidth(3, 200)

        for row_idx, user in enumerate(users_data):

            self.ui.table_users.setItem(row_idx, 0, QTableWidgetItem(user["emp_id"]))
            self.ui.table_users.setItem(row_idx, 1, QTableWidgetItem(user["name"]))
            self.ui.table_users.setItem(row_idx, 2, QTableWidgetItem(user["email"]))
            self.ui.table_users.setItem(
                row_idx, 3, QTableWidgetItem(user["designation"])
            )
            self.ui.table_users.setItem(
                row_idx, 4, QTableWidgetItem(str(user["basic_salary"]))
            )

    def handle_edit_user(self):
        row = self.ui.table_users.currentRow()
        if row < 0:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a user to edit."
            )
            return
        user_id = self.ui.table_users.item(row, 0).text()
        user_name = self.ui.table_users.item(row, 1).text()
        user_designation = self.ui.table_users.item(row, 2).text()
        user_email = self.ui.table_users.item(row, 3).text()
        user_basic_salary = self.ui.table_users.item(row, 4).text()

        self.ui.txt_reg_id.setText(user_id)
        self.ui.txt_reg_name.setText(user_name)
        self.ui.txt_reg_designation.setText(user_designation)
        self.ui.txt_reg_email.setText(user_email)
        self.ui.txt_reg_salary.setText(user_basic_salary)

        self.ui.txt_reg_id.setReadOnly(True)
        self.ui.txt_reg_id.setStyleSheet("background-color: #181825; color: #a6adc8;")

        self.ui.btn_reg_save.setText("🔄️ Update User")

        self.ui.tabWidget_users.setCurrentIndex(1)
        self.load_all_users()

    def handle_add_leave(self):

        emp_id = self.ui.combo_leave_user.currentData()

        if not emp_id:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a user to add leave."
            )
            return

        date = self.ui.dateEdit_leave.date().toString("yyyy-MM-dd")
        leave_type = self.ui.combo_leave_type.currentText()
        reason = self.ui.txt_leave_reason.toPlainText().strip()

        success = self.api.add_leave(emp_id, date, leave_type, reason)

        if success:
            QMessageBox.information(self.ui, "Success", "Leave added successfully.")
            self.ui.combo_leave_user.setCurrentIndex(0)
            self.ui.dateEdit_leave.setDate(QDate.currentDate())
            self.ui.combo_leave_type.setCurrentIndex(0)
            self.ui.txt_leave_reason.clear()
            self.load_all_leaves()

        else:
            QMessageBox.warning(
                self.ui,
                f"[⚠️ WARNING] A leave is already recorded for this user on {date}.",
            )

    def handle_leave_action(self, status):
        selected_row = self.ui.table_leaves.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a leave record to delete."
            )
            return
        name_item = self.ui.table_leaves.item(selected_row, 0)

        leave_id = name_item.data(Qt.UserRole)
        emp_name = self.ui.table_leaves.item(selected_row, 1).text()
        current_status = self.ui.table_leaves.item(selected_row, 4).text()

        if current_status == status:
            QMessageBox.information(self.ui, "Info", f"Leave is already {status}.")
            return
        action_text = "approve" if status == "Approved" else "reject"
        reply = QMessageBox.question(
            self.ui,
            "Confirm Action",
            f"Do you want to {action_text} leave for {emp_name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            response = self.api.update_leave_status(leave_id, status)

            if response.get("success"):
                QMessageBox.information(
                    self.ui, "Success", f"Leave {status} successfully!"
                )

                self.load_all_leaves()
                self.update_dashboard()
            else:
                error_msg = response.get(
                    "message", "Something went wrong in the server."
                )
                QMessageBox.critical(
                    self.ui, "Error", f"Failed to update leave:\n{error_msg}"
                )

    def load_all_attendance(self):
        attendance_response = self.api.get_daily_attendance()
        attendance_data = attendance_response.get("data", [])
        self.ui.table_attendance.setRowCount(len(attendance_data))
        self.ui.table_attendance.setColumnWidth(0, 150)
        self.ui.table_attendance.setColumnWidth(1, 150)
        for row_idx, attendance_record in enumerate(attendance_data):

            in_time_str = attendance_record["in_time"]
            out_time_str = attendance_record["out_time"]

            self.ui.table_attendance.setItem(
                row_idx, 0, QTableWidgetItem(attendance_record["user_id"])
            )
            self.ui.table_attendance.setItem(
                row_idx, 1, QTableWidgetItem(attendance_record["name"])
            )

            self.ui.table_attendance.setItem(
                row_idx, 2, QTableWidgetItem(attendance_record["in_time"] or "--")
            )
            self.ui.table_attendance.setItem(
                row_idx, 3, QTableWidgetItem(attendance_record["out_time"] or "--")
            )

            late_val, ot_val = self.calculate_times(in_time_str, out_time_str)
            if late_val != "--":
                self.ui.table_attendance.setItem(
                    row_idx, 4, QTableWidgetItem(f"-{late_val}")
                )
            elif ot_val != "--":
                self.ui.table_attendance.setItem(row_idx, 4, QTableWidgetItem(ot_val))
            else:
                self.ui.table_attendance.setItem(row_idx, 4, QTableWidgetItem("-"))
            early_leave_status = attendance_record["leave_type"]
            status_item = QTableWidgetItem()
            if attendance_record.get("leave_type") == "Early Leave":
                status_item.setText("🟡 Early Left")
            elif late_val != "--":
                status_item.setText("🔴 Late")
            else:
                status_item.setText("🟢 On Time")

            self.ui.table_attendance.setItem(row_idx, 5, status_item)

    def load_all_leaves(self):
        leaves_data = self.api.get_all_leaves().get("data", [])
        self.ui.table_leaves.setRowCount(len(leaves_data))

        self.ui.table_leaves.setColumnWidth(0, 100)  # User Info
        self.ui.table_leaves.setColumnWidth(1, 150)  # Name
        self.ui.table_leaves.setColumnWidth(2, 100)  # Date
        self.ui.table_leaves.setColumnWidth(3, 120)  # Type
        self.ui.table_leaves.setColumnWidth(4, 200)  # Notes

        for row_idx, leave in enumerate(leaves_data):
            name_item = QTableWidgetItem(leave["emp_id"])

            name_item.setData(Qt.UserRole, leave["_id"])
            self.ui.table_leaves.setItem(row_idx, 0, name_item)
            self.ui.table_leaves.setItem(
                row_idx, 1, QTableWidgetItem(leave["user"]["name"])
            )
            self.ui.table_leaves.setItem(
                row_idx, 2, QTableWidgetItem(leave["date"].split("T")[0])
            )
            self.ui.table_leaves.setItem(
                row_idx, 3, QTableWidgetItem(leave["leave_type"])
            )
            self.ui.table_leaves.setItem(
                row_idx, 4, QTableWidgetItem(leave["status"] or "--")
            )
            self.ui.table_leaves.setItem(
                row_idx, 5, QTableWidgetItem(leave["reason"] or "--")
            )

    def calculate_times(self, in_time_str, out_time_str):
        shift_start = datetime.strptime(self.current_settings["start"], "%H:%M:%S")
        shift_end = datetime.strptime(self.current_settings["end"], "%H:%M:%S")

        grace_period = timedelta(minutes=self.current_settings["grace"])
        min_ot = timedelta(minutes=self.current_settings["min_ot"])

        late_time = "--"
        ot_time = "--"

        if in_time_str and in_time_str != "--":
            in_time = datetime.strptime(in_time_str, "%H:%M:%S")
            if in_time > (shift_start + grace_period):
                late_time = str(in_time - shift_start).split(".")[0]
                # print(f"Late by: {late_time}")

        if out_time_str and out_time_str != "--":
            out_time = datetime.strptime(out_time_str, "%H:%M:%S")
            if out_time > (shift_end + min_ot):
                ot_time = str(out_time - shift_end).split(".")[0]
                # print(f"Overtime: {ot_time}")
        # print(f"Calculated times - Late: {late_time}, OT: {ot_time}")
        return late_time, ot_time

    def calculate_times_in_seconds(self, in_time_str, out_time_str):
        shift_start = datetime.strptime(self.current_settings["start"], "%H:%M:%S")
        shift_end = datetime.strptime(self.current_settings["end"], "%H:%M:%S")
        grace_period = timedelta(minutes=self.current_settings["grace"])
        min_ot = timedelta(minutes=self.current_settings["min_ot"])

        late_seconds = 0
        ot_seconds = 0

        if in_time_str and in_time_str != "--":
            in_time = datetime.strptime(in_time_str, "%H:%M:%S")
            if in_time > (shift_start + grace_period):
                late_seconds = (in_time - shift_start).total_seconds()

        if out_time_str and out_time_str != "--":
            out_time = datetime.strptime(out_time_str, "%H:%M:%S")
            if out_time > (shift_end + min_ot):
                ot_seconds = (out_time - shift_end).total_seconds()

        return late_seconds, ot_seconds

    def load_users_to_leave_dropdown(self):
        self.ui.combo_leave_user.clear()

        users = self.api.get_users_for_table().get("data", [])
        search_list = []

        for user in users:
            display_text = f"{user['name']} (ID: {user['emp_id']})"
            self.ui.combo_leave_user.addItem(display_text, user["emp_id"])
            search_list.append(display_text)
        self.ui.combo_leave_user.lineEdit().setPlaceholderText("-- Select User --")

        self.ui.combo_leave_user.setCurrentIndex(-1)

        completer = QCompleter(search_list)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.ui.combo_leave_user.setCompleter(completer)

    def handle_generate_report(self):
        start_date = self.ui.dateEdit_from.date().toString("yyyy-MM-dd")
        end_date = self.ui.dateEdit_to.date().toString("yyyy-MM-dd")
        report_type = self.ui.combo_report_type.currentText()

        print(f"Generating {report_type} report from {start_date} to {end_date}...")

        self.ui.table_report_preview.clear()
        self.ui.btn_export_xl.setEnabled(False)

        if report_type == "Detailed Attendance":
            self.generate_detailed_attendance_report(start_date, end_date)
        elif report_type == "Leave History":
            self.generate_leave_history_report(start_date, end_date)
        elif report_type == "Payroll Summary":
            self.generate_payroll_summary_report(start_date, end_date)
        elif report_type == "Master Summary (All Details)":
            self.generate_master_summary_report(start_date, end_date)

    def generate_detailed_attendance_report(self, start_date, end_date):

        data = self.api.get_detailed_attendance_report(start_date, end_date).get(
            "data", []
        )

        self.ui.table_report_preview.setColumnCount(7)
        self.ui.table_report_preview.setHorizontalHeaderLabels(
            ["Date", "Emp ID", "Name", "IN Time", "OUT Time", "Late / OT", "Status"]
        )
        self.ui.table_report_preview.setRowCount(len(data))

        self.ui.table_report_preview.setColumnWidth(0, 100)  # Date
        self.ui.table_report_preview.setColumnWidth(1, 100)  # ID
        self.ui.table_report_preview.setColumnWidth(2, 180)  # Name
        self.ui.table_report_preview.setColumnWidth(3, 100)  # IN
        self.ui.table_report_preview.setColumnWidth(4, 100)  # OUT
        self.ui.table_report_preview.setColumnWidth(5, 150)  # Late/OT
        self.ui.table_report_preview.setColumnWidth(6, 120)  # Status

        for row_idx, record in enumerate(data):

            ot_val_formated = self._format_time_display(record["ot_minutes"])

            in_time = record["in_time"] or "--"
            out_time = record["out_time"] or "--"
            late_val = record["late_minutes"] or "--"
            ot_val = ot_val_formated or "--"

            late_ot_item = QTableWidgetItem()
            if late_val != "--":
                late_ot_item.setText(f"- {late_val}")
                late_ot_item.setForeground(QColor("#f38ba8"))
            elif ot_val != "--":
                late_ot_item.setText(ot_val)
                late_ot_item.setForeground(QColor("#a6e3a1"))
            else:
                late_ot_item.setText("-")

            status_item = QTableWidgetItem()
            if record["status"] == "On Leave":
                status_item.setText("🟡 Early Left")
            elif late_val != "--":
                status_item.setText("🔴 Late")
            else:
                status_item.setText("🟢 Present")

            self.ui.table_report_preview.setItem(
                row_idx, 0, QTableWidgetItem(record["date"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 1, QTableWidgetItem(record["emp_id"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 2, QTableWidgetItem(record["name"])
            )
            self.ui.table_report_preview.setItem(row_idx, 3, QTableWidgetItem(in_time))
            self.ui.table_report_preview.setItem(row_idx, 4, QTableWidgetItem(out_time))
            self.ui.table_report_preview.setItem(row_idx, 5, late_ot_item)
            self.ui.table_report_preview.setItem(row_idx, 6, status_item)

        if len(data) > 0:
            self.ui.btn_export_xl.setEnabled(True)
        else:
            QMessageBox.information(
                self.ui,
                "No Data",
                f"No attendance records found between {start_date} and {end_date}.",
            )

    def generate_leave_history_report(self, start_date, end_date):
        data = self.api.get_leaves_by_date_range(start_date, end_date).get("data", [])
        self.ui.table_report_preview.setColumnCount(6)
        self.ui.table_report_preview.setHorizontalHeaderLabels(
            ["Date", "Emp ID", "Name", "Leave Type", "Reason / Notes", "Status"]
        )
        self.ui.table_report_preview.setRowCount(len(data))

        self.ui.table_report_preview.setColumnWidth(0, 120)  # Date
        self.ui.table_report_preview.setColumnWidth(1, 120)  # ID
        self.ui.table_report_preview.setColumnWidth(2, 150)  # Name
        self.ui.table_report_preview.setColumnWidth(3, 150)  # Type
        self.ui.table_report_preview.setColumnWidth(4, 250)  # Reason

        for row_idx, record in enumerate(data):
            print(record)
            self.ui.table_report_preview.setItem(
                row_idx, 0, QTableWidgetItem(record["date"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 1, QTableWidgetItem(record["emp_id"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 2, QTableWidgetItem(record["name"])
            )

            type_item = QTableWidgetItem(record["leave_type"])
            if record["leave_type"] == "Medical":
                type_item.setForeground(QColor("#f38ba8"))
            elif record["leave_type"] == "Early Leave":
                type_item.setForeground(QColor("#f9e2af"))
            else:
                type_item.setForeground(QColor("#89b4fa"))

            self.ui.table_report_preview.setItem(row_idx, 3, type_item)
            self.ui.table_report_preview.setItem(
                row_idx, 4, QTableWidgetItem(record["reason"] or "--")
            )

            self.ui.table_report_preview.setItem(
                row_idx, 5, QTableWidgetItem(record["status"] or "--")
            )

        if len(data) > 0:
            self.ui.btn_export_xl.setEnabled(True)
        else:
            QMessageBox.information(
                self.ui,
                "No Data",
                f"No leave records found between {start_date} and {end_date}.",
            )

    def generate_payroll_summary_report(self, start_date, end_date):

        month_str = start_date[:7]
        print(f"Fetching payroll for month: {month_str}")

        payroll_records = self.api.get_payroll_summary_report(month_str).get("data", [])

        self.ui.table_report_preview.setColumnCount(16)
        self.ui.table_report_preview.setHorizontalHeaderLabels(
            [
                "Emp ID",
                "Name",
                "Basic (Rs)",
                "Worked/Open",
                "Absent",
                "Late (Mins)",
                "OT (Hrs)",
                "OT Pay",
                "Bonus",
                "Late Penalty",
                "No-Pay",
                "EPF (8%)",
                "Gross Pay",
                "Total Ded.",
                "Net Pay",
                "Status",
            ]
        )

        self.ui.table_report_preview.setRowCount(len(payroll_records))

        self.ui.table_report_preview.setColumnWidth(0, 80)  # ID
        self.ui.table_report_preview.setColumnWidth(1, 150)  # Name
        self.ui.table_report_preview.setColumnWidth(2, 100)  # Basic
        self.ui.table_report_preview.setColumnWidth(3, 100)  # Worked/Open
        self.ui.table_report_preview.setColumnWidth(14, 120)  # Net Pay

        def make_item(val, color_hex=None, bold=False):
            item = QTableWidgetItem(str(val))
            if color_hex:
                item.setForeground(QColor(color_hex))
            if bold:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            return item

        for row_idx, record in enumerate(payroll_records):

            emp_id = record.get("emp_id", "--")
            user_info = record.get("user", {})
            name = (
                user_info.get("name", "Unknown")
                if isinstance(user_info, dict)
                else "Unknown"
            )
            basic_salary = record.get("basic_salary_snapshot", 0)

            att_summary = record.get("attendance_summary", {})
            open_days = att_summary.get("actual_open_days", 0)
            present_days = att_summary.get("present_days", 0)
            absent_days = att_summary.get("absent_days", 0)
            late_mins = att_summary.get("late_minutes", 0)
            ot_mins = att_summary.get("ot_minutes", 0)
            ot_hrs = round(ot_mins / 60, 2)

            earnings_list = record.get("earnings", [])
            ot_pay = next(
                (
                    item["amount"]
                    for item in earnings_list
                    if item["name"] == "Overtime Pay"
                ),
                0,
            )
            bonus = next(
                (
                    item["amount"]
                    for item in earnings_list
                    if item["name"] == "Attendance Bonus"
                ),
                0,
            )

            deductions_list = record.get("deductions", [])
            epf = next(
                (
                    item["amount"]
                    for item in deductions_list
                    if item["name"] == "EPF (8%)"
                ),
                0,
            )
            no_pay = next(
                (
                    item["amount"]
                    for item in deductions_list
                    if item["name"] == "No-Pay"
                ),
                0,
            )
            late_penalty = next(
                (
                    item["amount"]
                    for item in deductions_list
                    if item["name"] == "Late Penalty"
                ),
                0,
            )

            gross_pay = record.get("gross_pay", 0)
            total_ded = record.get("total_deductions", 0)
            net_pay = record.get("net_pay", 0)
            status = record.get("status", "Pending")

            self.ui.table_report_preview.setItem(row_idx, 0, make_item(emp_id))
            self.ui.table_report_preview.setItem(row_idx, 1, make_item(name))
            self.ui.table_report_preview.setItem(
                row_idx, 2, make_item(f"{basic_salary:,.2f}")
            )

            self.ui.table_report_preview.setItem(
                row_idx, 3, make_item(f"{present_days}/{open_days}")
            )
            self.ui.table_report_preview.setItem(
                row_idx,
                4,
                make_item(absent_days, "#f38ba8" if absent_days > 0 else None),
            )
            self.ui.table_report_preview.setItem(
                row_idx, 5, make_item(late_mins, "#f38ba8" if late_mins > 0 else None)
            )
            self.ui.table_report_preview.setItem(
                row_idx, 6, make_item(ot_hrs, "#a6e3a1" if ot_hrs > 0 else None)
            )

            self.ui.table_report_preview.setItem(
                row_idx,
                7,
                make_item(f"{ot_pay:,.2f}", "#a6e3a1" if ot_pay > 0 else None),
            )
            self.ui.table_report_preview.setItem(
                row_idx, 8, make_item(f"{bonus:,.2f}", "#a6e3a1" if bonus > 0 else None)
            )

            self.ui.table_report_preview.setItem(
                row_idx,
                9,
                make_item(
                    f"{late_penalty:,.2f}", "#f38ba8" if late_penalty > 0 else None
                ),
            )
            self.ui.table_report_preview.setItem(
                row_idx,
                10,
                make_item(f"{no_pay:,.2f}", "#f38ba8" if no_pay > 0 else None),
            )
            self.ui.table_report_preview.setItem(
                row_idx, 11, make_item(f"{epf:,.2f}", "#f38ba8" if epf > 0 else None)
            )

            self.ui.table_report_preview.setItem(
                row_idx, 12, make_item(f"{gross_pay:,.2f}")
            )
            self.ui.table_report_preview.setItem(
                row_idx,
                13,
                make_item(f"{total_ded:,.2f}", "#f38ba8" if total_ded > 0 else None),
            )
            self.ui.table_report_preview.setItem(
                row_idx, 14, make_item(f"{net_pay:,.2f}", "#89b4fa", bold=True)
            )

            status_color = "#a6e3a1" if status == "Paid" else "#f9e2af"
            self.ui.table_report_preview.setItem(
                row_idx, 15, make_item(status, status_color, bold=True)
            )

        if len(payroll_records) > 0:
            self.ui.btn_export_xl.setEnabled(True)
        else:
            QMessageBox.information(
                self.ui,
                "No Data",
                f"No payroll data found for the month of {month_str}.",
            )

    def generate_master_summary_report(self, start_date, end_date):

        all_users = self.db.get_users_for_table()
        attendance_data = self.db.get_attendance_by_date_range(start_date, end_date)
        leaves_data = self.db.get_leaves_by_date_range(start_date, end_date)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end_dt - start_dt

        master_records = []

        for i in range(delta.days + 1):
            current_date = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")

            for user in all_users:
                user_id = user["id"]
                user_name = user["name"]

                att_record = next(
                    (
                        item
                        for item in attendance_data
                        if item["user_id"] == user_id and item["date"] == current_date
                    ),
                    None,
                )

                leave_record = next(
                    (
                        item
                        for item in leaves_data
                        if item["user_id"] == user_id and item["date"] == current_date
                    ),
                    None,
                )

                in_time = "--"
                out_time = "--"
                status = ""
                remarks = "--"

                if att_record:
                    in_time = att_record["in_time"] or "--"
                    out_time = att_record["out_time"] or "--"
                    late_val, ot_val = self.calculate_times(in_time, out_time)

                    if att_record.get("leave_type") == "Early Leave":
                        status = "🟡 Early Left"
                    elif late_val != "--":
                        status = "🔴 Late"
                    else:
                        status = "🟢 Present"

                    if late_val != "--" and ot_val != "--":
                        remarks = f"{late_val} | {ot_val}"
                    elif late_val != "--":
                        remarks = late_val
                    elif ot_val != "--":
                        remarks = ot_val

                elif leave_record:
                    status = "🎌 On Leave"
                    remarks = f"Approved: {leave_record['leave_type']}"
                else:
                    status = "❌ Absent"

                master_records.append(
                    {
                        "date": current_date,
                        "id": user_id,
                        "name": user_name,
                        "status": status,
                        "in_time": in_time,
                        "out_time": out_time,
                        "remarks": remarks,
                    }
                )

        self.ui.table_report_preview.setColumnCount(7)
        self.ui.table_report_preview.setHorizontalHeaderLabels(
            ["Date", "Emp ID", "Name", "Status", "IN Time", "OUT Time", "Remarks"]
        )
        self.ui.table_report_preview.setRowCount(len(master_records))

        self.ui.table_report_preview.setColumnWidth(0, 100)  # Date
        self.ui.table_report_preview.setColumnWidth(1, 200)  # ID
        self.ui.table_report_preview.setColumnWidth(2, 200)  # Name
        self.ui.table_report_preview.setColumnWidth(3, 120)  # Status
        self.ui.table_report_preview.setColumnWidth(4, 100)  # IN
        self.ui.table_report_preview.setColumnWidth(5, 100)  # OUT
        self.ui.table_report_preview.setColumnWidth(6, 200)  # Remarks

        for row_idx, record in enumerate(master_records):
            self.ui.table_report_preview.setItem(
                row_idx, 0, QTableWidgetItem(record["date"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 1, QTableWidgetItem(record["id"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 2, QTableWidgetItem(record["name"])
            )

            status_item = QTableWidgetItem(record["status"])
            if "Absent" in record["status"]:
                status_item.setForeground(QColor("#f38ba8"))  # Red
            elif "Leave" in record["status"]:
                status_item.setForeground(QColor("#f9e2af"))  # Yellow
            elif "Present" in record["status"]:
                status_item.setForeground(QColor("#a6e3a1"))  # Green
            self.ui.table_report_preview.setItem(row_idx, 3, status_item)

            self.ui.table_report_preview.setItem(
                row_idx, 4, QTableWidgetItem(record["in_time"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 5, QTableWidgetItem(record["out_time"])
            )
            self.ui.table_report_preview.setItem(
                row_idx, 6, QTableWidgetItem(record["remarks"])
            )

        if len(master_records) > 0:
            self.ui.btn_export_xl.setEnabled(True)
        else:
            QMessageBox.information(
                self.ui, "No Data", "No data available to generate master report."
            )

    def export_to_excel(self):

        file_path, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Save Report As",
            "Attendance_Report.xlsx",  # Default නම
            "Excel Files (*.xlsx);;All Files (*)",
        )

        if not file_path:
            return

        self.ui.btn_export_xl.setText("⏳ Exporting...")
        self.ui.btn_export_xl.setEnabled(False)

        try:
            column_count = self.ui.table_report_preview.columnCount()
            headers = []
            for col in range(column_count):
                header_item = self.ui.table_report_preview.horizontalHeaderItem(col)
                headers.append(header_item.text() if header_item else f"Column {col}")

            row_count = self.ui.table_report_preview.rowCount()
            table_data = []

            for row in range(row_count):
                row_data = []
                for col in range(column_count):
                    item = self.ui.table_report_preview.item(row, col)
                    row_data.append(item.text() if item else "")
                table_data.append(row_data)

            df = pd.DataFrame(table_data, columns=headers)

            df.to_excel(file_path, index=False, engine="openpyxl")

            QMessageBox.information(
                self.ui,
                "Export Successful",
                f"Report successfully saved to:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self.ui, "Export Error", f"An error occurred while exporting:\n{str(e)}"
            )

        finally:
            self.ui.btn_export_xl.setText("📥 Export to Excel")
            self.ui.btn_export_xl.setEnabled(True)

    def load_settings_to_ui(self):
        try:
            response = self.api.get_settings()
            settings = response.get("data", {}) if response else {}

            start_str = settings.get("shift_start_time", "08:00:00")
            end_str = settings.get("shift_end_time", "17:00:00")
            grace = settings.get("grace_period_mins", 15)
            min_ot = settings.get("min_ot_mins", 30)
            std_days = settings.get("standard_working_days", 22)

            start_format = "HH:mm:ss" if len(start_str) > 5 else "HH:mm"
            end_format = "HH:mm:ss" if len(end_str) > 5 else "HH:mm"

            self.ui.timeEdit_start.setTime(QTime.fromString(start_str, start_format))
            self.ui.timeEdit_end.setTime(QTime.fromString(end_str, end_format))
            self.ui.spin_grace.setValue(grace)
            self.ui.spin_ot.setValue(min_ot)
            self.ui.spin_std_days.setValue(std_days)

            self.current_settings = {
                "start": start_str if len(start_str) > 5 else f"{start_str}:00",
                "end": end_str if len(end_str) > 5 else f"{end_str}:00",
                "grace": grace,
                "min_ot": min_ot,
                "std_days": std_days,
            }
        except Exception as e:
            print(f"Error loading settings from API: {e}")
            QMessageBox.warning(
                self.ui, "Settings Error", "Could not load settings from the server."
            )

    def save_settings(self):

        start = self.ui.timeEdit_start.time().toString("HH:mm:ss")
        end = self.ui.timeEdit_end.time().toString("HH:mm:ss")
        grace = self.ui.spin_grace.value()
        min_ot = self.ui.spin_ot.value()
        std_days = self.ui.spin_std_days.value()

        payload = {
            "shift_start_time": start,
            "shift_end_time": end,
            "grace_period_mins": grace,
            "min_ot_mins": min_ot,
            "standard_working_days": std_days,
        }

        try:
            response = self.api.update_settings(payload)

            if response and response.get("success"):
                self.current_settings = {
                    "start": start,
                    "end": end,
                    "grace": grace,
                    "min_ot": min_ot,
                    "std_days": std_days,
                }
                QMessageBox.information(
                    self.ui, "Success", "System Settings Updated Successfully! 🚀"
                )
            else:
                error_msg = (
                    response.get("message", "Unknown Server Error")
                    if response
                    else "No response"
                )
                QMessageBox.critical(
                    self.ui, "Error", f"Failed to save settings:\n{error_msg}"
                )

        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"An error occurred:\n{str(e)}")

        finally:
            self.ui.btn_save_settings.setEnabled(True)
            self.ui.btn_save_settings.setText("💾 Save Configurations")
        self.update_dashboard()

    def start_camera(self):
        users_data = self.cache_db.load_users()
        # print(users_data)
        # users_data = self.api.get_users_for_table()
        if not users_data:
            QMessageBox.warning(self.ui, "No Data", "Database is empty.")
            return
        self.ui.btn_scan_start.setEnabled(False)
        self.ui.lbl_status.setText("Status: Starting Camera...")
        self.scanner_thread = LiveScannerThread(users_data)
        self.scanner_thread.frame_update.connect(self.update_live_frame)
        self.scanner_thread.match_found.connect(self.show_match_results)
        self.scanner_thread.no_match.connect(self.show_no_match)
        self.scanner_thread.error_signal.connect(self.show_cam_error)
        self.scanner_thread.start()

    def stop_camera(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
        self.ui.lbl_live_camera.clear()
        self.ui.lbl_live_camera.setText("Camera Feed Offline")
        self.ui.btn_scan_start.setEnabled(True)
        self.ui.lbl_status.setText("Status: Camera Stopped.")

    def update_live_frame(self, qt_img):

        if not self.scanner_thread or not self.scanner_thread.running:
            return

        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self.ui.lbl_live_camera.width(),
            self.ui.lbl_live_camera.height(),
            Qt.KeepAspectRatio,
        )
        self.ui.lbl_live_camera.setPixmap(scaled)

    def show_match_results(self, user_data, qt_img):
        current_id = user_data["emp_id"]
        current_time = datetime.now()

        if hasattr(self, "cooldown_dict") and current_id in self.cooldown_dict:
            last_scan_time = self.cooldown_dict[current_id]
            time_passed = (current_time - last_scan_time).total_seconds()
            if time_passed < self.cooldown_time:
                print(
                    f"Cooldown active for {user_data['name']} (ID: {current_id}). Time passed: {time_passed:.2f} seconds."
                )
                return
        self.cooldown_dict[current_id] = current_time

        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        attendance_status = self.cache_db.mark_attendance(
            user_data["emp_id"], timestamp
        )
        print(
            f"Attendance status for {user_data['name']} (ID: {user_data['emp_id']}): {attendance_status}"
        )
        if attendance_status == "IN":
            msg = f"✅ IN: Welcome {user_data['name']}!"
            self.ui.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold;")
            self.ui.lbl_status.setText(msg)

            user_email = user_data.get("email", "")

            if user_email and user_email != "--":
                subject = "Attendance Logged Successfully! ✅"
                body = f"Hi {user_data['name']},\n\nYour attendance for today has been successfully recorded in the FaceRec System.\n\nHave a great day at work!\n\n- Admin"

                self.email_thread = EmailSenderThread(user_email, subject, body)
                self.email_thread.start()

        elif attendance_status == "OUT":
            msg = f"🛑 OUT: Goodbye {user_data['name']}!"
            self.ui.lbl_status.setStyleSheet("color: #FFA500; font-weight: bold;")
            self.ui.lbl_status.setText(msg)
        elif attendance_status == "OUT (EARLY)":
            msg = f"🛑 OUT (EARLY): Goodbye {user_data['name']}!"
            self.ui.lbl_status.setStyleSheet("color: #FFA500; font-weight: bold;")
            self.ui.lbl_status.setText(msg)

        elif attendance_status == "ALREADY_IN":
            msg = f"⚠️ ALREADY IN: {user_data['name']}"
            self.ui.lbl_status.setStyleSheet("color: #FFFF00; font-weight: bold;")
            self.ui.lbl_status.setText(msg)

        elif attendance_status == "COMPLETED":
            msg = f"🔒 COMPLETED: Already Left."
            self.ui.lbl_status.setStyleSheet("color: #808080; font-weight: bold;")
            self.ui.lbl_status.setText(msg)

        else:
            msg = f"✅ Identified {user_data['name']}"
            self.ui.lbl_status.setStyleSheet("color: white;")
            self.ui.lbl_status.setText(msg)
        print(f"Match found: {user_data['name']} (ID: {user_data['emp_id']})")
        # self.ui.lbl_status.setText(f"Status: ✅ Identified {user_data['name']}")
        self.ui.lbl_res_name.setText(f"Name: {user_data['name']}")
        self.ui.lbl_res_id.setText(f"Employee ID: {user_data['emp_id']}")
        self.ui.lbl_res_age.setText(f"Designation: {user_data['designation']}")
        # self.ui.txt_res_details.setText(user_data["designation"] or "--")
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self.ui.lbl_res_image.width(),
            self.ui.lbl_res_image.height(),
            Qt.KeepAspectRatio,
        )
        self.ui.lbl_res_image.setPixmap(scaled)
        self.load_all_attendance()

    def show_no_match(self):

        self.last_matched_id = None
        self.ui.lbl_status.setText("Status: ❌ Unknown Face")
        self.ui.lbl_res_name.setText("Name: Unknown")
        self.ui.lbl_res_id.setText("Student ID: --")
        self.ui.lbl_res_age.setText("Age: --")
        self.ui.txt_res_details.setText("--")
        self.ui.lbl_res_image.clear()

    def show_cam_error(self, err_msg):
        QMessageBox.critical(self.ui, "Camera Error", err_msg)
        self.stop_camera()

    def on_reg_success(self, name):
        QMessageBox.information(
            self.ui, "Success", f"{name} has been added to the database."
        )
        self.ui.txt_reg_id.clear()
        self.ui.txt_reg_name.clear()
        self.ui.txt_reg_designation.clear()
        self.ui.txt_reg_salary.clear()
        self.ui.txt_reg_email.clear()
        self.ui.lbl_reg_preview.clear()
        self.ui.btn_reg_save.setEnabled(True)
        self.ui.btn_reg_save.setText("💾 Save to Database")
        self.reg_img_path = ""
        self.load_all_users()

    def on_update_user_success(self, name):
        QMessageBox.information(
            self.ui, "Success", f"{name}'s information has been updated."
        )
        self.ui.txt_reg_id.clear()
        self.ui.txt_reg_name.clear()
        self.ui.txt_reg_designation.clear()
        self.ui.txt_reg_salary.clear()
        self.ui.txt_reg_email.clear()
        self.ui.btn_reg_save.setEnabled(True)
        self.ui.btn_reg_save.setText("💾 Save to Database")
        self.reg_img_path = ""
        self.load_all_users()

    def on_reg_error(self, error_msg):
        if error_msg == "Face Not Detected":
            QMessageBox.warning(
                self.ui,
                "Face Not Detected",
                "No face detected in the image. Please try again with a clear photo.",
            )
        else:
            QMessageBox.critical(self.ui, "Error", f"An error occurred: {error_msg}")
            print(f"Registration Error: {error_msg}")
        self.ui.btn_reg_save.setEnabled(True)
        self.ui.btn_reg_save.setText("💾 Save to Database")

    def _check_camera_status(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if cap is None or not cap.isOpened():
                return False

            cap.release()

            self.ui.lbl_stat_cam.setText("📷 Camera Module: Ready")
            self.ui.lbl_stat_cam.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.ui.btn_scan_start.setEnabled(True)
            return True
        except:
            self.ui.lbl_stat_cam.setText("⚠️ Camera Module: Not Found / Error")
            self.ui.lbl_stat_cam.setStyleSheet("color: #f38ba8; font-weight: bold;")
            self.ui.btn_scan_start.setEnabled(False)
            return False

    def _calculate_today_stats(self):
        today_stats = self.api.get_dashboard_summary().get("data", {})

        return {
            "total": today_stats.get("totalEmployees", 0),
            "present": today_stats.get("present", 0),
            "late": today_stats.get("late", 0),
            "leave": today_stats.get("onLeave", 0),
            "absent": today_stats.get("absent", 0),
        }

    def _update_summary_cards(self, stats):
        self.ui.val_total.setText(str(stats["total"]))
        self.ui.val_present.setText(str(stats["present"]))
        self.ui.val_late.setText(str(stats["late"]))
        self.ui.val_leave.setText(str(stats["leave"]))

    def _update_charts(self, stats):
        days_labels, present_counts, late_counts, absent_counts = [], [], [], []
        total_users = stats.get("total", 0)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=4)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        att_response = self.api.get_attendance_by_date_range(start_str, end_str)

        leaves_response = self.api.get_leaves_by_date_range(start_str, end_str)

        all_att = att_response.get("data", []) if isinstance(att_response, dict) else []
        all_leaves = (
            leaves_response.get("data", []) if isinstance(leaves_response, dict) else []
        )

        for i in range(4, -1, -1):
            target_date = end_date - timedelta(days=i)
            date_str = target_date.strftime("%Y-%m-%d")
            days_labels.append(target_date.strftime("%a"))

            daily_att = [
                r for r in all_att if r.get("date", "").split("T")[0] == date_str
            ]
            daily_leaves = [
                r
                for r in all_leaves
                if r.get("date", "").split("T")[0] == date_str
                and r.get("status") == "Approved"
            ]

            daily_present = len(daily_att)
            daily_leave = len(daily_leaves)
            daily_absent = max(0, total_users - (daily_present + daily_leave))

            daily_late = 0
            for rec in daily_att:
                in_time = rec.get("in_time", "--")
                out_time = rec.get("out_time", "--")

                late_sec, _ = self.calculate_times_in_seconds(in_time, out_time)

                if late_sec > (self.current_settings["grace"] * 60):
                    daily_late += 1

            present_counts.append(daily_present)
            late_counts.append(daily_late)
            absent_counts.append(daily_absent)

        bar_chart_view = self.create_bar_chart(
            days_labels, present_counts, late_counts, absent_counts
        )
        if self.ui.chart_container_bar.layout():
            QWidget().setLayout(self.ui.chart_container_bar.layout())
        bar_layout = QVBoxLayout(self.ui.chart_container_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(bar_chart_view)

        pie_chart_view = self.create_pie_chart(
            stats.get("present", 0),
            stats.get("late", 0),
            stats.get("leave", 0),
            stats.get("absent", 0),
        )
        if self.ui.chart_container_pie.layout():
            QWidget().setLayout(self.ui.chart_container_pie.layout())
        pie_layout = QVBoxLayout(self.ui.chart_container_pie)
        pie_layout.setContentsMargins(0, 0, 0, 0)
        pie_layout.addWidget(pie_chart_view)

    def sync_attendance_to_cloud(self):
        print("👀 QTimer is checking for data...")
        pending_records = self.cache_db.get_pending_attendance()
        if not pending_records:
            return

        for record in pending_records:
            emp_id = record["emp_id"]
            timestamp = record["timestamp"]
            record_id = record["id"]

            response = self.api.mark_attendance(emp_id, timestamp)

            if response["success"]:
                self.cache_db.delete_pending_attendance(record_id)
                print(f"Synced attendance for Employee ID: {emp_id} at {timestamp}")

            else:
                print(f"⚠️ Network Offline or Cloud Error. Pausing sync for {emp_id}.")
                break

    def _format_time_display(self, minutes):
        try:
            mins = int(minutes)
        except (ValueError, TypeError):
            return "-"

        if mins <= 0:
            return "-"

        hours = mins // 60
        remaining_mins = mins % 60

        if hours > 0 and remaining_mins > 0:
            return f"{hours}h {remaining_mins}m"
        elif hours > 0:
            return f"{hours}h"
        else:
            return f"{remaining_mins}m"

    def handle_run_payroll(self):
        month = self.ui.dateEdit_pr_month.date().toString("yyyy-MM")
        from_date = self.ui.dateEdit_pr_from.date().toString("yyyy-MM-dd")
        to_date = self.ui.dateEdit_pr_to.date().toString("yyyy-MM-dd")
        actual_open_days = self.ui.spin_pr_actual.value()

        std_days = self.ui.spin_std_days.value()

        if actual_open_days > std_days:
            QMessageBox.warning(
                self.ui,
                "Validation Error",
                "Actual Open Days cannot be greater than Standard Working Days.",
            )
            return

        reply = QMessageBox.question(
            self.ui,
            "Confirm Payroll Generation",
            f"Are you sure you want to generate payroll for {month}?\n\nFrom: {from_date}\nTo: {to_date}\nOpen Days: {actual_open_days}",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.No:
            return

        self.ui.btn_run_payroll.setEnabled(False)
        self.ui.btn_run_payroll.setText("⚙️ Processing Payroll... Please Wait")

        payload = {
            "month": month,
            "from_date": from_date,
            "to_date": to_date,
            "standard_working_days": std_days,
            "actual_open_days": actual_open_days,
        }

        self.pr_thread = PayrollGeneratorThread(self.api, payload)
        self.pr_thread.success_signal.connect(self.on_payroll_success)
        self.pr_thread.error_signal.connect(self.on_payroll_error)
        self.pr_thread.start()

    def on_payroll_success(self, response_data):
        count = len(response_data.get("processed_users", []))
        QMessageBox.information(
            self.ui,
            "Success! 🎉",
            f"{response_data.get('message', 'Success')}\n\nProcessed Employees: {count}\n\nYou can now view the results in the 'Reports' section.",
        )
        self.reset_pr_button()

    def on_payroll_error(self, error_msg):
        QMessageBox.critical(
            self.ui,
            "Generation Failed ❌",
            f"An error occurred while generating payroll:\n\n{error_msg}",
        )
        self.reset_pr_button()

    def reset_pr_button(self):
        self.ui.btn_run_payroll.setEnabled(True)
        self.ui.btn_run_payroll.setText("🚀 Run Payroll Engine")


if __name__ == "__main__":
    mp.freeze_support()
    global_font = QFont("Segoe UI", 10)
    app = QApplication(sys.argv)
    app.setFont(global_font)
    attendance_system = AttendanceSystem()
    attendance_system.ui.show()
    sys.exit(app.exec())
