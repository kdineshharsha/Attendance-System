from calendar import c
import dis
from math import e
from operator import le
import os
import re
import face_recognition
import multiprocessing as mp
import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QThread, Signal, QDate
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QFileDialog,
    QTableWidgetItem,
    QCompleter,
)

from scipy.datasets import face
from sympy import Q, use
from db_manager import DBManager
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from datetime import datetime, timedelta

# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
# os.environ["TF_NUM_INTEROP_THREADS"] = "8"


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

        except ValueError:
            pass
        except Exception as e:
            output_queue.put(("Error", str(e)))


class RegistrationThread(QThread):
    success_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, reg_img_path, id, name, age, details):
        super().__init__()
        self.reg_img_path = reg_img_path
        self.id = id
        self.name = name
        self.age = age
        self.details = details
        self.db = DBManager()

    def run(self):
        try:
            target_image = face_recognition.load_image_file(self.reg_img_path)
            embedding = face_recognition.face_encodings(target_image, model="cnn")[0]

            face_embedding = embedding
            self.db.add_user(self.id, self.name, self.age, self.details, face_embedding)
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
        self.ui.btn_reg_upload.clicked.connect(self.upload_photo)
        self.ui.btn_reg_capture.clicked.connect(self.capture_photo)
        self.ui.btn_reg_save.clicked.connect(self.save_database)
        self.ui.btn_scan_start.clicked.connect(self.start_camera)
        self.ui.btn_scan_stop.clicked.connect(self.stop_camera)
        self.ui.btn_user_edit.clicked.connect(self.handle_edit_user)
        self.ui.btn_leave_save.clicked.connect(self.handle_add_leave)
        self.ui.btn_refresh_att.clicked.connect(self.load_all_attendance)
        self.ui.btn_leave_delete.clicked.connect(self.handle_delete_leave)
        self.load_all_users()
        self.load_all_attendance()
        self.load_all_leaves()
        self.ui.dateEdit_leave.setDate(QDate.currentDate())
        self.load_users_to_leave_dropdown()

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
        id = self.ui.txt_reg_id.text().strip()
        name = self.ui.txt_reg_name.text().strip()
        age = self.ui.spin_reg_age.value()
        details = self.ui.txt_reg_details.toPlainText().strip()
        if not id or not name:
            QMessageBox.warning(
                self.ui, "Input Error", "ID and Name are required fields."
            )
            return
        self.ui.btn_reg_save.setEnabled(False)
        self.ui.btn_reg_save.setText("Saving...")
        self.reg_thread = RegistrationThread(self.reg_img_path, id, name, age, details)
        self.reg_thread.success_signal.connect(self.on_reg_success)
        self.reg_thread.error_signal.connect(self.on_reg_error)
        self.reg_thread.start()

    def load_all_users(self):
        users_data = self.db.get_users_for_table()
        print(users_data)

        self.ui.table_users.setRowCount(len(users_data))
        self.ui.table_users.setColumnWidth(0, 150)

        for row_idx, user in enumerate(users_data):

            self.ui.table_users.setItem(row_idx, 0, QTableWidgetItem(user["id"]))
            self.ui.table_users.setItem(row_idx, 1, QTableWidgetItem(user["name"]))
            self.ui.table_users.setItem(row_idx, 2, QTableWidgetItem(str(user["age"])))
            self.ui.table_users.setItem(row_idx, 3, QTableWidgetItem(user["details"]))
            leave_status = user["early_leave"]

            if leave_status == True:
                self.ui.table_users.setItem(row_idx, 4, QTableWidgetItem("✅ Granted"))
            else:
                self.ui.table_users.setItem(row_idx, 4, QTableWidgetItem("❌ None"))

        print(f"Loaded {len(users_data)} users from database.")

    def handle_edit_user(self):
        row = self.ui.table_users.currentRow()
        if row < 0:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a user to edit."
            )
            return
        user_id = self.ui.table_users.item(row, 0).text()
        user_name = self.ui.table_users.item(row, 1).text()
        user_age = self.ui.table_users.item(row, 2).text()
        user_details = self.ui.table_users.item(row, 3).text()
        user_details = self.ui.table_users.item(row, 3).text()

        self.ui.txt_reg_id.setText(user_id)
        self.ui.txt_reg_name.setText(user_name)
        self.ui.spin_reg_age.setValue(int(user_age))
        self.ui.txt_reg_details.setText(user_details)

        self.ui.txt_reg_id.setReadOnly(True)
        self.ui.txt_reg_id.setStyleSheet("background-color: #181825; color: #a6adc8;")

        self.ui.btn_reg_save.setText("🔄️ Update User")

        self.ui.tabWidget.setCurrentIndex(0)
        self.db.grant_early_leave(user_id)
        self.load_all_users()

    def handle_add_leave(self):

        user_id = self.ui.combo_leave_user.currentData()

        if not user_id:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a user to add leave."
            )
            return

        date = self.ui.dateEdit_leave.date().toString("yyyy-MM-dd")
        leave_type = self.ui.combo_leave_type.currentText()
        reason = self.ui.txt_leave_reason.toPlainText().strip()

        success = self.db.add_leave(user_id, date, leave_type, reason)

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

    def handle_delete_leave(self):
        selected_row = self.ui.table_leaves.currentRow()
        if selected_row < 0:
            QMessageBox.warning(
                self.ui, "Selection Error", "Please select a leave record to delete."
            )
            return
        name_item = self.ui.table_leaves.item(selected_row, 0)

        leave_id = name_item.data(Qt.UserRole)
        success = self.db.revoke_leave(leave_id)
        if success:
            QMessageBox.information(self.ui, "Success", "Leave revoked successfully.")
            self.load_all_leaves()

    def load_all_attendance(self):
        attendance_data = self.db.get_attendance_for_table()
        self.ui.table_attendance.setRowCount(len(attendance_data))
        self.ui.table_attendance.setColumnWidth(0, 150)
        self.ui.table_attendance.setColumnWidth(1, 150)
        print(attendance_data)
        for row_idx, attendance_record in enumerate(attendance_data):

            in_time_str = attendance_record["in_time"]
            out_time_str = attendance_record["out_time"]
            late_time, ot_time = self.calculate_times(in_time_str, out_time_str)
            print(late_time, ot_time)

            # print(attendance_record)
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
        leaves_data = self.db.get_all_leaves()
        self.ui.table_leaves.setRowCount(len(leaves_data))

        self.ui.table_leaves.setColumnWidth(0, 150)  # User Info
        self.ui.table_leaves.setColumnWidth(1, 150)  # Name
        self.ui.table_leaves.setColumnWidth(2, 100)  # Date
        self.ui.table_leaves.setColumnWidth(3, 120)  # Type
        self.ui.table_leaves.setColumnWidth(4, 200)  # Notes

        for row_idx, leave in enumerate(leaves_data):
            name_item = QTableWidgetItem(leave["user_id"])

            name_item.setData(Qt.UserRole, leave["id"])
            self.ui.table_leaves.setItem(row_idx, 0, name_item)
            self.ui.table_leaves.setItem(row_idx, 1, QTableWidgetItem(leave["name"]))
            self.ui.table_leaves.setItem(row_idx, 2, QTableWidgetItem(leave["date"]))
            self.ui.table_leaves.setItem(
                row_idx, 3, QTableWidgetItem(leave["leave_type"])
            )
            self.ui.table_leaves.setItem(
                row_idx, 4, QTableWidgetItem(leave["reason"] or "--")
            )

    def calculate_times(self, in_time_str, out_time_str):
        shift_start = datetime.strptime("08:00:00", "%H:%M:%S")
        shift_end = datetime.strptime("17:00:00", "%H:%M:%S")

        grace_period = timedelta(minutes=5)
        min_ot = timedelta(minutes=30)

        late_time = "--"
        ot_time = "--"

        if in_time_str:
            in_time = datetime.strptime(in_time_str, "%H:%M:%S")
            if in_time > (shift_start + grace_period):
                late_time = str(in_time - shift_start).split(".")[0]
                print(f"Late by: {late_time}")

        if out_time_str:
            out_time = datetime.strptime(out_time_str, "%H:%M:%S")
            if out_time > (shift_end + min_ot):
                ot_time = str(out_time - shift_end).split(".")[0]
                print(f"Overtime: {ot_time}")
        print(f"Calculated times - Late: {late_time}, OT: {ot_time}")
        return late_time, ot_time

    def load_users_to_leave_dropdown(self):
        self.ui.combo_leave_user.clear()

        self.ui.combo_leave_user.addItem("-- Select User --", None)

        users = self.db.get_users_for_table()
        search_list = []

        for user in users:
            display_text = f"{user['name']} (ID: {user['id']})"
            self.ui.combo_leave_user.addItem(display_text, user["id"])
            search_list.append(display_text)

        completer = QCompleter(search_list)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.ui.combo_leave_user.setCompleter(completer)

    def start_camera(self):
        users_data = self.db.load_users()
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
        attendance_status = self.db.mark_attendance(user_data["id"])
        print(
            f"Attendance status for {user_data['name']} (ID: {user_data['id']}): {attendance_status}"
        )
        if attendance_status == "IN":
            msg = f"✅ IN: Welcome {user_data['name']}!"
            self.ui.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold;")
            self.ui.lbl_status.setText(msg)

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
        print(f"Match found: {user_data['name']} (ID: {user_data['id']})")
        # self.ui.lbl_status.setText(f"Status: ✅ Identified {user_data['name']}")
        self.ui.lbl_res_name.setText(f"Name: {user_data['name']}")
        self.ui.lbl_res_id.setText(f"Student ID: {user_data['id']}")
        self.ui.lbl_res_age.setText(f"Age: {user_data['age']}")
        self.ui.txt_res_details.setText(user_data["details"])
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self.ui.lbl_res_image.width(),
            self.ui.lbl_res_image.height(),
            Qt.KeepAspectRatio,
        )
        self.ui.lbl_res_image.setPixmap(scaled)

    def show_no_match(self):
        self.ui.lbl_status.setText("Status: ❌ Unknown Face")
        self.ui.lbl_res_name.setText("Name: Unknown")
        self.ui.lbl_res_id.setText("Student ID: --")
        self.ui.lbl_res_age.setText("Age: --")
        self.ui.txt_res_details.setText("--")

    def show_cam_error(self, err_msg):
        QMessageBox.critical(self.ui, "Camera Error", err_msg)
        self.stop_camera()

    def on_reg_success(self, name):
        QMessageBox.information(
            self.ui, "Success", f"{name} has been added to the database."
        )
        self.ui.txt_reg_id.clear()
        self.ui.txt_reg_name.clear()
        self.ui.spin_reg_age.setValue(25)
        self.ui.txt_reg_details.clear()
        self.ui.btn_reg_save.setEnabled(True)
        self.ui.btn_reg_save.setText("💾 Save to Database")
        self.reg_img_path = ""
        self.load_all_users()
        self.load_all_attendance()

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


if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    attendance_system = AttendanceSystem()
    attendance_system.ui.show()
    sys.exit(app.exec())
