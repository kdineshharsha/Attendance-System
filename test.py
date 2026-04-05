import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QBarSet,
    QBarSeries,
    QBarCategoryAxis,
    QValueAxis,
    QPieSeries,
)
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtCore import Qt


class ChartTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Charts Testing Sandbox")
        self.resize(1000, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")  # Dark Theme

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 1. Bar Chart එක හදමු (Weekly Attendance)
        bar_chart_view = self.create_bar_chart()
        layout.addWidget(bar_chart_view)

        # 2. Pie Chart එක හදමු (Today's Status)
        pie_chart_view = self.create_pie_chart()
        layout.addWidget(pie_chart_view)

    def create_bar_chart(self):
        """සතියේ දවස් 5ට අදාළව Present, Late, Absent ගාණ පෙන්වන Bar Chart එක"""

        # 1. දත්ත (Data Sets) හදාගැනීම
        set_present = QBarSet("Present")
        set_late = QBarSet("Late")
        set_absent = QBarSet("Absent")

        # Dummy Data (සඳුදා ඉඳන් සිකුරාදා වෙනකන්)
        set_present.append([40, 42, 38, 45, 39])
        set_late.append([5, 3, 7, 2, 6])
        set_absent.append([5, 5, 5, 3, 5])

        # පාට වෙනස් කිරීම
        set_present.setColor(QColor("#a6e3a1"))  # Green
        set_late.setColor(QColor("#f9e2af"))  # Yellow
        set_absent.setColor(QColor("#f38ba8"))  # Red

        # 2. Series එකට දත්ත ටික දැමීම
        series = QBarSeries()
        series.append(set_present)
        series.append(set_late)
        series.append(set_absent)

        # 3. Chart එක හැදීම
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Weekly Attendance Summary")
        chart.setAnimationOptions(QChart.SeriesAnimations)  # ලස්සන ඇනිමේෂන් එකක් දානවා
        chart.setBackgroundBrush(QColor("#313244"))
        chart.setTitleBrush(QColor("#cdd6f4"))
        chart.legend().setLabelColor(QColor("#cdd6f4"))

        # X අක්ෂය (දවස් ටික)
        categories = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsColor(QColor("#cdd6f4"))
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        # Y අක්ෂය (ගණන)
        axis_y = QValueAxis()
        axis_y.setRange(0, 50)
        axis_y.setLabelsColor(QColor("#cdd6f4"))
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # 4. Chart View එකට දාලා රෙන්ඩර් කිරීම
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)  # කොන් සුමට කිරීම (Smooth edges)
        return chart_view

    def create_pie_chart(self):
        """අද දවසේ ළමයි ඇවිත් ඉන්න ප්‍රතිශතය පෙන්වන Pie Chart එක"""

        # 1. දත්ත (Series) හැදීම
        series = QPieSeries()

        # Dummy Data එකතු කිරීම (Text, Value)
        slice_present = series.append("Present (40)", 40)
        slice_late = series.append("Late (5)", 5)
        slice_leave = series.append("On Leave (3)", 3)
        slice_absent = series.append("Absent (2)", 2)

        # 2. කෑලි (Slices) වල පාට වෙනස් කිරීම
        slice_present.setBrush(QColor("#a6e3a1"))
        slice_late.setBrush(QColor("#f9e2af"))
        slice_leave.setBrush(QColor("#89b4fa"))
        slice_absent.setBrush(QColor("#f38ba8"))

        # වැඩියෙන්ම ඉන්න සෙට් එක (Present) ටිකක් එළියට පන්නලා පෙන්වනවා (Explode)
        slice_present.setExploded(True)
        slice_present.setLabelVisible(True)
        slice_present.setLabelColor(QColor("#cdd6f4"))

        # 3. Chart එක හැදීම
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Today's Status Breakdown")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setBackgroundBrush(QColor("#313244"))
        chart.setTitleBrush(QColor("#cdd6f4"))
        chart.legend().setLabelColor(QColor("#cdd6f4"))
        chart.legend().setAlignment(Qt.AlignBottom)

        # 4. Chart View එක හැදීම
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        return chart_view


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChartTester()
    window.show()
    sys.exit(app.exec())
