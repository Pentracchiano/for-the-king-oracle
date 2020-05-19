import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer
import reader.screen_reader as reader
import calculator.fight


class Overlay(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight,
                QtCore.Qt.AlignCenter |
                QtCore.Qt.AlignRight,
                QtCore.QSize(220, 700),
                QtWidgets.qApp.desktop().availableGeometry()
            ))

        self.setCentralWidget(QtWidgets.QWidget(self))
        layout = QtWidgets.QVBoxLayout()
        self.centralWidget().setLayout(layout)

        self.accuracy_label = QtWidgets.QLabel()
        self.damage_label = QtWidgets.QLabel()
        self.tokens_label = QtWidgets.QLabel()

        self.cumulative_probabilities_label = QtWidgets.QLabel()
        self.exact_probabilities_label = QtWidgets.QLabel()
        self.expected_damage_label = QtWidgets.QLabel()

        layout.addWidget(self.accuracy_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.damage_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.tokens_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.cumulative_probabilities_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.exact_probabilities_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.expected_damage_label, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(layout)

        dummy_character = calculator.fight.Character(0, 0, 0)
        self.dummy_encounter = calculator.fight.Encounter([dummy_character])

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()

    def gui_updater(self):
        accuracy = reader.get_accuracy()
        damage = reader.get_damage()
        tokens = reader.get_tokens()

        attack = calculator.fight.Attack(damage, accuracy, tokens, True)

        exact_string = ""
        for item in self.dummy_encounter.get_damage_exact_probabilities(attack, 0):
            exact_string += f"Damage: {item['damage']}\tProbability: {item['probability']:.3f}\n"

        cumulative_string = ""
        for item in self.dummy_encounter.get_damage_cumulative_probabilities(attack, 0):
            cumulative_string += f"Damage: {item['damage']}\tProbability: {item['probability']:.3f}\n"

        self.accuracy_label.setText(f"Accuracy: {accuracy}")
        self.damage_label.setText(f"Damage: {damage}")
        self.tokens_label.setText(f"Tokens: {tokens}")

        self.exact_probabilities_label.setText("EXACT PROBABILITIES:\n\n" + exact_string)
        self.cumulative_probabilities_label.setText("CUMULATIVE PROBABILITIES:\n\n" + cumulative_string)
        self.expected_damage_label.setText(f"Expected damage: {self.dummy_encounter.expected_damage(attack, 0)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    overlay = Overlay()
    gui_updater_timer = QTimer(overlay)
    gui_updater_timer.timeout.connect(overlay.gui_updater)
    gui_updater_timer.start(100)

    overlay.show()
    app.exec_()
