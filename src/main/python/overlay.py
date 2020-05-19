from screen_reader import ScreenReader
import fight as calculator
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QMainWindow, QStyle, qApp, QWidget, QVBoxLayout, QLabel


class Overlay(QMainWindow):
    def __init__(self, application_context):
        self.application_context = application_context
        self.reader = ScreenReader(application_context)

        QMainWindow.__init__(self)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QStyle.alignedRect(
                Qt.LeftToRight,
                Qt.AlignCenter |
                Qt.AlignRight,
                QSize(220, 700),
                qApp.desktop().availableGeometry()
            ))

        self.setCentralWidget(QWidget(self))
        layout = QVBoxLayout()
        self.centralWidget().setLayout(layout)

        self.accuracy_label = QLabel()
        self.damage_label = QLabel()
        self.tokens_label = QLabel()

        self.cumulative_probabilities_label = QLabel()
        self.exact_probabilities_label = QLabel()
        self.expected_damage_label = QLabel()

        layout.addWidget(self.accuracy_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.damage_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.tokens_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.cumulative_probabilities_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.exact_probabilities_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.expected_damage_label, alignment=Qt.AlignCenter)

        dummy_character = calculator.Character(0, 0, 0)
        self.dummy_encounter = calculator.Encounter([dummy_character])

    def mousePressEvent(self, event):
        qApp.quit()

    def gui_updater(self):
        accuracy = self.reader.get_accuracy()
        damage = self.reader.get_damage()
        tokens = self.reader.get_tokens()

        attack = calculator.Attack(damage, accuracy, tokens, True)

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


