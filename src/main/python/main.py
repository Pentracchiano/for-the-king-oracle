from PyQt5.QtCore import QTimer
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from overlay import Overlay

if __name__ == '__main__':
    app_context = ApplicationContext()
    overlay = Overlay(app_context)
    gui_updater_timer = QTimer(overlay)
    gui_updater_timer.timeout.connect(overlay.gui_updater)
    gui_updater_timer.start(100)

    overlay.show()
    app_context.app.exec_()
