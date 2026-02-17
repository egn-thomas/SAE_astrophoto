import sys
from PyQt5.QtWidgets import QApplication
from model import Model
from view import View
from controller import Controller   

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = Model()
    view = View()
    controller = Controller(model, view)
    view.show()
    sys.exit(app.exec_())