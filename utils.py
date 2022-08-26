from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QComboBox, QCompleter


def completion(word_list, widget, i=True):
    """ Autocompletion of sender and subject """
    word_set = set(word_list)
    completer = QCompleter(word_set)
    if i:
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    else:
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    widget.setCompleter(completer)


class Autocomplete(QComboBox):
    def __init__(self, items, parent=None, i=False, allow_duplicates=True):
        super(Autocomplete, self).__init__(parent)
        self.items = items
        self.insensitivity = i
        self.allowDuplicates = allow_duplicates
        self.init()

    def init(self):
        self.setEditable(True)
        self.setDuplicatesEnabled(self.allowDuplicates)
        self.addItems(self.items)
        self.setAutocompletion(self.items, i=self.insensitivity)

    def setAutocompletion(self, items, i):
        completion(items, self, i)