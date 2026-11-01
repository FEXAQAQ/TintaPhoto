"""Reading the interface text again after the language is changed (U-02).

Every string on screen is fetched with :func:`app.i18n.tr` while the widget
that shows it is built, so changing the language has no effect on what is
already on screen. Rather than rebuilding the window, which would throw away
the open photo and where the user had got to, each widget that holds text
offers a ``retranslate()`` method that sets its text again, and this walks the
window and calls them.

A widget's own ``retranslate()`` runs before its children are looked up, so a
widget is free to rebuild its children there: the new ones are walked and the
ones it replaced are never touched.
"""

from __future__ import annotations

from PySide6.QtCore import QObject


def retranslate_tree(root: QObject) -> int:
    """Ask *root* and everything under it to read its text again.

    Returns how many objects answered, which is what the tests count.
    """
    answered = 0
    retranslate = getattr(root, "retranslate", None)
    if callable(retranslate):
        retranslate()
        answered += 1
    for child in root.children():
        answered += retranslate_tree(child)
    return answered
