# Part of Spatial Math Toolbox for Python
# Copyright (c) 2000 Peter Corke
# MIT Licence, see details in top-level file: LICENCE

"""
Detect the interactive environment that Python is running in.

This module deliberately does not import Matplotlib, so it is cheap to import
and works when Matplotlib is not installed.
"""

import sys
from typing import Literal

__all__ = ["environment", "isnotebook"]

Environment = Literal["terminal", "ipython", "jupyter", "colab", "jupyterlite"]


def environment() -> Environment:
    """
    Determine the environment that Python is running in

    :return: the environment
    :rtype: str

    The result is one of:

    ========================  ====================================================
    ``"terminal"``            standard Python interpreter, or script
    ``"ipython"``             IPython running in a terminal
    ``"jupyter"``             Jupyter kernel, eg. JupyterLab, Notebook, VS Code or
                              qtconsole
    ``"colab"``               Google Colab
    ``"jupyterlite"``         JupyterLite, running in the browser (Pyodide)
    ========================  ====================================================

    .. note:: The different Jupyter front ends (JupyterLab, classic Notebook,
        VS Code, Jupyter Desktop) cannot be reliably told apart from within the
        kernel, so all are reported as ``"jupyter"``.  How figures will be
        rendered depends on the Matplotlib backend, not on the front end.

    :seealso: :func:`isnotebook`
    """
    # Colab and JupyterLite both also provide an IPython kernel, so test for
    # them first
    if "google.colab" in sys.modules:
        return "colab"
    if sys.platform == "emscripten":
        return "jupyterlite"

    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        return "terminal"  # probably standard Python interpreter

    if shell == "ZMQInteractiveShell":
        return "jupyter"  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return "ipython"
    else:
        return "terminal"  # other type (?)


def isnotebook() -> bool:
    """
    Determine if code is being run from a Jupyter notebook

    :return: True if running in a Jupyter kernel, Google Colab or JupyterLite
    :rtype: bool

    :seealso: :func:`environment`

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/39662359#39662359
    """
    return environment() in ("jupyter", "colab", "jupyterlite")
