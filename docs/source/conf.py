# spatialmath
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('..'))

from sphinx_codeautolink import clean_ipython, clean_pycon

# -- Project information -----------------------------------------------------

project = "Spatial Maths package"
copyright = "2020-, Peter Corke."
author = "Peter Corke"
try:
    import spatialmath

    version = spatialmath.__version__
except AttributeError:
    import re

    with open("../../pyproject.toml", "r") as f:
        m = re.compile(r'version\s*=\s*"([0-9\.]+)"').search(f.read())
        version = m[1]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinx_autorun",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx_copybutton",
    "sphinx_codeautolink",
]

suppress_warnings = [
    "codeautolink.match_block",
    "codeautolink.match_name",
    "config.cache",  # codeautolink_custom_blocks holds function refs, not picklable
]
#'sphinx.ext.autosummary',
# typehints_use_signature_return = True

# inheritance_node_attrs = dict(style='rounded,filled', fillcolor='lightblue')
inheritance_node_attrs = dict(style="rounded")

autosummary_generate = True
autodoc_member_order = "groupwise"
# bysource

# options for spinx_autorun, used for inline examples
#  choose UTF-8 encoding to allow for Unicode characters, eg. ansitable
#  Python session setup, turn off color printing for SE3, set NumPy precision
autorun_languages = {}
autorun_languages["pycon_output_encoding"] = "UTF-8"
autorun_languages["pycon_input_encoding"] = "UTF-8"
autorun_languages[
    "pycon_runfirst"
] = """
from spatialmath import SE3
SE3._color = False
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from ansitable import ANSITable
ANSITable._color = False
"""


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["test_*"]

add_module_names = False
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'
# html_theme = 'pyramid'
# html_theme = 'sphinxdoc'

html_theme_options = {
    #'github_user': 'petercorke',
    #'github_repo': 'spatialmath-python',
    #'logo_name': False,
    "logo_only": False,
    #'description': 'Spatial maths and geometry for Python',
    "display_version": True,
    "prev_next_buttons_location": "both",
    "analytics_id": "G-11Q6WJM565",
}
html_logo = "../figs/CartesianSnakes_LogoW.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# autodoc_mock_imports = ["numpy", "scipy"]
html_last_updated_fmt = "%d-%b-%Y"
# extensions = ['rst2pdf.pdfbuilder']
# pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Your Name'),]
latex_engine = "xelatex"
# maybe need to set graphics path in here somewhere
# \graphicspath{{figures/}{../figures/}{C:/Users/me/Documents/project/figures/}}
# https://stackoverflow.com/questions/63452024/how-to-include-image-files-in-sphinx-latex-pdf-files
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    #'releasename':" ",
    # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
    # 'fncychap': '\\usepackage[Lenny]{fncychap}',
    "fncychap": "\\usepackage{fncychap}",
}

# -------- RVC maths notation -------------------------------------------------------#

# see https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx
mathjax3_config = {
    "tex": {
        "macros": {
            # RVC Math notation
            #  - not possible to do the if/then/else approach
            #  - subset only
            "presup": [r"\,{}^{\scriptscriptstyle #1}\!", 1],
            # groups
            "SE": [r"\mathbf{SE}(#1)", 1],
            "SO": [r"\mathbf{SO}(#1)", 1],
            "se": [r"\mathbf{se}(#1)", 1],
            "so": [r"\mathbf{so}(#1)", 1],
            # vectors
            "vec": [r"\boldsymbol{#1}", 1],
            "dvec": [r"\dot{\boldsymbol{#1}}", 1],
            "ddvec": [r"\ddot{\boldsymbol{#1}}", 1],
            "fvec": [r"\presup{#1}\boldsymbol{#2}", 2],
            "fdvec": [r"\presup{#1}\dot{\boldsymbol{#2}}", 2],
            "fddvec": [r"\presup{#1}\ddot{\boldsymbol{#2}}", 2],
            "norm": [r"\Vert #1 \Vert", 1],
            # matrices
            "mat": [r"\mathbf{#1}", 1],
            "dmat": [r"\dot{\mathbf{#1}}", 1],
            "fmat": [r"\presup{#1}\mathbf{#2}", 2],
            # skew matrices
            "sk": [r"\left[#1\right]", 1],
            "skx": [r"\left[#1\right]_{\times}", 1],
            "vex": [r"\vee\left( #1\right)", 1],
            "vexx": [r"\vee_{\times}\left( #1\right)", 1],
            # quaternions
            "q": r"\mathring{q}",
            "fq": [r"\presup{#1}\mathring{q}", 1],
        }
    }
}


autorun_languages = {}
autorun_languages["pycon_output_encoding"] = "UTF-8"
autorun_languages["pycon_input_encoding"] = "UTF-8"
autorun_languages[
    "pycon_runfirst"
] = """
from spatialmath import SE3
SE3._color = False
import numpy as np
np.set_printoptions(precision=4, suppress=True)
"""

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -------- Options favicon -------------------------------------------------------#

html_static_path = ["_static"]
# create favicons online using https://favicon.io/favicon-converter/
favicons = [
    {
        "rel": "icon",
        "sizes": "16x16",
        "href": "favicon-16x16.png",
        "type": "image/png",
    },
    {
        "rel": "icon",
        "sizes": "32x32",
        "href": "favicon-32x32.png",
        "type": "image/png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "apple-touch-icon.png",
        "type": "image/png",
    },
    {
        "rel": "android-chrome",
        "sizes": "192x192",
        "href": "android-chrome-192x192.png",
        "type": "image/png",
    },
    {
        "rel": "android-chrome",
        "sizes": "512x512",
        "href": "android-chrome-512x512.png",
        "type": "image/png",
    },
]

autodoc_type_aliases = {"SO3Array": "SO3Array"}
