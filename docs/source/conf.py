# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import inspect
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any


import logging

# Configure the logger
logging.basicConfig(
    filename='docs.log',  # Log file name
    level=logging.DEBUG,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


# import sphinx_autosummary_accessors

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, str(Path("../..").resolve()))


# -- Project information -----------------------------------------------------

project = 'Polars Extensions'
copyright = '2025, Jayden Rasband'
author = 'Jayden Rasband'
release = '1.2.1'

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    # Third-party extensions
    "autodocsumm",
    "numpydoc",
    # "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
    "sphinx_reredirects",
    "sphinx_toolbox.more_autodoc.overloads",
    "recommonmark"
]

# Render docstring text in `single backticks` as code.
default_role = "code"

maximum_signature_line_length = 88

# Below setting is used by
# sphinx-autosummary-accessors - build docs for namespace accessors like `Series.str`
# https://sphinx-autosummary-accessors.readthedocs.io/en/stable/
templates_path = ["_templates", 
                #   sphinx_autosummary_accessors.templates_path
                  ]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store"]

# Hide overload type signatures
# sphinx_toolbox - Box of handy tools for Sphinx
# https://sphinx-toolbox.readthedocs.io/en/latest/
overloads_location = ["bottom"]
language = 'python'


# -- Options for HTML output -------------------------------------------------


html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]  # relative to html_static_path
html_show_sourcelink = False

# key site root paths
static_assets_root = ""
github_root = "https://github.com/jrasband-dev/polars-extensions"
web_root = "https://jrasband-dev.github.io/polars-extensions/"
asset_root = "https://raw.githubusercontent.com/jrasband-dev/polars-extensions/main"

# Specify version for version switcher dropdown menu
git_ref = os.environ.get("POLARS_EXTENSIONS_VERSION", "main")
version_match = re.fullmatch(r"py-(\d+)\.\d+\.\d+.*", git_ref)
switcher_version = version_match.group(1) if version_match is not None else "dev"

# html_js_files = [
#     (
#         "https://plausible.io/js/script.js",
#         {"data-domain": "docs.pola.rs,combined.pola.rs", "defer": "defer"},
#     ),
# ]

html_theme_options = {
    # "external_links": [
    #     {
    #         "name": "User guide",
    #         "url": f"{web_root}/",
    #     },
    # ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_root,
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Pypi",
            "url": "https://pypi.org/project/polars-extensions/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/in/jayden-rasband-303449133/",
            "icon": "fa-brands fa-linkedin",
        },
        {
            "name": "YouTube",
            "url": "https://www.youtube.com/@polarsCodeAcademy",
            "icon": "fa-brands fa-youtube",
        },
    ],
    # "logo": {
    #     "image_light": f"{asset_root}/static/metric-forge-black.png",
    #     "image_dark": f"{asset_root}/static/metric-forge-white.png",
    # },
    # "switcher": {
    #     "json_url": f"{web_root}/api/python/dev/_static/version_switcher.json",
    #     "version_match": switcher_version,
    # },
    "show_version_warning_banner": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "check_switcher": False,
}

# sphinx-favicon - Add support for custom favicons
# https://github.com/tcmetzger/sphinx-favicon

# favicons = [
#     {
#         "rel": "icon",
#         "sizes": "32x32",
#         "href": f"{static_assets_root}/icons/favicon-32x32.png",
#     },
#     {
#         "rel": "apple-touch-icon",
#         "sizes": "180x180",
#         "href": f"{static_assets_root}/icons/touchicon-180x180.png",
#     },
# ]


# sphinx-ext-linkcode - Add external links to source code
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
def linkcode_resolve(domain: str, info: dict[str, Any]) -> str | None:
    """
    Determine the URL corresponding to Python object.

    Based on pandas equivalent:
    https://github.com/pandas-dev/pandas/blob/main/doc/source/conf.py#L629-L686
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    conf_dir_path = Path(__file__).absolute().parent
    project_root = (conf_dir_path.parent.parent / "polars-extensions").absolute()

    fn = os.path.relpath(fn, start=project_root)
    return f"{github_root}/blob/{git_ref}/polars-extensions/{fn}{linespec}"


def _minify_classpaths(s: str) -> str:
    logging.debug(f'classpath: {s}')
    return re.sub(
        pattern=r"metric_forge\.ecommerce\.([^.]+\.[^.]+)",  
        repl=r"\1",  # Replace with just the captured part
        string=s,
    )



def process_signature(  # noqa: D103
    app: object,
    what: object,
    name: object,
    obj: object,
    opts: object,
    sig: str,
    ret: str,
) -> tuple[str, str]:
    logging.debug(f'process_sig: {sig}')
    logging.debug(f'return_sig: {ret}')
    return (
        _minify_classpaths(sig) if sig else sig,
        _minify_classpaths(ret) if ret else ret,
    )


def setup(app: Any) -> None:  # noqa: D103
    app.connect("autodoc-process-signature", process_signature)