#!/usr/bin/env python3

# -- Path setup --------------------------------------------------------------

import os
import sys
import builtins

builtins.__sphinx_build__ = True

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)

# -- Project information -----------------------------------------------------

project = "DRTK"
copyright = "2024, Meta"
author = "Meta"

release = "0.1"

# -- General configuration ---------------------------------------------------

# html_theme = 'furo'
# html_theme = "sphinx_book_theme"
html_theme = "pydata_sphinx_theme"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_markdown_builder',
              'sphinx.ext.autodoc',
              "sphinx.ext.autosummary",
              "sphinx.ext.viewcode",
              'sphinx.ext.napoleon',
              "sphinxcontrib.katex",
              "sphinx.ext.intersphinx",
              "myst_parser",
              "sphinx_design",
              ]
autosummary_generate = True
katex_prerender = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

master_doc = "index"
autodoc_typehints = "none"
html_context = {
#     "default_mode": "light"
}

html_theme_options = {
    # 'github_user': 'facebookresearch',
    # 'github_repo': 'DRTK',
    # 'description': "Differentiable Rendering Toolkit",
    # 'github_banner': True,
    # 'github_button': True,
    # 'sidebar_collapse': True,
    # "collapse_navbar": True,
    #
    # "repository_url": "https://github.com/facebookresearch/DRTK",
    # "use_repository_button": True,

    "navbar_align": "content",
    "navbar_start": ["navbar-logo"], # , "icon-links"
    # "navbar_end": ["navbar-icon-links"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],

    "collapse_navigation": True,
    "secondary_sidebar_items": ["page-toc"],

    "show_prev_next" : False,
    "back_to_top_button": False,

    "pygments_light_style" : 'a11y-light',
    "pygments_dark_style": 'a11y-dark',

    "github_user": "facebookresearch",
    "github_repo": "DRTK",
    "github_version": "main",
    "doc_path": "docs/source",

    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/facebookresearch/DRTK",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github fa-2xl",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
            "attributes": {
            }
        }
    ],
    "logo": {
        "text": "DRTK",
        # "image_light": "_static/logo-light.png",
        # "image_dark": "_static/logo-dark.png",
    }
}

html_static_path = ['_static']
html_css_files = ["custom.css"]
