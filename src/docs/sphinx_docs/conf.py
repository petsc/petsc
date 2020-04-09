# Configuration file for the Sphinx documentation builder.
#
# Much of this file was generated automatically with sphinx-quickstart
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess
import re
import datetime

sys.path.append(os.path.abspath('./ext'))

# -- Project information -----------------------------------------------------

project = 'PETSc'
copyright = str(datetime.date.today().year)
author = 'PETSc'

with open(os.path.join('..', '..', '..', 'include', 'petscversion.h'),'r') as version_file:
    buf = version_file.read()
    petsc_release_flag = re.search(' PETSC_VERSION_RELEASE[ ]*([0-9]*)',buf).group(1)
    major_version      = re.search(' PETSC_VERSION_MAJOR[ ]*([0-9]*)',buf).group(1)
    minor_version      = re.search(' PETSC_VERSION_MINOR[ ]*([0-9]*)',buf).group(1)
    subminor_version   = re.search(' PETSC_VERSION_SUBMINOR[ ]*([0-9]*)',buf).group(1)
    patch_version      = re.search(' PETSC_VERSION_PATCH[ ]*([0-9]*)',buf).group(1)

    if petsc_release_flag == '0':
        version = 'dev'
        release = 'dev'
    else:
        version = '.'.join([major_version, minor_version])
        release = '.'.join([major_version,minor_version,subminor_version])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []
extensions.append('sphinx.ext.graphviz')
extensions.append('sphinx.ext.mathjax')
extensions.append('sphinxcontrib.bibtex')
extensions.append('html5_petsc')          # Overrides HTML5Translator

master_doc = 'index'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- General Config Options ---------------------------------------------------

# Graphviz config which searches for correct installation of a DOT language parser
# shipped with graphviz

try:
    result = subprocess.check_output(
        "which -a dot",
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True)
    result = result[:-1]
except subprocess.CalledProcessError as e:
    print("\nCan't find a working graphviz install!")
else:
    print("\nFound DOT install: {}\n".format(result))

graphviz_dot = str(result)

# Set default highlighting language
highlight_language = 'c'
autosummary_generate = True
