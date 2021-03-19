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

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('./ext'))


# -- Sphinx Version and Config -----------------------------------------------
# Sphinx will error and refuse to build if not equal to version
needs_sphinx='2.4.4'

# Sphinx-build fails for any broken __internal__ links. For external use make linkcheck.
nitpicky = True

# -- Project information -----------------------------------------------------

project = 'PETSc'
copyright = '1991-%d, UChicago Argonne, LLC and the PETSc Development Team' % datetime.date.today().year
author = 'The PETSc Development Team'

# Allow todo's to be emitted, turn off for build!
todo_include_todos=True
todo_emit_warnings=True

# Little copy-and-paste button by code blocks, from sphinx_copybutton package
# https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r"[>]{1,3}"
copybutton_prompt_is_regexp = True

with open(os.path.join('..', '..', '..', 'include', 'petscversion.h'),'r') as version_file:
    buf = version_file.read()
    petsc_release_flag = re.search(' PETSC_VERSION_RELEASE[ ]*([0-9]*)',buf).group(1)
    major_version      = re.search(' PETSC_VERSION_MAJOR[ ]*([0-9]*)',buf).group(1)
    minor_version      = re.search(' PETSC_VERSION_MINOR[ ]*([0-9]*)',buf).group(1)
    subminor_version   = re.search(' PETSC_VERSION_SUBMINOR[ ]*([0-9]*)',buf).group(1)
    patch_version      = re.search(' PETSC_VERSION_PATCH[ ]*([0-9]*)',buf).group(1)

    if petsc_release_flag == '0':
        version = subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')
        release = version
    else:
        version = '.'.join([major_version, minor_version])
        release = '.'.join([major_version,minor_version,subminor_version])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_copybutton',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'sphinxcontrib.rsvgconverter',
    'html5_petsc',
]

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

html_logo = os.path.join('..','website','images','PETSc-TAO_RGB.svg')
html_favicon = os.path.join('..','website','images','PETSc_RGB-logo.png')

# -- Options for LaTeX output --------------------------------------------

bibtex_bibfiles = ['../tex/petsc.bib','../tex/petscapp.bib']
latex_engine = 'xelatex'

# Specify how to arrange the documents into LaTeX files.
# This allows building only the manual.
latex_documents = [
        ('manual/index', 'manual.tex', 'PETSc Users Manual', author, 'manual', False)
        ]

latex_additional_files = [
    'manual/anl_tech_report/ArgonneLogo.pdf',
    'manual/anl_tech_report/ArgonneReportTemplateLastPage.pdf',
    'manual/anl_tech_report/ArgonneReportTemplatePage2.pdf',
    'manual/anl_tech_report/first.inc',
    'manual/anl_tech_report/last.inc',
]

latex_elements = {
    'maketitle': r'\newcommand{\techreportversion}{%s}' % version +
r'''
\input{first.inc}
\sphinxmaketitle
''',
    'printindex': r'''
\printindex
\input{last.inc}
''',
    'fontpkg': r'''
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
'''
}


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
numfig = True

# We must check what kind of builder the app uses to adjust
def builder_init_handler(app):
    import genteamtable
    print("============================================")
    print("    GENERATING TEAM TABLE FROM CONF.PY      ")
    print("============================================")
    genDirName = "generated"
    cwdPath = os.path.dirname(os.path.realpath(__file__))
    genDirPath = os.path.join(cwdPath, genDirName)
    if "PETSC_GITLAB_PRIVATE_TOKEN" in os.environ:
        token = os.environ["PETSC_GITLAB_PRIVATE_TOKEN"]
    else:
        token = None
    genteamtable.main(genDirPath, token, app.builder.name)
    return None

# Supposedly the safer way to add additional css files. Setting html_css_files will
# overwrite previous versions of the variable that some extension may have set. This will
# add our css files in addition to it.
def setup(app):
    # Register the builder_init_handler to be called __after__ app.builder has been initialized
    app.connect('builder-inited', builder_init_handler)
    app.add_css_file('css/pop-up.css')
    app.add_css_file('css/colorbox.css')
    app.add_css_file('css/petsc-team-container.css')
