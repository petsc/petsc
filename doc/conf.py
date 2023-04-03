# Configuration file for the Sphinx documentation builder.
#
# For information on options, see
#   http://www.sphinx-doc.org/en/master/config
#

import os
import sys
import subprocess
import re
import datetime
import shutil

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('./ext'))

import add_man_page_redirects
import build_classic_docs
import fix_man_page_edit_links
import make_links_relative
import update_htmlmap_links
import fix_pydata_margins


if not os.path.isdir("images"):
    print("-----------------------------------------------------------------------------")
    print("ERROR")
    print("images directory does not seem to exist.")
    print("To clone the required repository, try")
    print("   make images")
    print("-----------------------------------------------------------------------------")
    raise Exception("Aborting because images missing")


# -- Project information -------------------------------------------------------

project = 'PETSc'
copyright = '1991-%d, UChicago Argonne, LLC and the PETSc Development Team' % datetime.date.today().year
author = 'The PETSc Development Team'

with open(os.path.join('..', 'include', 'petscversion.h'),'r') as version_file:
    buf = version_file.read()
    petsc_release_flag = re.search(' PETSC_VERSION_RELEASE[ ]*([0-9]*)',buf).group(1)
    major_version      = re.search(' PETSC_VERSION_MAJOR[ ]*([0-9]*)',buf).group(1)
    minor_version      = re.search(' PETSC_VERSION_MINOR[ ]*([0-9]*)',buf).group(1)
    subminor_version   = re.search(' PETSC_VERSION_SUBMINOR[ ]*([0-9]*)',buf).group(1)

    git_describe_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')
    if petsc_release_flag == '0':
        version = git_describe_version
        release = git_describe_version
    else:
        version = '.'.join([major_version, minor_version])
        release = '.'.join([major_version,minor_version,subminor_version])


# -- General configuration -----------------------------------------------------

# The information on the next line must also be the same in requirements.txt
needs_sphinx='5.3'
nitpicky = True  # checks internal links. For external links, use "make linkcheck"
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build*', 'images', 'Thumbs.db', '.DS_Store','community/meetings/pre-2023']
highlight_language = 'c'
numfig = True

# -- Extensions ----------------------------------------------------------------

extensions = [
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'sphinxcontrib.rsvgconverter',
    'myst_parser',
    'html5_petsc',
    'sphinx_remove_toctrees',
]

copybutton_prompt_text = '$ '

bibtex_bibfiles = ['petsc.bib']

myst_enable_extensions = ["fieldlist", "dollarmath", "amsmath", "deflist"]

remove_from_toctrees = ['manualpages/*/[A-Z]*','changes/2*','changes/3*']

# -- Options for HTML output ---------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_logo_light = os.path.join('images', 'logos', 'PETSc_TAO_logos', 'PETSc-TAO', 'web', 'PETSc-TAO_RGB.svg')
html_logo_dark = os.path.join('images', 'logos', 'PETSc_TAO_logos', 'PETSc-TAO', 'web', 'PETSc-TAO_RGB_white.svg')

html_static_path = ['_static', html_logo_light, html_logo_dark]

# use much smaller font for h1, h2 etc. They are absurdly large in the standard style
# https://pydata-sphinx-theme.readthedocs.io/en/v0.12.0/user_guide/styling.html
html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/petsc/petsc",
            "icon": "fab fa-gitlab",
        },
    ],
    "use_edit_page_button": True,
    "footer_items": ["copyright", "sphinx-version", "last-updated"],
#    "secondary_sidebar_items" : ["edit-this-page"],
     "header_links_before_dropdown": 10,
    "logo": {
        "image_light": os.path.basename(html_logo_light),
        "image_dark": os.path.basename(html_logo_dark)
    }
}

try:
  git_ref = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
  git_ref_release = subprocess.check_output(["git", "rev-parse", "origin/release"]).rstrip()
  edit_branch = "release" if git_ref == git_ref_release else "main"
except subprocess.CalledProcessError:
  print("WARNING: determining branch for page edit links failed")
  edit_branch = "main"

html_context = {
    "github_url": "https://gitlab.com",
    "github_user": "petsc",
    "github_repo": "petsc",
    "github_version": edit_branch,
    "doc_path": "doc",
}

html_logo = html_logo_light
html_favicon = os.path.join('images', 'logos', 'PETSc_TAO_logos', 'PETSc', 'petsc_favicon.png')
html_last_updated_fmt = r'%Y-%m-%dT%H:%M:%S%z (' + git_describe_version + ')'


# -- Options for LaTeX output --------------------------------------------------
latex_engine = 'xelatex'

# How to arrange the documents into LaTeX files, building only the manual.
latex_documents = [
        ('manual/index', 'manual.tex', 'PETSc/TAO Users Manual', author, 'manual', False)
        ]

latex_additional_files = [
    'images/manual/anl_tech_report/ArgonneLogo.pdf',
    'images/manual/anl_tech_report/ArgonneReportTemplateLastPage.pdf',
    'images/manual/anl_tech_report/ArgonneReportTemplatePage2.pdf',
    'manual/anl_tech_report/first.inc',
    'manual/anl_tech_report/last.inc',
]

latex_elements = {
    'maketitle': r'\newcommand{\techreportversion}{%s}' % version +
r'''
\input{first.inc}
''',
    'printindex': r'''
\printindex
\input{last.inc}
''',
    'fontpkg': r'''
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
    'tableofcontents' : r''
}


# -- Setup and event callbacks -------------------------------------------------

def setup(app):
        app.connect('builder-inited', builder_init_handler)
        app.connect('build-finished', build_finished_handler)


def builder_init_handler(app):
    if app.builder.name.endswith('html'):
        _build_classic_docs(app, 'pre')
        _update_htmlmap_links(app)


def build_finished_handler(app, exception):
    if app.builder.name.endswith('html'):
        _build_classic_docs(app, 'post')
        _fix_links(app, exception)
        _fix_man_page_edit_links(app, exception)
        fix_pydata_margins.fix_pydata_margins(app.outdir)
        if app.builder.name == 'dirhtml':
            _add_man_page_redirects(app, exception)
        # remove sources for manual pages since they are automatically generated and should not be looked at on the website
        if os.path.isdir(os.path.join(app.outdir,'_sources','manualpages')):
            shutil.rmtree(os.path.join(app.outdir,'_sources','manualpages'))
        if app.builder.name == 'html':
            print("==========================================================================")
            print("    open %s/index.html in your browser to view the documentation " % app.outdir)
            print("==========================================================================")

def _add_man_page_redirects(app, exception):
    if exception is None:
        print("============================================")
        print("    Adding man pages redirects")
        print("============================================")
        add_man_page_redirects.add_man_page_redirects(app.outdir)

def _build_classic_docs(app, stage):
    '''Builds the .md versions of the manual pages and the .html version of the source code'''
    build_classic_docs.main(stage,app.outdir)

def _fix_man_page_edit_links(app, exception):
    if exception is None:
        print("============================================")
        print("    Fixing man page edit links")
        print("============================================")
        fix_man_page_edit_links.fix_man_page_edit_links(app.outdir)

#
#   The following two scripts are needed because the Sphinx html and dirhtml builds save the output html
#   files at different levels of the directory hierarchy. file.rst -> file.html with html but
#   file.rst -> file/index.html with dirhtml and we want both to work correctly using relative links.

def _fix_links(app, exception):
    """We need to manage our own relative paths in the User's Manual for the source code files which
       are auto-generated by c2html outside of Sphinx so Sphinx cannot directly handle those links for use.
       We use the string PETSC_DOC_OUT_ROOT_PLACEHOLDER in URLs in the Sphinx .rst files as a stand in
       for the root directory that needs to be constructed based on if the Sphinx build is html or dirhtml
    """
    if exception is None:
        print("============================================")
        print("    Fixing relative links")
        print("============================================")
        make_links_relative.make_links_relative(app.outdir)


def _update_htmlmap_links(app):
    """htmlmap maps from manualpage names to relative locations in the generated documentation directory
       hierarchy. The format of the directory location needs to be different for the Sphinx html and dirhtml
       builds
    """
    print("============================================")
    print("    Updating htmlmap")
    print("============================================")
    update_htmlmap_links.update_htmlmap_links(app.builder,os.path.join('manualpages','htmlmap'))
