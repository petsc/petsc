# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import typing
import datetime
import importlib
import sphobjinv
import functools
from sphinx.ext.napoleon.docstring import NumpyDocstring

sys.path.insert(0, os.path.abspath('.'))
_today = datetime.datetime.now()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

package = 'petsc4py'


def pkg_version():
    import re
    here = os.path.dirname(__file__)
    pardir = [os.path.pardir] * 2
    topdir = os.path.join(here, *pardir)
    srcdir = os.path.join(topdir, 'src')
    with open(os.path.join(srcdir, 'petsc4py', '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]


project = 'PETSc for Python'
author = 'Lisandro Dalcin'
copyright = f'{_today.year}, {author}'

release = pkg_version()
version = release.rsplit('.', 1)[0]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

needs_sphinx = '5.0.0'

default_role = 'any'

nitpicky = True
nitpick_ignore = [
    ('envvar', 'NUMPY_INCLUDE'),
    ('py:class', 'ndarray'),  # FIXME
]
nitpick_ignore_regex = [
    (r'c:.*', r'MPI_.*'),
    (r'c:.*', r'Petsc.*'),
    (r'envvar', r'(LD_LIBRARY_)?PATH'),
    (r'envvar', r'(MPICH|OMPI|MPIEXEC)_.*'),
]

toc_object_entries = False
toc_object_entries_show_parents = 'hide'
# python_use_unqualified_type_names = True

autodoc_class_signature = 'separated'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_mock_imports = []
autodoc_type_aliases = {
}

autosummary_context = {
    'synopsis': {},
    'autotype': {},
}


def _mangle_petsc_intersphinx():
    """Preprocess the keys in PETSc's intersphinx inventory.

    PETSc have intersphinx keys of the form::

        manualpages/Vec/VecShift

    instead of:

        petsc.VecShift

    This function downloads their object inventory and strips the leading path
    elements so that references to PETSc names actually resolve."""
    inv = sphobjinv.Inventory(url="https://petsc.org/main/objects.inv")

    for obj in inv.objects:
        if obj.name.startswith("manualpages"):
            obj.name = "petsc." + "/".join(obj.name.split("/")[2:])
            obj.role = "class"
            obj.domain = "py"

    sphobjinv.writebytes("petsc_objects.inv",
                         sphobjinv.compress(inv.data_file(contract=True)))


_mangle_petsc_intersphinx()


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
    'pyopencl': ('https://documen.tician.de/pyopencl/', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest/', None),
    'petsc': ('https://petsc.org/main/', 'petsc_objects.inv'),
}

napoleon_preprocess_types = True

try:
    import sphinx_rtd_theme
    if 'sphinx_rtd_theme' not in extensions:
        extensions.append('sphinx_rtd_theme')
except ImportError:
    sphinx_rtd_theme = None


def _setup_numpy_typing():
    try:
        import numpy as np
    except ImportError:
        np = type(sys)('numpy')
        sys.modules[np.__name__] = np
        np.dtype = type('dtype', (), {})
        np.dtype.__module__ = np.__name__

    try:
        import numpy.typing as npt
    except ImportError:
        npt = type(sys)('numpy.typing')
        np.typing = npt
        sys.modules[npt.__name__] = npt
        npt.__all__ = []
        for attr in ['ArrayLike', 'DTypeLike']:
            setattr(npt, attr, typing.Any)
            npt.__all__.append(attr)


def _setup_mpi4py_typing():
    try:
        import mpi4py
    except ImportError:
        pkg = type(sys)('mpi4py')
        sys.modules[pkg.__name__] = pkg
        pkg.mod = type(sys)('mpi4py.MPI')
        sys.modules[pkg.mod.__name__] = pkg.mod
        for clsname in (
            'Intracomm',
            'Datatype',
            'Op',
        ):
            cls = type(clsname, (), {})
            cls.__module__ = pkg.mod.__name__
            setattr(pkg.mod, clsname, cls)


def _patch_domain_python():
    try:
        from numpy.typing import __all__ as numpy_types
    except ImportError:
        numpy_types = []

    numpy_types = set(numpy_types)
    for name in numpy_types:
        autodoc_type_aliases[name] = f'~numpy.typing.{name}'

    from sphinx.domains.python import PythonDomain
    PythonDomain.object_types['data'].roles += ('class',)


def _setup_autodoc(app):
    from sphinx.ext import autodoc
    from sphinx.util import inspect
    from sphinx.util import typing

    #

    def stringify_annotation(annotation, mode='fully-qualified-except-typing'):
        qualname = getattr(annotation, '__qualname__', '')
        module = getattr(annotation, '__module__', '')
        args = getattr(annotation, '__args__', None)
        if module == 'builtins' and qualname and args is not None:
            args = ', '.join(stringify_annotation(a, mode) for a in args)
            return f'{qualname}[{args}]'
        return stringify_annotation_orig(annotation, mode)

    try:
        stringify_annotation_orig = typing.stringify_annotation
        inspect.stringify_annotation = stringify_annotation
        typing.stringify_annotation = stringify_annotation
        autodoc.stringify_annotation = stringify_annotation
        autodoc.typehints.stringify_annotation = stringify_annotation
    except AttributeError:
        stringify_annotation_orig = typing.stringify
        inspect.stringify_annotation = stringify_annotation
        typing.stringify = stringify_annotation
        autodoc.stringify_typehint = stringify_annotation

    #

    class ClassDocumenterMixin:

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.config.autodoc_class_signature == 'separated':
                members = self.options.members
                special_members = self.options.special_members
                if special_members is not None:
                    for name in ('__new__', '__init__'):
                        if name in members:
                            members.remove(name)
                        if name in special_members:
                            special_members.remove(name)

    class ClassDocumenter(
        ClassDocumenterMixin,
        autodoc.ClassDocumenter,
    ):
        pass

    class ExceptionDocumenter(
        ClassDocumenterMixin,
        autodoc.ExceptionDocumenter,
    ):
        pass

    app.add_autodocumenter(ClassDocumenter, override=True)
    app.add_autodocumenter(ExceptionDocumenter, override=True)


def _monkey_patch_see_also():
    """Rewrite the role of names in "see also" sections.

    Napoleon uses :obj: for all names found in "see also" sections but we
    need :all: so that references to labels work."""

    _parse_numpydoc_see_also_section = \
        NumpyDocstring._parse_numpydoc_see_also_section

    @functools.wraps(NumpyDocstring._parse_numpydoc_see_also_section)
    def wrapper(*args, **kwargs):
        out = _parse_numpydoc_see_also_section(*args, **kwargs)
        return [line.replace(":obj:", ":any:") for line in out]

    NumpyDocstring._parse_numpydoc_see_also_section = wrapper


_monkey_patch_see_also()


def setup(app):
    _setup_numpy_typing()
    _setup_mpi4py_typing()
    _patch_domain_python()
    _setup_autodoc(app)

    try:
        from petsc4py import PETSc
    except ImportError:
        autodoc_mock_imports.append('PETSc')
        return
    del PETSc.DA  # FIXME

    sys_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    import apidoc
    sys.dont_write_bytecode = sys_dwb

    name = PETSc.__name__
    here = os.path.abspath(os.path.dirname(__file__))
    outdir = os.path.join(here, apidoc.OUTDIR)
    source = os.path.join(outdir, f'{name}.py')
    getmtime = os.path.getmtime
    generate = (
        not os.path.exists(source)
        or getmtime(source) < getmtime(PETSc.__file__)
        or getmtime(source) < getmtime(apidoc.__file__)
    )
    if generate:
        apidoc.generate(source)
    module = apidoc.load_module(source)
    apidoc.replace_module(module)

    modules = [
        'petsc4py',
    ]
    typing_overload = typing.overload
    typing.overload = lambda arg: arg
    for name in modules:
        mod = importlib.import_module(name)
        ann = apidoc.load_module(f'{mod.__file__}i', name)
        apidoc.annotate(mod, ann)
    typing.overload = typing_overload


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = f'{package}-man'


# -- Options for LaTeX output ---------------------------------------------

# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', f'{package}.tex', project, author, 'howto'),
]

latex_elements = {
    'papersize': 'a4',
}


# -- Options for manual page output ---------------------------------------

# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', package, project, [author], 3)
]


# -- Options for Texinfo output -------------------------------------------

# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', package, project, author,
     package, f'{project}.', 'Miscellaneous'),
]


# -- Options for Epub output ----------------------------------------------

# Output file base name for ePub builder.
epub_basename = package
