# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import re
import os
import shutil
import sys
import subprocess
import typing
import datetime
import importlib
import sphobjinv
import functools
import pylit
from sphinx import __version__ as sphinx_version
from sphinx.ext.napoleon.docstring import NumpyDocstring
from packaging.version import Version

sys.path.insert(0, os.path.abspath('.'))
_today = datetime.datetime.now()

# FIXME: allow building from build?

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

package = 'petsc4py'
project = 'petsc4py'   # shown in top left corner of the petsc4py documentation

docdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(docdir, *[os.path.pardir] * 2))


def pkg_version():
    with open(os.path.join(topdir, 'src', package, '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]


def get_doc_branch():
    release = 1
    if topdir.endswith(os.path.join(os.path.sep, 'src', 'binding', package)):
        rootdir = os.path.abspath(os.path.join(topdir, *[os.path.pardir] * 3))
        rootname = package.replace('4py', '')
        version_h = os.path.join(rootdir, 'include', f'{rootname}version.h')
        if os.path.exists(version_h) and os.path.isfile(version_h):
            release_macro = f'{rootname.upper()}_VERSION_RELEASE'
            version_re = re.compile(rf'#define\s+{release_macro}\s+([-]*\d+)')
            with open(version_h, 'r') as f:
                release = int(version_re.search(f.read()).groups()[0])
    return 'release' if release else 'main'


__project__ = 'PETSc for Python'
__author__ = 'Lisandro Dalcin'
__copyright__ = f'{_today.year}, {__author__}'

release = pkg_version()
version = release.rsplit('.', 1)[0]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.extlinks',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

default_role = 'any'

pygments_style = 'tango'

nitpicky = True
nitpick_ignore = [
    ('envvar', 'NUMPY_INCLUDE'),
    ('py:class', 'ndarray'),  # FIXME
    ('py:class', 'typing_extensions.Self'),
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
autodoc_type_aliases = {}

autosummary_context = {
    'synopsis': {},
    'autotype': {},
}

suppress_warnings = []
if Version(sphinx_version) >= Version(
    '7.4'
):  # https://github.com/sphinx-doc/sphinx/issues/12589
    suppress_warnings.append('autosummary.import_cycle')

# Links depends on the actual branch -> release or main
www = f'https://gitlab.com/petsc/petsc/-/tree/{get_doc_branch()}'
extlinks = {'sources': (f'{www}/src/binding/petsc4py/src/%s', '%s')}

napoleon_preprocess_types = True

try:
    import sphinx_rtd_theme

    if 'sphinx_rtd_theme' not in extensions:
        extensions.append('sphinx_rtd_theme')
except ImportError:
    sphinx_rtd_theme = None

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest/', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
    'pyopencl': ('https://documen.tician.de/pyopencl/', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest/', None),
    'petsc': ('https://petsc.org/release/', None),
}


def _mangle_petsc_intersphinx():
    """Preprocess the keys in PETSc's intersphinx inventory.

    PETSc have intersphinx keys of the form:

        manualpages/Vec/VecShift

    instead of:

        petsc.VecShift

    This function downloads their object inventory and strips the leading path
    elements so that references to PETSc names actually resolve."""

    website = intersphinx_mapping['petsc'][0].partition('/release/')[0]
    branch = get_doc_branch()
    doc_url = f'{website}/{branch}/'
    if 'LOC' in os.environ and os.path.isfile(
        os.path.join(os.environ['LOC'], 'objects.inv')
    ):
        inventory_url = 'file://' + os.path.join(os.environ['LOC'], 'objects.inv')
    else:
        inventory_url = f'{doc_url}objects.inv'
    print('Using PETSC inventory from ' + inventory_url)
    inventory = sphobjinv.Inventory(url=inventory_url)
    print(inventory)

    for obj in inventory.objects:
        if obj.name.startswith('manualpages'):
            obj.name = 'petsc.' + '/'.join(obj.name.split('/')[2:])
            obj.role = 'class'
            obj.domain = 'py'

    new_inventory_filename = 'petsc_objects.inv'
    sphobjinv.writebytes(
        new_inventory_filename, sphobjinv.compress(inventory.data_file(contract=True))
    )
    intersphinx_mapping['petsc'] = (doc_url, new_inventory_filename)


_mangle_petsc_intersphinx()


def _setup_mpi4py_typing():
    pkg = type(sys)('mpi4py')
    mod = type(sys)('mpi4py.MPI')
    mod.__package__ = pkg.__name__
    sys.modules[pkg.__name__] = pkg
    sys.modules[mod.__name__] = mod
    for clsname in (
        'Intracomm',
        'Datatype',
        'Op',
    ):
        cls = type(clsname, (), {})
        cls.__module__ = mod.__name__
        setattr(mod, clsname, cls)


def _patch_domain_python():
    from sphinx.domains.python import PythonDomain

    PythonDomain.object_types['data'].roles += ('class',)


def _setup_autodoc(app):
    from sphinx.ext import autodoc
    from sphinx.util import inspect
    from sphinx.util import typing

    #

    def stringify_annotation(annotation, *p, **kw):
        qualname = getattr(annotation, '__qualname__', '')
        module = getattr(annotation, '__module__', '')
        args = getattr(annotation, '__args__', None)
        if module == 'builtins' and qualname and args is not None:
            args = ', '.join(stringify_annotation(a, *p, **kw) for a in args)
            return f'{qualname}[{args}]'
        return stringify_annotation_orig(annotation, *p, **kw)

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

    inspect.TypeAliasForwardRef.__repr__ = lambda self: self.name

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


def _monkey_patch_returns():
    """Rewrite the role of names in "Returns" sections.

    This is needed because Napoleon uses ``:class:`` for the return types
    and this does not work with type aliases like ``ArrayScalar``. To resolve
    this we swap ``:class:`` for ``:any:``.

    """
    _parse_returns_section = NumpyDocstring._parse_returns_section

    @functools.wraps(NumpyDocstring._parse_returns_section)
    def wrapper(*args, **kwargs):
        out = _parse_returns_section(*args, **kwargs)
        for role in (':py:class:', ':class:'):
            out = [line.replace(role, ':any:') for line in out]
        return out

    NumpyDocstring._parse_returns_section = wrapper


def _monkey_patch_see_also():
    """Rewrite the role of names in "see also" sections.

    Napoleon uses :obj: for all names found in "see also" sections but we
    need :all: so that references to labels work."""

    _parse_numpydoc_see_also_section = NumpyDocstring._parse_numpydoc_see_also_section

    @functools.wraps(NumpyDocstring._parse_numpydoc_see_also_section)
    def wrapper(*args, **kwargs):
        out = _parse_numpydoc_see_also_section(*args, **kwargs)
        for role in (':py:obj:', ':obj:'):
            out = [line.replace(role, ':any:') for line in out]
        return out

    NumpyDocstring._parse_numpydoc_see_also_section = wrapper


def _apply_monkey_patches():
    """Modify Napoleon types after parsing to make references work."""
    _monkey_patch_returns()
    _monkey_patch_see_also()


_apply_monkey_patches()


def _process_demos(*demos):
    # Convert demo .py files to rst. Also copy the .py file so it can be
    # linked from the demo rst file.
    try:
        os.mkdir('demo')
    except FileExistsError:
        pass
    for demo in demos:
        demo_dir = os.path.join('demo', os.path.dirname(demo))
        demo_src = os.path.join(os.pardir, os.pardir, 'demo', demo)
        try:
            os.mkdir(demo_dir)
        except FileExistsError:
            pass
        with open(demo_src, 'r') as infile:
            with open(
                os.path.join(os.path.join('demo', os.path.splitext(demo)[0] + '.rst')),
                'w',
            ) as outfile:
                converter = pylit.Code2Text(infile)
                outfile.write(str(converter))
        demo_copy_name = os.path.join(demo_dir, os.path.basename(demo))
        shutil.copyfile(demo_src, demo_copy_name)
        html_static_path.append(demo_copy_name)
    with open(os.path.join('demo', 'demo.rst'), 'w') as demofile:
        demofile.write("""
petsc4py demos
==============

.. toctree::

""")
        for demo in demos:
            demofile.write('    ' + os.path.splitext(demo)[0] + '\n')
        demofile.write('\n')


html_static_path = []
_process_demos('poisson2d/poisson2d.py')


def setup(app):
    _setup_mpi4py_typing()
    _patch_domain_python()
    _monkey_patch_returns()
    _monkey_patch_see_also()
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

    from petsc4py import typing as tp

    for attr in tp.__all__:
        autodoc_type_aliases[attr] = f'~petsc4py.typing.{attr}'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'navigation_with_keys': True,
    'footer_end': ['theme-version', 'last-updated'],
}
git_describe_version = (
    subprocess.check_output(['git', 'describe', '--always']).strip().decode('utf-8')  # noqa: S603, S607
)
html_last_updated_fmt = r'%Y-%m-%dT%H:%M:%S%z (' + git_describe_version + ')'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = f'{package}-man'


# -- Options for LaTeX output ---------------------------------------------

# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', f'{package}.tex', __project__, __author__, 'howto'),
]

latex_elements = {
    'papersize': 'a4',
}


# -- Options for manual page output ---------------------------------------

# (source start file, name, description, authors, manual section).
man_pages = [('index', package, __project__, [__author__], 3)]


# -- Options for Texinfo output -------------------------------------------

# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        'index',
        package,
        __project__,
        __author__,
        package,
        f'{__project__}.',
        'Miscellaneous',
    ),
]


# -- Options for Epub output ----------------------------------------------

# Output file base name for ePub builder.
epub_basename = package
