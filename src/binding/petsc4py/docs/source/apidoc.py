import os
import sys
import inspect
import textwrap


def is_cyfunction(obj):
    return type(obj).__name__ == 'cython_function_or_method'


def is_function(obj):
    return (
        inspect.isbuiltin(obj)
        or is_cyfunction(obj)
        or type(obj) is type(ord)
    )


def is_method(obj):
    return (
        inspect.ismethoddescriptor(obj)
        or inspect.ismethod(obj)
        or is_cyfunction(obj)
        or type(obj) in (
            type(str.index),
            type(str.__add__),
            type(str.__new__),
        )
    )


def is_classmethod(obj):
    return (
        inspect.isbuiltin(obj)
        or type(obj).__name__ in (
            'classmethod',
            'classmethod_descriptor',
        )
    )


def is_staticmethod(obj):
    return (
        type(obj).__name__ in (
            'staticmethod',
        )
    )

def is_constant(obj):
    return isinstance(obj, (int, float, str, dict))

def is_datadescr(obj):
    return inspect.isdatadescriptor(obj) and not hasattr(obj, 'fget')


def is_property(obj):
    return inspect.isdatadescriptor(obj) and hasattr(obj, 'fget')


def is_class(obj):
    return inspect.isclass(obj) or type(obj) is type(int)


class Lines(list):

    INDENT = " " * 4
    level = 0

    @property
    def add(self):
        return self

    @add.setter
    def add(self, lines):
        if lines is None:
            return
        if isinstance(lines, str):
            lines = textwrap.dedent(lines).strip().split('\n')
        indent = self.INDENT * self.level
        for line in lines:
            self.append(indent + line)


def signature(obj):
    doc = obj.__doc__
    doc = doc or f"{obj.__name__}: Any"  # FIXME remove line
    sig = doc.partition('\n')[0].split('.', 1)[-1]
    return sig or None


def docstring(obj):
    doc = obj.__doc__
    doc = doc or '' # FIXME
    if is_class(obj):
        doc = doc.strip()
    else:
        doc = doc.partition('\n')[2]
    summary, _, docbody = doc.partition('\n')
    summary = summary.strip()
    docbody = textwrap.dedent(docbody).strip()
    if docbody:
        doc = f'"""{summary}\n\n{docbody}\n\n"""'
    else:
        doc = f'"""{summary}"""'
    doc = textwrap.indent(doc, Lines.INDENT)
    return doc


def visit_data(constant):
    name, value = constant
    typename = type(value).__name__
    kind = "Constant" if isinstance(value, int) else "Object"
    init = f"_def({typename}, '{name}')"
    doc = f"#: {kind} ``{name}`` of type :class:`{typename}`"
    return f"{name}: {typename} = {init}  {doc}\n"


def visit_function(function):
    sig = signature(function)
    doc = docstring(function)
    body = Lines.INDENT + "..."
    return f"def {sig}:\n{doc}\n{body}\n"


def visit_method(method):
    sig = signature(method)
    doc = docstring(method)
    body = Lines.INDENT + "..."
    return f"def {sig}:\n{doc}\n{body}\n"


def visit_datadescr(datadescr, name=None):
    sig = signature(datadescr)
    doc = docstring(datadescr)
    name = sig.partition(':')[0].strip() or datadescr.__name__
    type = sig.partition(':')[2].strip() or 'Any'
    sig = f"{name}(self) -> {type}"
    body = Lines.INDENT + "..."
    return f"@property\ndef {sig}:\n{doc}\n{body}\n"


def visit_property(prop, name=None):
    sig = signature(prop.fget)
    name = name or prop.fget.__name__
    type = sig.rsplit('->', 1)[-1].strip()
    sig = f"{name}(self) -> {type}"
    doc = f'"""{prop.__doc__}"""'
    doc = textwrap.indent(doc, Lines.INDENT)
    body = Lines.INDENT + "..."
    return f"@property\ndef {sig}:\n{doc}\n{body}\n"


def visit_constructor(cls, name='__init__', args=None):
    init = (name == '__init__')
    argname = cls.__mro__[-2].__name__.lower()
    argtype = cls.__name__
    initarg = args or f"{argname}: Optional[{argtype}] = None"
    selfarg = 'self' if init else 'cls'
    rettype = 'None' if init else argtype
    arglist = f"{selfarg}, {initarg}"
    sig = f"{name}({arglist}) -> {rettype}"
    ret = '...' if init else 'return super().__new__(cls)'
    body = Lines.INDENT + ret
    return f"def {sig}:\n{body}"


def visit_class(cls, outer=None, done=None):
    skip = {
        '__doc__',
        '__dict__',
        '__module__',
        '__weakref__',
        '__pyx_vtable__',
        '__lt__',
        '__le__',
        '__ge__',
        '__gt__',
        '__enum2str',  # FIXME refactor implemetation
        '_traceback_', # FIXME maybe refactor?
    }
    special = {
        '__len__': "__len__(self) -> int",
        '__bool__': "__bool__(self) -> bool",
        '__hash__': "__hash__(self) -> int",
        '__int__': "__int__(self) -> int",
        '__index__': "__int__(self) -> int",
        '__str__': "__str__(self) -> str",
        '__repr__': "__repr__(self) -> str",
        '__eq__': "__eq__(self, other: object) -> bool",
        '__ne__': "__ne__(self, other: object) -> bool",
    }
    constructor = (
        '__new__',
        '__init__',
    )

    qualname = cls.__name__
    cls_name = cls.__name__
    if outer is not None and cls_name.startswith(outer):
        cls_name = cls_name[len(outer):]
        qualname = f"{outer}.{cls_name}"

    override = OVERRIDE.get(qualname, {})
    done = set() if done is None else done
    lines = Lines()

    base = cls.__base__
    if base is object:
        lines.add = f"class {cls_name}:"
    else:
        lines.add = f"class {cls_name}({base.__name__}):"
    lines.level += 1

    lines.add = docstring(cls)

    for name in ('__new__', '__init__', '__hash__'):
        if name in cls.__dict__:
            done.add(name)

    dct = cls.__dict__
    keys = list(dct.keys())

    def dunder(name):
        return name.startswith('__') and name.endswith('__')

    def members(seq):
        for name in seq:
            if name in skip:
                continue
            if name in done:
                continue
            if dunder(name):
                if name not in special and name not in override:
                    done.add(name)
                    continue
            yield name

    for name in members(keys):
        attr = getattr(cls, name)
        if is_class(attr):
            done.add(name)
            lines.add = visit_class(attr, outer=cls_name)
            continue

    for name in members(keys):

        if name in override:
            done.add(name)
            lines.add = override[name]
            continue

        if name in special:
            done.add(name)
            sig = special[name]
            lines.add = f"def {sig}: ..."
            continue

        attr = getattr(cls, name)

        if is_method(attr):
            done.add(name)
            if name == attr.__name__:
                obj = dct[name]
                if is_classmethod(obj):
                    lines.add = "@classmethod"
                elif is_staticmethod(obj):
                    lines.add = "@staticmethod"
                lines.add = visit_method(attr)
            elif False:
                lines.add = f"{name} = {attr.__name__}"
            continue

        if is_datadescr(attr):
            done.add(name)
            lines.add = visit_datadescr(attr)
            continue

        if is_property(attr):
            done.add(name)
            lines.add = visit_property(attr, name)
            continue

        if is_constant(attr):
            done.add(name)
            lines.add = visit_data((name, attr))
            continue

    leftovers = [name for name in keys if
                 name not in done and name not in skip]
    if leftovers:
        raise RuntimeError(f"leftovers: {leftovers}")

    lines.level -= 1
    return lines


def visit_module(module, done=None):
    skip = {
        '__doc__',
        '__name__',
        '__loader__',
        '__spec__',
        '__file__',
        '__package__',
        '__builtins__',
        '__pyx_capi__',
        '__pyx_unpickle_Enum',  # FIXME review
    }

    done = set() if done is None else done
    lines = Lines()

    keys = list(module.__dict__.keys())
    keys.sort(key=lambda name: name.startswith("_"))

    constants = [
        (name, getattr(module, name)) for name in keys
        if all((
            name not in done and name not in skip,
            is_constant(getattr(module, name)),
        ))
    ]
    for _, value in constants:
        cls = type(value)
        name = cls.__name__
        if name in done or name in skip:
            continue
        if cls.__module__ == module.__name__:
            done.add(name)
            lines.add = visit_class(cls)
            lines.add = ""
    for attr in constants:
        name, value = attr
        done.add(name)
        if name in OVERRIDE:
            lines.add = OVERRIDE[name]
        else:
            lines.add = visit_data((name, value))
    if constants:
        lines.add = ""

    for name in keys:
        if name in done or name in skip:
            continue
        value = getattr(module, name)

        if is_class(value):
            done.add(name)
            if value.__name__ != name:
                continue
            if value.__module__ != module.__name__:
                continue
            lines.add = visit_class(value)
            lines.add = ""
            instances = [
                (k, getattr(module, k)) for k in keys
                if all((
                    k not in done and k not in skip,
                    type(getattr(module, k)) is value,
                ))
            ]
            for attrname, attrvalue in instances:
                done.add(attrname)
                lines.add = visit_data((attrname, attrvalue))
            if instances:
                lines.add = ""
            continue

        if is_function(value):
            done.add(name)
            if name == value.__name__:
                lines.add = visit_function(value)
            else:
                lines.add = f"{name} = {value.__name__}"
            continue

    lines.add = ""
    for name in keys:
        if name in done or name in skip:
            continue
        value = getattr(module, name)
        done.add(name)
        if name in OVERRIDE:
            lines.add = OVERRIDE[name]
        else:
            lines.add = visit_data((name, value))

    leftovers = [name for name in keys if
                 name not in done and name not in skip]
    if leftovers:
        raise RuntimeError(f"leftovers: {leftovers}")
    return lines


IMPORTS = """
from __future__ import annotations
import sys
from typing import (
    Any,
    Union,
    Literal,
    Optional,
    NoReturn,
    Final,
)
from typing import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
    Mapping,
)
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy
from numpy import dtype, ndarray
from mpi4py.MPI import (
    Intracomm,
    Datatype,
    Op,
)

class _dtype:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

IntType: dtype = _dtype('IntType')
RealType: dtype =  _dtype('RealType')
ComplexType: dtype = _dtype('ComplexType')
ScalarType: dtype = _dtype('ScalarType')
"""

HELPERS = """
class _Int(int): pass
class _Str(str): pass
class _Float(float): pass
class _Dict(dict): pass

def _repr(obj):
    try:
        return obj._name
    except AttributeError:
        return super(obj).__repr__()

def _def(cls, name):
    if cls is int:
       cls = _Int
    if cls is str:
       cls = _Str
    if cls is float:
       cls = _Float
    if cls is dict:
       cls = _Dict

    obj = cls()
    obj._name = name
    if '__repr__' not in cls.__dict__:
        cls.__repr__ = _repr
    return obj
"""

OVERRIDE = {
}

TYPING = """
from .typing import *
"""


def visit_petsc4py_PETSc(done=None):
    from petsc4py import PETSc
    lines = Lines()
    lines.add = f'"""{PETSc.__doc__}"""'
    lines.add = IMPORTS
    lines.add = ""
    lines.add = HELPERS
    lines.add = ""
    lines.add = visit_module(PETSc)
    lines.add = ""
    lines.add = TYPING
    return lines


def generate(filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'w') as f:
        for line in visit_petsc4py_PETSc():
            print(line, file=f)


def load_module(filename, name=None):
    if name is None:
        name, _ = os.path.splitext(
            os.path.basename(filename))
    module = type(sys)(name)
    module.__file__ = filename
    module.__package__ = name.rsplit('.', 1)[0]
    old = replace_module(module)
    with open(filename) as f:
        exec(f.read(), module.__dict__)  # noqa: S102
    restore_module(old)
    return module


_sys_modules = {}


def replace_module(module):
    name = module.__name__
    assert name not in _sys_modules
    _sys_modules[name] = sys.modules[name]
    sys.modules[name] = module
    return _sys_modules[name]


def restore_module(module):
    name = module.__name__
    assert name in _sys_modules
    sys.modules[name] = _sys_modules[name]
    del _sys_modules[name]


def annotate(dest, source):
    try:
        dest.__annotations__ = source.__annotations__
    except AttributeError:
        pass
    if isinstance(dest, type):
        for name in dest.__dict__.keys():
            if hasattr(source, name):
                obj = getattr(dest, name)
                annotate(obj, getattr(source, name))
    if isinstance(dest, type(sys)):
        for name in dir(dest):
            if hasattr(source, name):
                obj = getattr(dest, name)
                mod = getattr(obj, '__module__', None)
                if dest.__name__ == mod:
                    annotate(obj, getattr(source, name))
        for name in dir(source):
            if not hasattr(dest, name):
                setattr(dest, name, getattr(source, name))


OUTDIR = 'reference'

if __name__ == '__main__':
    generate(os.path.join(OUTDIR, 'petsc4py.PETSc.py'))
