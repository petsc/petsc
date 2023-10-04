#!/usr/bin/env python3
"""
# Created: Thu Jul 27 13:53:56 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import sys

from typing import (
  TYPE_CHECKING,
  Union, Optional, TypedDict, TypeVar, NamedTuple, Generic, Protocol,
  overload, cast as TYPE_CAST
)

from .__version__ import py_version_lt

py_version_lt(3, 9)
# Dummy function call conditional to make sure you remember to remove the following line
from typing import List, Dict, Tuple, Type

py_version_lt(3, 10)
# Also TypeAlias below
if sys.version_info >= (3, 10):
  # novermin
  from typing import ParamSpec
else:
  from typing_extensions import ParamSpec

if TYPE_CHECKING:
  import re
  import pathlib
  import weakref
  import clang.cindex as clx # type: ignore[import]

  from typing import Any, NoReturn, ClassVar, SupportsInt

  if sys.version_info >= (3, 10):
    # novermin
    from typing import TypeAlias
  else:
    from typing_extensions import TypeAlias

  from collections.abc import (
    Iterator, Iterable, Generator, Callable, Collection, Sequence, Container, Mapping
  )

  ##
  # CLASSES
  ##

  from .classes._cursor     import Cursor
  from .classes._linter     import Linter, WeakList
  from .classes._patch      import Patch
  from .classes._path       import Path
  from .classes._pool       import WorkerPoolBase, ParallelPool, SerialPool
  from .classes._src_pos    import SourceLocation, SourceRange
  from .classes._attr_cache import AttributeCache
  from .classes._scope      import Scope
  from .classes._weak_list  import WeakList
  from .classes._add_line   import Addline
  from .classes._diag       import (
    DiagnosticMapProxy, DiagnosticMap, DiagnosticsManagerCls, DiagnosticKind, Diagnostic
  )

  PathLike: TypeAlias           = Union[pathlib.Path, Path]
  StrPathLike: TypeAlias        = Union[PathLike, str]
  CursorLike: TypeAlias         = Union[clx.Cursor, Cursor]
  SourceLocationLike: TypeAlias = Union[clx.SourceLocation, SourceLocation]
  SourceRangeLike: TypeAlias    = Union[clx.SourceRange, SourceRange]
  PoolImpl                      = TypeVar('PoolImpl', bound=WorkerPoolBase)
  PathDiffPair: TypeAlias       = Tuple[Path, str]
  CondensedDiags: TypeAlias     = Dict[Path, List[str]]
  WeakListType: TypeAlias       = WeakList[Tuple[str, bool, int]]
  WeakListRef: TypeAlias        = weakref.ReferenceType[WeakListType]

  class Formattable(Protocol):
    # Want to do the following,
    #
    # def formatted(self, num_before_context: int = 0, num_after_context: int = 0, num_context: int = 0, view: bool = False, highlight: bool = True, trim: bool = True) -> str:
    #   ...
    #
    # def raw(self, num_before_context: int = 0, num_after_context: int = 0, num_context: int = 0, trim: bool = True, tight: bool = False) -> str:
    #   ...
    #
    # but then mypy complains
    #
    # error: Argument "crange" to "make_diagnostic_message" of
    # "PetscDocString" has incompatible type "Cursor"; expected "Optional[Formattable]"  [arg-type]
    #               'Parameter list defined here', crange=docstring.cursor
    #                                                     ^~~~~~~~~~~~~~~~
    # note: Following member(s) of "Cursor" have conflicts:
    # note:     Expected:
    # note:         def formatted(self, num_before_context: int = ..., num_after_context: int = ..., num_context: int = ..., view: bool = ..., highlight: bool = ..., trim: bool = ...) -> str
    # note:     Got:
    # note:         def formatted(self, **kwargs: Any) -> str
    # note:     Expected:
    # note:         def raw(self, num_before_context: int = ..., num_after_context: int = ..., num_context: int = ..., trim: bool = ..., tight: bool = ...) -> str
    # note:     Got:
    # note:         def raw(self, **kwargs: Any) -> str
    #
    # So instead we find ourselves reduced to this...
    def formatted(self, **kwargs: Any) -> str:
      ...

    def raw(self, **kwargs: Any) -> str:
      ...

  ##
  # DOCS
  ##

  from .classes.docs._doc_str          import Verdict, PetscDocString
  from .classes.docs._doc_section_base import (
    DescribableItem, SectionBase, Synopsis, ParameterList, Prose, VerbatimBlock, InlineList,
  )
  from .classes.docs._doc_section      import (
    FunctionSynopsis, EnumSynopsis, FunctionParameterList, OptionDatabaseKeys, Notes, DeveloperNotes,
    References, FortranNotes, SourceCode, Level, SeeAlso
  )

  SectionImpl        = TypeVar('SectionImpl', bound=SectionBase)
  PetscDocStringImpl = TypeVar('PetscDocStringImpl', bound=PetscDocString)
  SynopsisImpl       = TypeVar('SynopsisImpl', bound=Synopsis)

  ##
  # UTIL
  ##

  from .util._clang   import CXTranslationUnit, ClangFunction
  from .util._color   import Color
  from .util._utility import PrecompiledHeader

  ExceptionKind = TypeVar('ExceptionKind', bound=Exception)

  ##
  # CHECKS
  ##

  FunctionChecker: TypeAlias = Callable[[Linter, Cursor, Cursor], None]
  DocChecker: TypeAlias      = Callable[[Linter, Cursor], None]

del py_version_lt
