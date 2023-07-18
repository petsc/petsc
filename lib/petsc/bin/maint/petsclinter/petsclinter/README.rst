PETSc Clang Linter - *Failing your pipelines since 2021!*
*********************************************************

This directory contains the meat and potatoes of the clang linter. At a very high level,
the linter walks through the abstract syntax tree (AST) of a parsed file looking for nodes
(cursor) of specific kinds, e.g. function calls, type definitions, function definitions,
etc. Once it arrives at a cursor of matching type, it checks whether it has a handler
configured for that cursor. If it does, it will call the handler (which may attach some
number of diagnostics). If there is no handler, nothing happens.

Once the linter has finished processing a file (by reaching the end of the AST), the
caller can extract the generated diagnostics and do something with them.

Caveats
=======

1. A byproduct of using libclang to generate the AST is that **the linter can only process
what clang can compile**. For this reason, the CI run of the linter needs to have almost
every package installed to fully lint the library.

Directories
===========

- checks

  Contains all of the concrete checkers to determine whether or not a particular construct
  is correct. Also contains the registration mechanism to let external packages (at this
  point only SLEPc) register additional checkers. See `Checkers` for more information on
  how to make a new checker.

- classes

  Contains all of the classes used by the linting process. Generally speaking, not
  something that external users need to see, but rather utility or wrappers to provide
  some functionality.

- classes/docs

  Contains specifically the classes to lint docstrings.

- util

  General utility functions that don't fit anywhere else. For example, there are a bunch
  of clang type helpers, or functions that build a set of compiler flags etc.


Organization
============

Generally speaking, the hierarchy is checks -> classes -> util, where a -> indicates an
import dependency (for example, checks imports stuff from classes). Therefore, it is
fairly vital that imports **do not** go backwards, otherwise you will introduce a bunch of
circular import errors that are a pain in the neck to untangle.

What Kind Of Things Should The Linter Be Checking?
--------------------------------------------------

First and foremost, the linter is there to check things which a compiler cannot. If it is
possible for you to express your constraint in terms of normal C/C++ code, **then do that
instead**. Only when you simply cannot do it, should you consider adding checks to the
linter. For example, consider the ubiquitous ``PetscAssertPointer()`` which is used

::

   PetscErrorCode Foo(PetscInt *i)
   {
     ...
     PetscAssertPointer(i, 1);


Note the second parameter, which indicates the argument number in the parent function. We
can't really enforce the fact that ``idx_num`` must be ``1`` via e.g. the type system. A
naive next solution might be to parse ``__func__`` and try and find ``i`` in it. But this
falls apart as soon as ``i`` is being any kind of indirection. Perhaps we have

::

   PetscErrorCode Foo(MyStruct *str)
   {
     int *foo = str->foo;
     ...
     PetscAssertPointer(foo, 1);


where ``str->foo`` is also a pointer. Since this pointer "originates" from ``str`` it is
correct for the argument number to match ``str``s (i.e. ``1``). But a naive parsing of
``__func__`` would never catch this.

This is a prime example of something that cannot be checked in the source alone, and
requires a higher-level semantic view of the function to reliably check (in fact, it was
_the_ motivating case for the creation of the linter!).

Checkers
--------

How do you register a new checker? I.e. given

::

   #define MyCheckingMacro(some_obj, some_value, idx_num) ...

1. Provide a non-macro stub for your function, that is guarded by
   ``PETSC_CLANG_STATIC_ANALYZER``. You do not need to provide a definition anywhere in
   the code, and it not necessarily match type-for-type exactly with what you
   expect. All that matters is that when the linter sees the code, it sees a function not
   a macro.

   ::

      #if defined(PETSC_CLANG_STATIC_ANALYZER)
      // the linter is parsing the code
      template <typename T, typename U>
      void MyCheckingMacro(T, U, int);
      #else
      #define MyCheckingMacro(some_obj, some_value, idx_num) ...
      #endif


2. Define a checking function under ``checks/_code.py``. The checking function must take 3
   arguments and return ``None``:

   ::

      def checkMyCheckingMacro(linter: Linter, func: Cursor, parent: Cursor) -> None:
        # ...
        return

   - linter: The linter instance. Any errors created should be logged with it
   - func: This will be a cursor corresponding to a detected call to ``MyCheckingMacro()``
   - parent: This will be the cursor corresponding to the "parent" function of ``MyCheckingMacro()``

   I.e. given

   ::

      PetscErrorCode PetscFooFunction(SomeObj a, ...)
      {              ~~~~~~ 1 ~~~~~~~
        ...
        MyCheckingMacro(a, SOME_VALUE, 1);
        ~~~~~ 2 ~~~~~~~


   Then ``parent`` will be the cursor for ``(1)``, and ``func`` will be the cursor for
   ``(2)``. You should implement all your checks for this instance entirely within this
   function, it is called only once for each found instance of a function call.

   Usually, you will not need to write a new checking function, as the vast majority of
   PETSc macros are simple and follow the same basic structure. See the other functions
   for examples of pre-existing checkers. If you need to write a new one, you will have to
   plumb through checks/_util.py for examples.

3. The final step is to add your checker to the set of registered checks. In
   ``checks/_register.py::__register_all_symbol_checks()`` you will find a dict mapping
   names to python checker functions. You must add your check to this dict.

   ::

      default_checks = {
        "PetscValidHeaderSpecificType"       : _code.checkPetscValidHeaderSpecificType,
        "PetscValidHeaderSpecific"           : _code.checkPetscValidHeaderSpecific,
        "PetscValidHeader"                   : _code.checkPetscValidHeader,
        ...
        "MyCheckingMacro"                    : _coda.checkMyCheckingMacro


   Take extra care that "MyCheckingMacro" exactly matches (including case) the name of the
   macro. The keys to this dictionary is ultimately what the linter uses to determine
   whether A. to check a call at all, and B. which function to check with.
