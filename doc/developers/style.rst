PETSc Style and Usage Guide
===========================

The PETSc team uses certain conventions to make the source code
consistent and hence easier to maintain. We will interchangeably use the
terminology *subclass*, *implementation*, or *type* [1]_ to refer to a
concrete realization of an abstract base class. For example,
``KSPGMRES`` is a type for the base class ``KSP``.

Names
-----

Consistency of names for variables, functions, and so on is extremely
important. We use several conventions

#. All function names and enum types consist of acronyms or words, each
   of which is capitalized, for example, ``KSPSolve()`` and
   ``MatGetOrdering()``.

#. All enum elements and macro variables are named with all capital
   letters. When they consist of several complete words, there is an
   underscore between each word. For example, ``MAT_FINAL_ASSEMBLY``.

#. Functions that are private to PETSc (not callable by the application
   code) either

   -  have an appended ``_Private`` (for example, ``StashValues_Private``)
      or

   -  have an appended ``_Subtype`` (for example, ``MatMultSeq_AIJ``).

   In addition, functions that are not intended for use outside of a
   particular file are declared ``static``. Also see item
   on symbol visibility in :ref:`usage_of_petsc_functions_and_macros`.

#. Function names in structures (for example, ``_matops``) are the same
   as the base application function name without the object prefix and
   are in small letters. For example, ``MatMultTranspose()`` has a
   structure name of ``multtranspose``.

#. Names of implementations of class functions should begin with the
   function name, an underscore, and the name of the implementation, for
   example, ``KSPSolve_GMRES()``.

#. Each application-usable function begins with the name of the class
   object, followed by any subclass name, for example,
   ``ISInvertPermutation()``, ``MatMult()``, or
   ``KSPGMRESSetRestart()``.

#. Functions that PETSc provides as defaults for user-providable
   functions end with ``Default`` (for example, ``PetscSignalHandlerDefault()``).

#. Options database keys are lower case, have an underscore between
   words, and match the function name associated with the option without
   the word “set” or “get”, for example, ``-ksp_gmres_restart``.

#. Specific ``XXXType`` values (for example, ``MATSEQAIJ``) do not have
   an underscore in them unless they refer to another package that uses
   an underscore, for example, ``MATSOLVERSUPERLU_DIST``.

Coding Conventions and Style
----------------------------

Within the PETSc source code, we adhere to the following guidelines so
that the code is uniform and easily maintained.

C Formatting
~~~~~~~~~~~~

The white space and basic layout is controlled by the ``.clang-format`` file in the root directory. You can run the formatter in a single directory or in the entire repository with ``make clangformat``. All merge requests must be properly formatted; this is automatically checked for merge requests with ``make checkclangformat``.

Even with the use of ``clang-format`` there are still many decisions about code formatting that must be constantly made. A subset of these are automatically checked for merge requests with ``make checkbadSource``.

#. The prototypes for functions should not include the names of the
   variables

   ::

       PETSC_EXTERN PetscErrorCode MyFunction(PetscInt); // Correct
       PETSC_EXTERN PetscErrorCode MyFunction(PetscInt myvalue); // Incorrect

#. All local variables of a particular type (for example, ``PetscInt``) should be listed
   on the same line if possible; otherwise, they should be listed on adjacent lines. Note
   that pointers of different arity are considered to be different types. ``clang-format`` automatically
   handles the indenting shown below.

   ::

      // Correct
      PetscInt   a,b,c;
      PetscInt  *d,*e;
      PetscInt **f;

      // Incorrect
      PetscInt a,b,c,*d,*e,**f;

#. Local variables should be initialized when declared when possible

   ::

      // Correct
      PetscInt a,b,c = 11;

      // Incorrect
      PetscInt a,b,c;

      PetscFunctionBegin;
      c = 11;

#. All PETSc subroutine code blocks *must* start with a single blank line between the local variable
   declarations followed by ``PetscFunctionBegin``.

   ::

      // Correct
      PetscInt x;

      PetscFunctionBegin;

      // Incorrect
      PetscInt x;
      PetscFunctionBegin;

      // Incorrect
      PetscInt x;
      y = 11;

#. Functions in PETSc examples including ``main()`` should have  ``PetscFunctionBeginUser`` as the very first line after the local variable declarations.

#. PETSc functions should almost always return with ``PetscFunctionReturn(PETSC_SUCCESS)`` not ``return 0``. ``PetscFunctionReturn()`` should never return a non-zero integer value.
   It may, on occasion, return ``MPI_SUCCESS``, NULL, or a ``PetscObject`` such as a ``PetscViewer`` depending on the return type of the function.

#. Functions that do use return should use ``return xx;`` rather than ``return(xx);``

#. All PETSc functions must have their return value checked for errors using the
   ``PetscCall()`` macro. This should be wrapped around the function in question.

   ::

      PetscCall(MyFunction(...)); // Correct
      PetscErrorCode ierr = MyFunction(...);PetscCall(ierr); // Incorrect

   The only exceptions to this rule are begin-end style macros which embed local variables
   or loops as part of their expansion
   (e.g. ``PetscOptionsBegin()``/``PetscOptionsEnd()``).  These handle errors internally
   and do not need error checking.

   ::

      // Correct
      PetscOptionsBegin(...);
      PetscOptionsEnd();


   As a rule, always try to wrap the function first, if this fails to compile you do
   not need to add the error checking.

   Calls to external package functions are generally made with ``PetscCallExternal()`` or its variants that are specialized for particular packages, for example ``PetscCallBLAS()``

#. Single operation ``if`` and ``else`` commands should not be wrapped in braces. They should be done as follows,

   ::

       if ( ) XXXX;
       else YYY;

   not

   ::

       if ( ) {XXXX;}
       else {YYY;}

#. Do not leave sections of commented-out code or dead source code protected with ``ifdef foo`` in the source files.

#. Use classic block comments (``/* There must be a space before the first word in the comment and a space at the end */``,
   (``/*Do not do this*/``) for multi-line comments, and ``// Comment`` for single-line comments in source files.

#. Do not put a ``*`` at the beginning or end of each line of a multi-line comment.

#. Do not use ``/* ---- ... ----- */`` or similar constructs to separate parts of source code files.

#. Use appropriate grammar and spelling in the comments.

#. All variables must be declared at the beginning of the code block (C89
   style), never mixed in with code. When variables are only used in a limited
   scope, it is encouraged to declare them in that scope. For example:

   ::

       if (cond) {
         PetscScalar *tmp;

         PetscCall(PetscMalloc1(10,&tmp));
         // use tmp
         PetscCall(PetscFree(tmp));
       }

   The only exception to this variables used exclusively within a ``for`` loop, which must
   be declared inside the loop initializer:

   ::

       // Correct
       for (PetscInt i=0; i<n; ++i) {
         // loop body
       }

   ::

       // Correct, variable used outside of loop
       PetscInt i;

   ::

       for (i=0; i<n; ++i) {
         // loop body
       }
       j = i;

   ::

       // Incorrect
       PetscInt i;
       ...
       for (i=0; i<n; ++i) {
         // loop body
       }

#. Developers are allowed to use // to split very long lines when it improves code readability. For example

   ::

       f[j][i].omega = xdot[j][i].omega + uxx + uyy //
                     + (vxp * (u - x[j][i - 1].omega) + vxm * (x[j][i + 1].omega - u)) * hy //
                     + (vyp * (u - x[j - 1][i].omega) + vym * (x[j + 1][i].omega - u)) * hx //
                     - .5 * grashof * (x[j][i + 1].temp - x[j][i - 1].temp) * hy;

#. The use of clang-format off is allowed in the source code but should only be used when absolutely necessary. It should not
   be used when trailing // to split lines works.

   ::

       // clang-format off
       f ...
       // clang-format on

#. `size` and `rank` should be used exclusively for the results of `MPI_Comm_size()` and `MPI_Comm_rank()` and other variable names for these values should be avoided unless necessary.

C Usage
~~~~~~~

#. Do not use language features that are not in the intersection of C99, C++11, and MSVC
   v1900+ (Visual Studio 2015).  Examples of such features include variable-length arrays.
   Note that variable-length arrays (including VLA-pointers) are not supported in C++ and
   were made optional in C11. You may use designated initializers via the
   ``PetscDesignatedInitializer()`` macro.

#. Array and pointer arguments where the array values are not changed
   should be labeled as ``const`` arguments.

#. Scalar values passed to functions should *never* be labeled as
   ``const``.

#. Subroutines that would normally have a ``void**`` argument to return
   a pointer to some data should be prototyped as ``void*``.
   This prevents the caller from having to put a ``(void**)`` cast in
   each function call. See, for example, ``DMDAVecGetArray()``.

#. Do not use the ``register`` directive.

#. Use ``if (v == NULL)`` or  ``if (flg == PETSC_TRUE)``, instead of using ``if (!v)`` or ``if (flg)`` or ``if (!flg)``.

#. Avoid ``#ifdef`` or ``#ifndef`` when possible. Rather, use ``#if defined(...`` or ``#if
   !defined(...``.  Better, use ``PetscDefined()`` (see below). The only exception to this
   rule is for header guards, where the ``#ifndef`` form is preferred (see below).

#. Header guard macros should include the full name and end in ``_FILE_EXTENSION`` of the
   file and be formed using ``#ifndef``. For example:

     // my_petsc_header_file.h
     #ifndef MY_PETSC_HEADER_FILE_H
     #define MY_PETSC_HEADER_FILE_H

     #endif // MY_PETSC_HEADER_FILE_H

#. Never use system random number generators such as ``rand()`` in PETSc
   code or examples because these can produce different results on
   different systems thus making portability testing difficult. Instead
   use ``PetscRandom`` which produces the exact same results regardless
   of system it is used on.

#. Never use system math functions or string functions in PETSc source code or examples, instead use the PETSc versions.

#. Variadic macros may be used in PETSc, but must work with MSVC v1900+ (Visual Studio
   2015). Most compilers have conforming implementations of the C99/C++11 rules for
   ``__VA_ARGS__``, but MSVC's implementation is not conforming and may need workarounds.
   See ``PetscDefined()`` for an example of how to work around MSVC's limitations to write
   a macro that is usable in both.

.. _usage_of_petsc_functions_and_macros:

Usage of PETSc Functions and Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Lengthy conditional preprocessor blocks should mark any ``#else`` or ``#endif``
   directives with a comment containing (or explaining) either the boolean condition or
   the name of the macro if the first directive is testing whether one is defined. One
   should be able to read any part of the macro block and be able to find or deduce the
   initial ``#if``. That is:

   ::

       #if defined(MY_MACRO)
       // many lines of code
       #else // MY_MACRO (use name of macro)
       // many more lines of code
       #endif // MY_MACRO

       #if MY_MACRO > 10
       // code
       #else // MY_MACRO < 10
       // more code
       #endif // MY_MACRO > 10

#. Nested preprocessor blocks should be indent the text (*not* the ``#``) following the
   normal indentation rules outlined above. For example:

   ::

       // Right
       #if MY_VARIABLE > 10
       #  if MY_OTHER_VARIABLE > 15
       #    define BIG_VARIABLE 1


       // Wrong
       #if MY_VARIABLE > 10
         #if MY_OTHER_VARIABLE > 15
           #define BIG_VARIABLE 1

       // Wrong
       #if MY_VARIABLE > 10
       #if MY_OTHER_VARIABLE > 15
       #define BIG_VARIABLE 1

#. Public PETSc include files, ``petsc*.h``, should not reference
   private PETSc ``petsc/private/*impl.h`` include files.

#. Public and private PETSc include files cannot reference include files
   located in the PETSc source tree.

#. All public functions must sanity-check their arguments using the appropriate
   ``PetscValidXXX()`` macros. These must appear between ``PetscFunctionBegin`` and
   ``PetscFunctionReturn()`` For example

   ::

       PetscErrorCode PetscPublicFunction(Vec v, PetscScalar *array, PetscInt collectiveInt)
       {
         PetscFunctionBegin;
         PetscValidHeaderSpecific(v,VEC_CLASSID,1);
         PetscValidScalarPointer(array,2);
         PetscValidLogicalCollectiveInt(v,collectiveInt,3);
         ...
         PetscFunctionReturn(PETSC_SUCCESS);
       }

   See ``include/petsc/private/petscimpl.h`` and search for "PetscValid" to see all
   available checker macros.

#. When possible, use ``PetscDefined()`` instead of preprocessor conditionals.
   For example use:

   ::

       if (PetscDefined(USE_DEBUG)) { ... }

   instead of:

   ::

       #if defined(PETSC_USE_DEBUG)
         ...
       #endif

   The former usage allows syntax and type checking in all configurations of
   PETSc, where as the latter needs to be compiled with and without debugging
   just to confirm that it compiles.

#. The first line of the executable statements in functions must be
   ``PetscFunctionBegin;``

#. Use ``PetscFunctionReturn(returnvalue)``, not
   ``return(returnvalue);``

#. *Never* put a function call in a ``return`` statement; do not write

   ::

       PetscFunctionReturn( somefunction(...) ); /* Incorrect */

#. Do *not* put a blank line immediately after ``PetscFunctionBegin;``
   or a blank line immediately before ``PetscFunctionReturn(PETSC_SUCCESS);``.

#. Do not use ``sqrt()``, ``pow()``, ``sin()``, and so on directly in
   PETSc C/C++ source code or examples (usage is fine in Fortran source
   code). Rather, use ``PetscSqrtScalar()``, ``PetscSqrtReal()``, and so
   on, depending on the context. See ``petscmath.h`` for expressions to
   use.

#. Do not include ``assert.h`` in PETSc source code. Do not use
   ``assert()``, it doesn’t play well in the parallel MPI world.
   You may use ``PetscAssert()`` where appropriate.

#. Try to make error messages short but informative. The user should be able to reasonably
   diagnose the greater problem from your error message.

#. Except in code that may be called before PETSc is fully initialized,
   always use ``PetscMallocN()`` (for example, ``PetscMalloc1()``),
   ``PetscCallocN()``, ``PetscNew()``, and ``PetscFree()``, not
   ``malloc()`` and ``free()``.

#. MPI routines and macros that are not part of the 2.1 standard
   should not be used in PETSc without appropriate ``configure``
   checks and ``#if PetscDefined()`` checks. Code should also be provided
   that works if the MPI feature is not available, for example,

   ::

       #if PetscDefined(HAVE_MPI_REDUCE_LOCAL)
         ierr = MPI_Reduce_local(inbuf,inoutbuf,count,MPIU_INT,MPI_SUM);CHKERRMPI(ierr);
       #else
         ierr = MPI_Reduce(inbuf,inoutbuf,count,MPIU_INT,
                           MPI_SUM,0,PETSC_COMM_SELF);CHKERRMPI(ierr);
       #endif

#. Do not introduce PETSc routines that provide essentially the same
   functionality as an available MPI routine. For example, do not write
   a routine ``PetscGlobalSum()`` that takes a scalar value and performs
   an ``MPI_Allreduce()`` on it. Instead, use the MPI routine
   ``MPI_Allreduce()`` directly in the code.

#. Never use a local variable counter such as ``PetscInt flops = 0;`` to
   accumulate flops and then call ``PetscLogFlops();`` *always* just
   call ``PetscLogFlops()`` directly when needed.

#. Library symbols meant to be directly usable by the user should be declared
   ``PETSC_EXTERN`` in their respective public header-file. Symbols intended to be for
   internal use only should instead be declared ``PETSC_INTERN``. Note that doing so is
   not necessary in the case of symbols local to a single translation unit, these should
   be declared ``static``. Note that PETSc can be configured to build a separate shared
   library for each top-level class (``Mat``, ``Vec``, ``KSP``, and so on) and that plugin
   implementations of these classes can be included as separate shared libraries; thus,
   otherwise private symbols may need to be marked ``PETSC_SINGLE_LIBRARY_INTERN``. For
   example

   -  ``MatStashCreate_Private()`` is marked ``PETSC_INTERN`` as it is used
      across compilation units, but only within the ``Mat`` package;

   -  all functions, such as ``KSPCreate()``, included in the public
      headers (``include/petsc*.h``) should be marked ``PETSC_EXTERN``;

   - ``PetscDeviceInitializeDefaultDevice_Internal()`` is marked
     ``PETSC_SINGLE_LIBRARY_INTERN`` as it may be used across library boundaries, but is
     not intended to be visible to users;

#. Before removing or renaming an API function, type, or enumerator,
   ``PETSC_DEPRECATED_XXX()`` should be used in the relevant header file
   to indicate the new, correct usage and the version number where the
   deprecation will first appear. The old function or type, with the
   deprecation warning, should remain for at least one major release.
   The function or type’s manual page should be updated (see :ref:`manual_page_format`).
   For example,

   ::

       typedef NewType OldType PETSC_DEPRECATED_TYPEDEF("Use NewType (since version 3.9)");

       PETSC_DEPRECATED_FUNCTION("Use NewFunction() (since version 3.9)") PetscErrorCode OldFunction();

       #define OLD_ENUMERATOR_DEPRECATED  OLD_ENUMERATOR PETSC_DEPRECATED_ENUM("Use NEW_ENUMERATOR (since version 3.9)")
       typedef enum {
         OLD_ENUMERATOR_DEPRECATED = 3,
         NEW_ENUMERATOR = 3
       } MyEnum;

   Note that after compiler preprocessing, the enum above would be transformed to something like

   ::

       typedef enum {
         OLD_ENUMERATOR __attribute((deprecated)) = 3,
         NEW_ENUMERATOR = 3
       } MyEnum;

#. Before removing or renaming an options database key,
   ``PetscOptionsDeprecated()`` should be used for at least one major
   release.

#. The format strings in PETSc ASCII output routines, such as
   ``PetscPrintf``, take a ``%" PetscInt_FMT "`` for all PETSc variables of type ``PetscInt``,
   not a ``%d``.

#. All arguments of type ``PetscReal`` to PETSc ASCII output routines,
   such as ``PetscPrintf``, must be cast to ``double``, for example,

   ::

       PetscPrintf(PETSC_COMM_WORLD,"Norm %g\n",(double)norm);

Formatted Comments
------------------

PETSc uses formatted comments and the Sowing packages
:cite:`gropp1993sowing` :cite:`gropp1993sowing2`
to generate documentation (manual pages) and the Fortran interfaces. Documentation
for Sowing and the formatting may be found at
http://wgropp.cs.illinois.edu/projects/software/sowing/; in particular,
see the documentation for ``doctext``. Currently doctext produces Markdown files ending in .md which
are later processed by Sphinx.

-  | ``/*@``
   | a formatted comment of a function that will be used for both documentation and a Fortran interface.

-  | ``/*@C``
   | a formatted comment of a function that will be used only for documentation, not to generate a Fortran interface. In general, such labeled C functions should have a custom Fortran interface provided. Functions that take ``char*`` or function pointer arguments must have the ``C`` symbol and a custom Fortran interface provided.

-  | ``/*E``
   | a formatted comment of an enum used for documentation only. Note that each of these needs to be listed in ``lib/petsc/conf/bfort-petsc.txt`` as a native and defined in the corresponding ``include/petsc/finclude/petscxxx.h`` Fortran include file and the values set as parameters in the file ``src/SECTION/f90-mod/petscSUBSECTION.h``, for example, ``src/vec/f90-mod/petscis.h``.

-  | ``/*S``
   | a formatted comment for a data type such as ``KSP``. Note that each of these needs to be listed in ``lib/petsc/conf/bfort-petsc.txt`` as a ``nativeptr``.

-  | ``/*J``
   | a formatted comment for a string type such as ``KSPType``.

-  | ``/*MC``
   | a formatted comment of a CPP macro or enum value for documentation.

The Fortran interface files supplied manually by developer go into the two
directories ``ftn-custom`` and ``f90-custom``, while those generated by
Sowing go into ``ftn-auto``.

Each include file that contains formatted comments needs to have a line of the form

   ::

       /* SUBMANSEC = submansec (for example Sys) */

preceded by and followed by a blank line. For source code this information is found in the makefile in that source code's directory in the format

   ::

       MANSEC   = DM
       SUBMANSEC= DMPlex

.. _manual_page_format :

Manual Page Format
~~~~~~~~~~~~~~~~~~

Each function, typedef, class, macro, enum, and so on in the public API
should include the following data, correctly formatted (see codes
section) to generate complete manual pages and Fortran interfaces with
Sowing. All entries below should be separated by blank lines. Except
where noted, add a newline after the section headings.

#. The item’s name, followed by a dash and brief (one-sentence)
   description.

#. If documenting a function implemented with a preprocessor macro
   (e.g., ``PetscOptionsBegin()``), an explicit ``Synopsis:`` section
   noting the required header and the function signature.

#. If documenting a function, a description of the function’s
   “collectivity”.

   -  ``Not Collective`` if the function need not be called on multiple (or possibly all) MPI
      ranks

   -  ``Collective [on xxx]`` if the function is a collective operation.
      If it is not with respect to the first argument then it is with respect
      to the argument with the variable name xxx. Note that xxx is the variable name, not the variable type.

   -  ``Logically Collective [on xxx][; yyy must contain common value]``
      if the function is collective but does not require any actual
      synchronization (e.g. setting class parameters uniformly). Any
      argument yyy which must have the same value on all ranks of the
      MPI communicator should be noted here.

#. If the function is not supported in Fortran then after the collective information, on the same line,
   one should provide ``; No Fortran support``.

#. If documenting a function with input parameters, a list of input
   parameter descriptions in an ``Input Parameter(s):`` section.

#. If documenting a function with output parameters, a list of output
   parameter descriptions in an ``Output Parameter(s):`` section.

#. If any input or output parameters are function pointers they should be documented in the style

   :: code-block:: sh

      Calling sequence of `func()`:
      $ PetscErrorCode func(PetscInt arg);
      . arg - the integer argument

#. If documenting a function that interacts with the options database, a
   list of options database keys in an ``Options Database Key(s):``
   section.

#. ``Level:`` (no newline) followed by ``beginner``,
   ``intermediate``, ``advanced``, ``developer``, or ``deprecated``. This
   should be listed before the various ``Note(s):`` sections.

#. (Optional) a ``Note(s):`` section containing in-depth discussion,
   technical caveats, special cases, and so on. If it is ambiguous
   whether returned pointers/objects need to be freed/destroyed by the
   user or not, this information should be mentioned here.

#. (If applicable) a ``Fortran Note(s):`` section detailing any relevant
   differences in calling or using the item from Fortran.

#. (If applicable) a ``Developer Note(s):`` section detailing any relevant
   information about the code for developers, for example, why a
   particular algorithm was implemented.

#. ``.seealso:`` (no newline, no spaces to the left of this text), followed by a list of related manual
   pages. These manual pages should usually also point back to this
   manual page in their ``seealso:`` sections. This is the final entry in the
   comment. There should be no blank line  after the ``.seealso`` items.

#. All PETSc functions that appear in a manual page (except the one in the header at the top) should end with a ``()`` and be enclosed
   in single back tick marks. All PETSc enum types and macros etc should also be enclosed in single back tick marks.
   This includes each item listed in the .seealso lines.

.. [1]
   Type also refers to the string name of the subclass.

Spelling and Capitalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Proper nouns, including Unix, Linux, X Windows, and Microsoft Windows should be fully written out and capitalized. This includes all operating systems.

#. Company names and product names should be capitalized.

#. Company names and terms that are traditionally all capitalized, for example, NVIDIA and CUDA should be all capitalized.

#. Unix should not be all capitalized.

#. Microsoft Windows should always be written out with two words. That is it should not be shortened to Windows.

#. CMake should be capitalized as shown.

#. BLAS and LAPACK are written in full capitalization


References
----------

.. bibliography:: /petsc.bib
   :filter: docname in docnames
