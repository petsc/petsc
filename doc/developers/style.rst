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
   underscore between each word. For example, ``mat,MAT_FINAL_ASSEMBLY``.

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

#. *No* tabs are allowed in *any* of the source code.

#. All PETSc function bodies are indented two characters.

#. Each additional level of loops, ``if`` statements, and so on is
   indented two more characters.

#. Wrapping lines should be avoided whenever possible.

#. Source code lines do not have a hard length limit; generally, we like
   them less than 150 characters wide.

#. The local variable declarations should be aligned. For example, use
   the style

   ::

       PetscScalar a;
       PetscInt    i,j;

   instead of

   ::

       PetscScalar a;
       PetscInt i,j; /* Incorrect */

#. Assignment and comparison operations, for example, ``x = 22.0`` or
   ``x < 22.0``, should have single spaces around the operator. This
   convention is true even when assignments are given directly in a line
   that declares the variable, such as ``PetscReal r = 22.3``. The
   exception is when these symbols are used in a ``for`` loop; then,
   there should be no spaces, for example, ``for (i=0; i<m; i++)``.
   Comparisons in ``while()`` constructs should have the spaces.

#. When declaring variables there should be no space between multiple
   variables, for example, ``PetscReal a,b,c``, not
   ``PetscReal a, b, c``.

#. The prototypes for functions should not include the names of the
   variables; for example, write

   ::

       PETSC_EXTERN PetscErrorCode MyFunction(PetscInt);

   not

   ::

       PETSC_EXTERN PetscErrorCode MyFunction(PetscInt myvalue); /* Incorrect */

#. All local variables of a particular type (for example, ``PetscInt``)
   should be listed on the same line if possible; otherwise, they should
   be listed on adjacent lines.

#. Equal signs should be aligned in regions where possible.

#. There *must* be a single blank line between the local variable
   declarations and the body of the function.

#. Indentation for ``if`` statements *must* be done as follows.

   ::

       if ( ) {
         ....
       } else {
         ....
       }

#. *Never* have

   ::

       if ( )
         a single indented line /* Incorrect */

   or

   ::

       for ( )
         a single indented line /* Incorrect */

   Instead, use either

   ::

       if ( ) a single statement

   or

   ::

       if ( ) {
         a single indented line
       }

   Note that error checking is a separate statement, so the following is
   *incorrect*

   ::

       if ( ) ierr = XXX();CHKERRQ(ierr); /* Incorrect */

   and instead you should use

   ::

       if ( ) {
         ierr = XXX();CHKERRQ(ierr);
       }

#. Always have a space between ``if`` or ``for`` and the following
   ``()``.

#. The open brace should be on the same line
   as the ``if ( )`` test, ``for ( )``, and so forth, not on its own
   line, for example,

   ::

        } else {

   instead of

   ::

        }
        else { /* Incorrect */

   See the next item for an exception. The closing
   brace should *always* be on its own line.

#. In function declarations, the opening brace
   should be on the *next* line, not on the same line as the function
   name and arguments. This is an exception to the previous item.

#. Do not leave sections of commented-out code in the source files.

#. Use classic block comments (``/* Comment */``) for multi-line comments and
   for *all* comments in headers.  Single-line comments in source files (*not*
   headers) may use the C99/C++ style (``// Comment``).  The rationale is that
   it must be possible for users to build applications using strict ``-std=c89``
   even though PETSc (since v3.14) uses select C99 features internally.

#. All variables must be declared at the beginning of the code block (C89
   style), never mixed in with code.  When variables are only used in a limited
   scope, it is encouraged to declare them in that scope.  For example::

     if (cond) {
       PetscScalar *tmp;

       ierr = PetscMalloc1(10,&tmp);CHKERRQ(ierr);
       // use tmp
       ierr = PetscFree(tmp);CHKERRQ(ierr);
     }

   It is also permissible to use ``for`` loop declarations::

     for (PetscInt i=0; i<n; i++) {
       // loop body
     }

#. Do not include a space after a ``(`` or before a ``)``. Do not write

   ::

       ierr = PetscMalloc1( 10,&a );CHKERRQ(ierr); /* Incorrect */

   but instead write

   ::

       ierr = PetscMalloc1(10,&a);CHKERRQ(ierr);

#. Do not use a space after the ``)`` in a cast or between the type and
   the ``*`` in a cast.

#. Do not include a space before or after a comma in lists. That is, do
   not write

   ::

       ierr = func(a, 22.0);CHKERRQ(ierr); /* Incorrect */

   but instead write

   ::

       ierr = func(a,22.0);CHKERRQ(ierr);

C Usage
~~~~~~~

#. Array and pointer arguments where the array values are not changed
   should be labeled as ``const`` arguments.

#. Scalar values passed to functions should *never* be labeled as
   ``const``.

#. Subroutines that would normally have a ``void**`` argument to return
   a pointer to some data should actually be prototyped as ``void*``.
   This prevents the caller from having to put a ``(void**)`` cast in
   each function call. See, for example, ``DMDAVecGetArray()``.

#. Do not use the ``register`` directive.

#. Do not use ``if (v == NULL)`` or
   ``if (flg == PETSC_TRUE)`` or ``if (flg == PETSC_FALSE)``. Instead, use
   ``if (!v)`` or ``if (flg)`` or ``if (!flg)``.

#. Do not use ``#ifdef`` or ``#ifndef``. Rather, use ``#if defined(...``
   or ``#if !defined(...``.  Better, use ``PetscDefined()`` (see below).

#. Never use system random number generators such as ``rand()`` in PETSc
   code or examples because these can produce different results on
   different systems thus making portability testing difficult. Instead
   use ``PetscRandom`` which produces the exact same results regardless
   of system it is used on.

#. Variadic macros may be used in PETSc source files, but must work with MSVC
   and must not be required in public headers (which must be usable with strict
   ``-std=c89``).  Most compilers have conforming implementations of the
   C99/C++11 rules for ``__VA_ARGS__``, but MSVC's implementation is not
   conforming and may need workarounds.  See ``PetscDefined()`` for an example
   of how to work around MSVC's limitations to write a macro that is usable in
   both.

#. Do not use language features that are not in the intersection of C99, C++11,
   and MSVC.  Examples of such features include designated initializers and
   variable-length arrays.  Note that variable-length arrays (including
   VLA-pointers) are not supported in C++ and were made optional in C11 and that
   designated initializers are not in C++.

.. _usage_of_petsc_functions_and_macros:

Usage of PETSc Functions and Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       PetscFunctionReturn(0);
     }

   See ``include/petsc/private/petscimpl.h`` and search for "PetscValid" to see all
   available checker macros.

#. When possible, use ``PetscDefined()`` instead of preprocessor conditionals.
   For example use::

     if (PetscDefined(USE_DEBUG)) { ... }

   instead of::

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
   or a blank line immediately before ``PetscFunctionReturn(0);``.

#. Do not use ``sqrt()``, ``pow()``, ``sin()``, and so on directly in
   PETSc C/C++ source code or examples (usage is fine in Fortran source
   code). Rather, use ``PetscSqrtScalar()``, ``PetscSqrtReal()``, and so
   on, depending on the context. See ``petscmath.h`` for expressions to
   use.

#. Do not include ``assert.h`` in PETSc source code. Do not use
   ``assert()``, it doesn’t play well in the parallel MPI world.

#. The macros ``SETERRQ()`` and ``CHKERRQ()`` should be on the same line
   as the routine to be checked unless doing so violates the 150
   character-width-rule. Try to make error messages short but
   informative.

#. Do not include a space before ``CHKXXX()``. That is, do not write

   ::

       ierr = PetscMalloc1(10,&a); CHKERRQ(ierr); /* Incorrect */

   but instead write

   ::

       ierr = PetscMalloc1(10,&a);CHKERRQ(ierr);

#. Except in code that may be called before PETSc is fully initialized,
   always use ``PetscMallocN()`` (for example, ``PetscMalloc1()``),
   ``PetscCallocN()``, ``PetscNew()``, and ``PetscFree()``, not
   ``malloc()`` and ``free()``.

#. MPI routines and macros that are not part of the 1.0 or 1.1 standard
   should not be used in PETSc without appropriate ``configure``
   checks and ``#if defined()`` checks. Code should also be provided
   that works if the MPI feature is not available, for example,

   ::

       #if defined(PETSC_HAVE_MPI_IN_PLACE)
         ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,
                               recvcounts,displs,MPIU_INT,comm);CHKERRQ(ierr);
       #else
         ierr = MPI_Allgatherv(lens,sendcount,MPIU_INT,lens,recvcounts,
                               displs,MPIU_INT,comm);CHKERRQ(ierr);
       #endif

#. Do not introduce PETSc routines that provide essentially the same
   functionality as an available MPI routine. For example, do not write
   a routine ``PetscGlobalSum()`` that takes a scalar value and performs
   an ``MPI_Allreduce()`` on it. Instead, use the MPI routine
   ``MPI_Allreduce()`` directly in the code.

#. Never use a local variable counter such as ``PetscInt flops = 0;`` to
   accumulate flops and then call ``PetscLogFlops();`` *always* just
   call ``PetscLogFlops()`` directly when needed.

#. Library functions should be declared
   ``PETSC_INTERN`` if they are intended to be visible only within a
   single PETSc shared library. They should be declared ``PETSC_EXTERN``
   if intended to be visible across shared libraries. Note that PETSc
   can be configured to build a separate shared library for each
   top-level class (``Mat``, ``Vec``, ``KSP``, and so on) and that
   plugin implementations of these classes can be included as separate
   shared libraries; thus, private functions may need to be marked
   ``PETSC_EXTERN``. For example,

   -  ``MatStashCreatePrivate`` is marked ``PETSC_INTERN`` as it is used
      across compilation units, but only within the ``Mat`` package;

   -  all functions, such as ``KSPCreate()``, included in the public
      headers (``include/petsc*.h``) should be marked ``PETSC_EXTERN``;

#. Before removing or renaming an API function, type, or enumerator,
   ``PETSC_DEPRECATED_XXX()`` should be used in the relevant header file
   to indicate the new, correct usage and the version number where the
   deprecation will first appear. For example,

   ::

       typedef NewType OldType PETSC_DEPRECATED_TYPEDEF("Use NewType (since version 3.9)");
       PETSC_DEPRECATED_FUNCTION("Use NewFunction() (since version 3.9)") PetscErrorCode OldFunction();
       #define OLD_ENUMERATOR_DEPRECATED  OLD_ENUMERATOR PETSC_DEPRECATED_ENUM("Use NEW_ENUMERATOR (since version 3.9)")
       typedef enum {
         OLD_ENUMERATOR_DEPRECATED = 3,
         NEW_ENUMERATOR = 3
       } MyEnum;

   The old function or type, with the deprecation warning, should remain
   for at least one major release. The function or type’s manual page
   should be updated (see :ref:`manual_page_format`).

#. Before removing or renaming an options database key,
   ``PetscOptionsDeprecated()`` should be used for at least one major
   release.

#. The format strings in PETSc ASCII output routines, such as
   ``PetscPrintf``, take a ``%D`` for all PETSc variables of type ``PetscInt``,
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
see the documentation for ``doctext``.

-  | ``/*@``
   | a formatted comment of a function that will be used for both
   | documentation and a Fortran interface.

-  | ``/*@C``
   | a formatted comment of a function that will be used only for
   | documentation, not to generate a Fortran interface. In general, such
   | labeled C functions should have a custom Fortran interface provided.
   | Functions that take ``char*`` or function pointer arguments must have
   | the ``C`` symbol and a custom Fortran interface provided.

-  | ``/*E``
   | a formatted comment of an enum used for documentation only. Note
   | that each of these needs to be listed in
   | ``lib/petsc/conf/bfort-petsc.txt`` as a native and defined in the
   | corresponding ``include/petsc/finclude/petscxxx.h`` Fortran include
   | file and the values set as parameters in the file
   | ``src/SECTION/f90-mod/petscSUBSECTION.h``, for example,
   | ``src/vec/f90-mod/petscis.h``.

-  | ``/*S``
   | a formatted comment for a data type such as ``KSP``. Note that each
   | of these needs to be listed in ``lib/petsc/conf/bfort-petsc.txt`` as
   | a ``nativeptr``.

-  | ``/*MC``
   | a formatted comment of a CPP macro or enum value for documentation.

The Fortran interface files supplied by the user go into the two
directories ``ftn-custom`` and ``f90-custom``, while those generated by
Sowing go into ``ftn-auto``.

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
   “collectivity” (whether all ranks in an MPI communicator need to
   participate). Unless otherwise noted, it’s assumed that this
   collectivity is with respect to the MPI communicator associated with
   the first argument.

   -  ``Not Collective`` if the function need not be called on all MPI
      ranks

   -  ``Collective [on XXX]`` if the function is a collective operation
      (with respect to the MPI communicator associated with argument
      ``XXX``)

   -  ``Logically Collective [on XXX][; YYY must contain common value]``
      if the function is collective but does not require any actual
      synchronization (e.g. setting class parameters uniformly). Any
      argument YYY which must have the same value on all ranks of the
      MPI communicator should be noted here.

#. If documenting a function with input parameters, a list of input
   parameter descriptions in an ``Input Parameters:`` section.

#. If documenting a function with output parameters, a list of output
   parameter descriptions in an ``Output Parameters:`` section.

#. If documenting a function that interacts with the options database, a
   list of options database keys in an ``Options Database Keys:``
   section.

#. (Optional) a ``Notes:`` section containing in-depth discussion,
   technical caveats, special cases, and so on. If it is ambiguous
   whether returned pointers/objects need to be freed/destroyed by the
   user or not, this information should be mentioned here.

#. (If applicable) a ``Fortran Notes:`` section detailing any relevant
   differences in calling or using the item from Fortran.

#. ``Level:`` (no newline) followed by ``beginner``,
   ``intermediate``, ``advanced``, ``developer``, or ``deprecated``.

#. ``.seealso:`` (no newline), followed by a list of related manual
   pages. These manual pages should usually also point back to this
   manual page in their ``seealso:`` sections.

.. [1]
   Type also refers to the string name of the subclass.

References
----------

.. bibliography:: /../src/docs/tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: /../src/docs/tex/petscapp.bib
   :filter: docname in docnames
