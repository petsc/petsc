/**

*   \page dev-style-guide Style Guide

The %PETSc team uses certain conventions to make our source code consistent. Groups
developing classes compatible with %PETSc are, of course, free to organize their
own source code anyway they like.

\section dev-style-guide-names Names
Consistency of names for variables, functions, etc. is extremely
important in making the package both usable and maintainable.
We use several conventions:
  - All function names and enum types consist of words, each of
      which is capitalized, for example `KSPSolve()` and
      `MatGetOrdering()`.
  - All enum elements and macro variables are capitalized. When
      they consist of several complete words, there is an underscore between each word.
  - Functions that are private to %PETSc (not callable by the
      application code) either
        - have an appended `_Private` (for example,
           `StashValues_Private`) or
        - have an appended `_<class>Subtype` (for example,
           `MatMult_SeqAIJ`).

      In addition, functions that are not intended for use outside
      of a particular file are declared static.
  - Function names in structures are the same as the base application
      function name without the object prefix, and all are in small letters.
      For example, `MatMultTranspose()` has a structure name of
      `multtranspose()`.
  - Each application usable function begins with the name of the class object, followed by any subclass name,
      for example, `ISInvertPermutation()`, `MatMult()` or `KSPGMRESSetRestart()`.
  - Options database keys are lower case, have an underscore between words and match the function name associated with the option without the word set. 
      For example, `-ksp_gmres_restart`.


\section dev-style-guide-coding-conventions Coding Conventions and Style Guide

Within the %PETSc source code, we adhere to the following guidelines
so that the code is uniform and easily maintainable:

  - All %PETSc function bodies are indented two characters.
  - Each additional level of loops, if statements, etc. is indented
      two more characters.
  - Wrapping lines should be avoided whenever possible.
  - Source code lines do not have a hard length limit, generally we prefer lines to be less than 150 characters wide.
  - The local variable declarations should be aligned. For example,
      use the style
\code
   int    i,j;
   Scalar a;
\endcode
instead of
\code
   int i,j;
   Scalar a;
\endcode
  - All local variables of a particular type (e.g., `int`) should be
      listed on the same line if possible; otherwise, they should be listed
      on adjacent lines.
  - Equal signs should be aligned in regions where possible.
  - There **must** be a single blank line
      between the local variable declarations and the body of the function.
  - The first line of the executable statments must be PetscFunctionBegin;
  - The following text should be before each function
\code
#undef __FUNCT__
#define __FUNCT__ ``FunctionName''
\endcode
this is used by various macros (for example the error handlers) to always know
what function one is in.
  - Use `PetscFunctionReturn(returnvalue);`, not `return(returnvalue);`
  - **Never** put a function call in a return statment; do not do
\code
   PetscFunctionReturn( somefunction(...) );
\endcode
  - Do **not** put a blank line immediately after PetscFunctionBegin; or
a blank line immediately before PetscFunctionReturn(0);.
  - Indentation for `if` statements **must** be done as
\code
   if (  ) {
     ....
   } else {
     ....
   }
\endcode
\item **Never**  have
\code
   if (  )
     a single indented line
\endcode
or
\code
   for (  )
     a single indented line
\endcode
instead use either
\code
   if (  ) a single line
\endcode
or
\code
   if (  ) {
     a single indented line
   }
\endcode
  - **No** tabs are allowed in **any** of the source code.
  - The open bracket `{` should be on the same line as the `if ()` test, `for ()`, etc. never on
      its own line. The closing bracket `}` should **always** be on its own line.
  - In function declaration the open bracket `{` should be on the **next** line, not on the same line as the function name and
      arguments. This is an exception to the rule above.
  - The macros SETERRQ() and CHKERRQ() should be on the
      same line as the routine to be checked unless this violates the
      150 character width rule. Try to make error messages short, but
      informative.
  - **No** space after a `(` or before a `)`. No space before the `CHKXXX()`. That is, do not write
\code
   ierr = PetscMalloc( 10*sizeof(int),&a ); CHKERRQ(ierr);
\endcode
instead write
\code
   ierr = PetscMalloc(10*sizeof(int),&a);CHKERRQ(ierr);
\endcode
  - **No** space after the `)` in a cast, no space between the type and the `*` in a cast.
  - **No** space before or after a comma in lists
That is, do not write
\code
    int a, b,c;
    ierr = func(a, 22.0);CHKERRQ(ierr);
\endcode
instead write
\code
    int a,b,c;
    ierr = func(a,22.0);CHKERRQ(ierr);
\endcode

  - Subroutines that would normally have `void **` argument to return a pointer to some data, should actually be protyped as as `void*`.
    This prevents the caller from having to put a `(void**)` cast in each function call. See, for example, `DMDAVecGetArray()`.

  - Do not use the `register` directive.
  - Never use a local variable counter like `PetscInt flops = 0`; to accumulate flops and then call `PetscLogFlops()`. **Always** just
      call `PetscLogFlops()` directly when needed.
  - Do not use `if (rank == 0)` or `if (v == PETSC\_NULL)` or `if (flg == PETSC_TRUE)` or `if (flg == PETSC_FALSE)`
instead use `if (!rank)` or `if (!v)` or `if (flg)` or `if (!flg)`.
  - Do not use `#ifdef` or `#ifndef`, rather use `#if defined(...)` or `#if !defined(...)`
  - MPI routines and macros that are not part of the 1.0 or 1.1 standard should not be used in PETSc without appropriate `./configure` checks and `#if defined()` checks the code. Code should also be provided that works if the MPI feature is not available. For example,
\code
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr  = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
#else
    ierr  = MPI_Allgatherv(lens+A->rmap->rstart,sendcount,MPIU_INT,lens,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
#endif
\endcode
  - There shall be no %PETSc routines introduced that provide essentially the same functionality as an available MPI routine. For example, one should not write a routine `PetscGlobalSum()` that takes a scalar value and performs an `MPI_Allreduce()` on it. One should use `MPI_Allreduce()` directly in the code.
  - XXXTypes (for example KSPType) do not have an underscore in them, unless they refer to another package that uses an underscore, for example `MATSOLVERSUPERLU_DIST`.


\section dev-style-guide-option-names Option Names

Since consistency simplifies usage and code maintenance, the names of
%PETSc routines, flags, options, etc. have been selected with great care.
The default option names are of the form `-<class>_sub<class>_name`.
For example, the option name for the basic convergence tolerance for
the KSP package is `-ksp_atol`. In addition, operations in different
packages of a similar nature have a similar name.  For example, the option
name for the basic convergence tolerance for the SNES package is
`-snes_atol`.

When a `Set` is included in a function name, it is dropped in the options key.
For example `KSPGMRESSetRestart()` becomes `-ksp_gmres_restart`.

*/
