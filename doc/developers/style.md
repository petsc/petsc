(style)=

# PETSc Style and Usage Guide

The PETSc team uses certain conventions to make the source code
consistent and easier to maintain. We will interchangeably use the
terminology *subclass*, *implementation*, or *type* [^footnote-1] to refer to a
concrete realization of an abstract base class. For example,
`KSPGMRES` is a type for the base class `KSP`.

## Names

Consistency of names for variables, functions, and so on is extremely
important. We use several conventions

01. All function names and enum types consist of acronyms or words, each
    of which is capitalized, for example, `KSPSolve()` and
    `MatGetOrdering()`.

02. All enum elements and macro variables are named with all capital
    letters. When they consist of several complete words, there is an
    underscore between each word. For example, `MAT_FINAL_ASSEMBLY`.

03. Functions that are private to PETSc (not callable by the application
    code) either

    - have an appended `_Private` (for example, `StashValues_Private`)
      or
    - have an appended `_Subtype` (for example, `MatMultSeq_AIJ`).

    In addition, functions that are not intended for use outside of a
    particular file are declared `static`. Also, see the item
    on symbol visibility in {ref}`usage_of_petsc_functions_and_macros`.

04. Function names in structures (for example, `_matops`) are the same
    as the base application function name without the object prefix and
    in lowercase. For example, `MatMultTranspose()` has a
    structure name of `multtranspose`.

05. Names of implementations of class functions should begin with the
    function name, an underscore, and the name of the implementation, for
    example, `KSPSolve_GMRES()`.

06. Each application-usable function begins with the name of the class
    object, followed by any subclass name, for example,
    `ISInvertPermutation()`, `MatMult()`, or
    `KSPGMRESSetRestart()`.

07. Functions that PETSc provides as defaults for user-providable
    functions end with `Default` (for example, `PetscSignalHandlerDefault()`).

08. Options database keys are lower case, have an underscore between
    words, and match the function name associated with the option without
    the word “set” or “get”, for example, `-ksp_gmres_restart`.

09. Specific `XXXType` values (for example, `MATSEQAIJ`) do not have
    an underscore in them unless they refer to another package that uses
    an underscore, for example, `MATSOLVERSUPERLU_DIST`.

10. Typedefs for functions should end in `Fn` as in, for example, `SNESFunctionFn`.

(stylepetsccount)=

## PETSc and standard datatypes

1. `PetscInt` is generally used for array indices, and array lengths. It
   is a signed 32-bit or 64-bit `int` depending on the `./configure` option
   `--with-64-bit-indices`. There is the possibility of integer overflow with the
   32-bit version.

2. `PetscCount` is `ptrdiff_t`, (on 64-bit systems it is 64-bits) and should be used for array sizes (number of entries)
   and indices that may become large. `PetscIntCast()` should always be used when converting
   to `PetscInt` from `PetscCount`. Since, in most configurations an array of `PetscCount` requires twice the memory
   of an array of `PetscInt` most index arrays (such as `ISGetIndices()` use `PetscInt`,
   when these arrays get too large then `--with-64-bit-indices` must be used to
   `./configure` PETSc. In most cases it is appropriate to use `PetscCount` in lieu of `PetscInt64`.

3. `size_t` is used for variables that contain the amount of memory, generally in bytes.
   It should **not** be used for the number of
   entries in an array, or to index into an array, that should be `PetscCount`, or `PetscInt`.
   Though `size_t` is unsigned and hence can have values larger then those that can be stored
   in a `PetscCount` those sizes will never be reached in practice so it is ok to cast with `(PetscCount)`
   from a `size_t` variable to a `PetscCount` variable, but **not** a `PetscInt`.
   One should not blindly cast from a `PetscCount` or a `PetscInt`
   to `size_t` since, when the value is negative, it will produce garbage.

4. **Never** blindly put in a cast from a potentially longer (in number of bits) to a shorter integer such as

   ```
   PetscInt64 a
   PetscInt b
   b = (PetscInt)a
   ```

   simply to prevent a compiler warning. Use the appropriate PETSc cast function unless you
   absolutely know the value will fit in the lesser number of bits.

5. MPI 4.0 supports the use of `MPI_Count` (large count) for many MPI functions that previously used `int` in a new API where the MPI function
   names end in `_c`. Since not all installed MPI implementations have such support, use `MPIU_XXX()` routines
   that use `PetscCount` for count arguments and use the large count MPI versions when possible.
   When not possible they first check the size of the input count arguments and error if they
   will not fit in the MPI required `int`, if they fit then the standard MPI functions are automatically called.

   ```
   sizeof(MPI_Count) >= sizeof(PetscCount) >= sizeof(PetscInt)
   sizeof(PetscInt64) >= sizeof(PetscCount) >= sizeof(PetscInt)
   sizeof(MPI_Count) may be strictly greater than sizeof(PetscInt64)
   ```

## Coding Conventions and Style

Within the PETSc source code, we adhere to the following guidelines so
that the code is uniform and easily maintained.

### C Formatting

The `.clang-format` file in the PETSc root directory controls the white space and basic layout. You can run the formatter in the entire repository with `make clangformat`. All merge requests must be properly formatted; this is automatically checked for merge requests with `make checkclangformat`.

Even with the use of `clang-format` there are still many decisions about code formatting that must be constantly made. A subset of these is automatically checked for merge requests with `make checkbadSource`.

01. The prototypes for functions should not include the names of the
    variables

    ```
    PETSC_EXTERN PetscErrorCode MyFunction(PetscInt); // Correct
    PETSC_EXTERN PetscErrorCode MyFunction(PetscInt myvalue); // Incorrect
    ```

02. All local variables of a particular type (for example, `PetscInt`) should be listed
    on the same line if possible; otherwise, they should be listed on adjacent lines. Note
    that pointers of different arity (levels of indirection) are considered to be different types. `clang-format` automatically
    handles the indenting shown below.

    ```
    // Correct
    PetscInt   a, b, c;
    PetscInt  *d, *e;
    PetscInt **f;

    // Incorrect
    PetscInt a, b, c, *d, *e, **f;
    ```

03. Local variables should be initialized in their declaration when possible

    ```
    // Correct
    PetscInt a = 11;

    PetscFunctionBegin;
    // use a

    // Incorrect
    PetscInt a;

    PetscFunctionBegin;
    a = 11;
    // use a
    ```

04. All PETSc subroutine code blocks *must* start with a single blank line between the local variable
    declarations followed by `PetscFunctionBegin`.

    ```
    // Correct
    PetscInt x;

    PetscFunctionBegin;

    // Incorrect
    PetscInt x;
    PetscFunctionBegin;

    // Incorrect
    PetscInt x;
    y = 11;
    ```

05. Functions in PETSc examples, including `main()` should have `PetscFunctionBeginUser` as the first line after the local variable declarations.

06. PETSc functions that begin `PetscFunctionBegin` must always return via `PetscFunctionReturn()`, or `PetscFunctionReturnVoid()`, not `return`. If the function returns a `PetscErrorCode`, then it must always return with `PetscFunctionReturn(PETSC_SUCCESS)`.

07. Functions that do use return should use `return xx;` rather than `return(xx);`

08. All PETSc function calls must have their return value checked for errors using the
    `PetscCall()` macro. This should be wrapped around the function in question.

    ```
    PetscCall(MyFunction(...)); // Correct
    PetscErrorCode ierr = MyFunction(...);PetscCall(ierr); // Incorrect
    ```

    The only exceptions to this rule are begin-end style macros which embed local variables
    or loops as part of their expansion
    (e.g. `PetscOptionsBegin()`/`PetscOptionsEnd()`). These handle errors internally
    and do not need error checking.

    ```
    // Correct
    PetscOptionsBegin(...);
    PetscOptionsEnd();
    ```

    As a rule, always try to wrap the function first; if this fails to compile, you do
    not need to add the error checking.

    Calls to external package functions are generally made with `PetscCallExternal()` or its variants that are specialized for particular packages, for example `PetscCallBLAS()`

09. Single operation `if` and `else` commands should not be wrapped in braces. They should be done as follows,

    ```
    if ( ) XXXX;
    else YYY;
    ```

    not

    ```
    if ( ) {XXXX;}
    else {YYY;}
    ```

10. Do not leave sections of commented-out code or dead source code protected with `ifdef foo` in the source files.

11. Use classic block comments (`/* There must be a space before the first word in the comment and a space at the end */`,
    (`/*Do not do this*/`) for multi-line comments, and `// Comment` for single-line comments in source files.

12. Do not put a `*` at the beginning or end of each line of a multi-line comment.

13. Do not use `/* ---- ... ----- */` or similar constructs to separate parts of source code files.

14. Use appropriate grammar and spelling in the comments.

15. All variables must be declared at the beginning of the code block (C89
    style), never mixed in with code. However, when variables are only used in a limited
    scope, it is encouraged to declare them in that scope. For example:

    ```
    if (cond) {
      PetscScalar *tmp;

      PetscCall(PetscMalloc1(10, &tmp));
      // use tmp
      PetscCall(PetscFree(tmp));
    }
    ```

    The only exception to this is variables used exclusively within a `for` loop, which must
    be declared inside the loop initializer:

    ```
    // Correct
    for (PetscInt i = 0; i < n; ++i) {
      // loop body
    }
    ```

    ```
    // Correct, variable used outside of loop
    PetscInt i;
    ```

    ```
    for (i = 0; i < n; ++i) {
      // loop body
    }
    j = i;
    ```

    ```
    // Incorrect
    PetscInt i;
    ...
    for (i = 0; i < n; ++i) {
      // loop body
    }
    ```

16. Developers can use // to split very long lines when it improves code readability. For example

    ```
    f[j][i].omega = xdot[j][i].omega + uxx + uyy //
                  + (vxp * (u - x[j][i - 1].omega) + vxm * (x[j][i + 1].omega - u)) * hy //
                  + (vyp * (u - x[j - 1][i].omega) + vym * (x[j + 1][i].omega - u)) * hx //
                  - .5 * grashof * (x[j][i + 1].temp - x[j][i - 1].temp) * hy;
    ```

17. The use of `// clang-format off` is allowed in the source code but should only be used when necessary. It should not
    be used when trailing // to split lines works.

    ```
    // clang-format off
    f ...
    // clang-format on
    ```

18. `size` and `rank` should be used exclusively for the results of `MPI_Comm_size()` and `MPI_Comm_rank()` and other variable names for these values should be avoided unless necessary.

### C Usage

01. Do not use language features that are not in the intersection of C99, C++11, and MSVC
    v1900+ (Visual Studio 2015). Examples of such banned features include variable-length arrays.
    Note that variable-length arrays (including VLA-pointers) are not supported in C++ and
    were made optional in C11. You may use designated initializers via the
    `PetscDesignatedInitializer()` macro.

02. Array and pointer arguments where the array values are not changed
    should be labeled as `const` arguments.

03. Scalar values passed to functions should *never* be labeled as
    `const`.

04. Subroutines that would normally have a `void **` argument to return
    a pointer to some data should be prototyped as `void *`.
    This prevents the caller from having to put a `(void **)` cast in
    each function call. See, for example, `DMDAVecGetArray()`.

05. Do not use the `register` directive.

06. Use `if (v == NULL)` or `if (flg == PETSC_TRUE)`, instead of using `if (!v)` or `if (flg)` or `if (!flg)`.

07. Avoid `#ifdef` or `#ifndef` when possible. Rather, use `#if defined` or `#if
    !defined`. Better, use `PetscDefined()` (see below). The only exception to this
    rule is for header guards, where the `#ifndef` form is preferred (see below).

08. Header guard macros should be done using `#pragma once`. This must be the very first
    non-comment line of the file. There must be no leading or trailing empty (non-comment)
    lines in the header. For example, do

    ```
    /*
      It's OK to have

      comments
    */
    // before the guard
    #pragma once

    // OK, other headers included after the guard
    #include <petscdm.h>
    #include <petscdevice.h>

    // OK, other preprocessor symbols defined after the guard
    #define FOO_BAR_BAZ

    // OK, regular symbols defined after the guard
    typedef struct _p_PetscFoo *PetscFoo;
    ...
    ```

    Do not do

    ```
    // ERROR, empty lines at the beginning of the header



    // ERROR, included other headers before the guard
    #include <petscdm.h>
    #include <petscdevice.h>

    // ERROR, defined other preprocessor symbols before the guard
    #define FOO_BAR_BAZ

    // ERROR, defined regular symbols before the guard
    typedef struct _p_PetscFoo *PetscFoo;

    #pragma once
    ```

09. Never use system random number generators such as `rand()` in PETSc
    code or examples because these can produce different results on
    different systems, thus making portability testing difficult. Instead,
    use `PetscRandom` which produces the same results regardless
    of the system used.

10. Variadic macros may be used in PETSc, but must work with MSVC v1900+ (Visual Studio
    2015). Most compilers have conforming implementations of the C99/C++11 rules for
    `__VA_ARGS__`, but MSVC's implementation is not conforming and may need workarounds.
    See `PetscDefined()` for an example of how to work around MSVC's limitations to write
    a macro that is usable in both.

(usage_of_petsc_functions_and_macros)=

### Usage of PETSc Functions and Macros

01. Lengthy conditional preprocessor blocks should mark any `#else` or `#endif`
    directives with a comment containing (or explaining) either the boolean condition or
    the macro's name if the first directive tests whether one is defined. One
    should be able to read any part of the macroblock and find or deduce the
    initial `#if`. That is:

    ```
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
    ```

02. Public PETSc include files, `petsc*.h`, should not reference
    private PETSc `petsc/private/*impl.h` include files.

03. Public and private PETSc include files cannot reference include files
    located in the PETSc source tree.

04. All public functions must sanity-check their arguments using the appropriate
    `PetscValidXXX()` macros. These must appear between `PetscFunctionBegin` and
    `PetscFunctionReturn()` For example

    ```
    PetscErrorCode PetscPublicFunction(Vec v, PetscScalar *array, PetscInt collectiveInt)
    {
      PetscFunctionBegin;
      PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
      PetscAssertPointer(array, 2);
      PetscValidLogicalCollectiveInt(v, collectiveInt, 3);
      ...
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    ```

    See `include/petsc/private/petscimpl.h` and search for "PetscValid" to see all
    available checker macros.

05. When possible, use `PetscDefined()` instead of preprocessor conditionals.
    For example, use:

    ```
    if (PetscDefined(USE_DEBUG)) { ... }
    ```

    instead of:

    ```
    #if defined(PETSC_USE_DEBUG)
      ...
    #endif
    ```

    The former usage allows syntax and type-checking in all configurations of
    PETSc, whereas the latter needs to be compiled with and without debugging
    to confirm that it compiles.

06. *Never* put a function call in a `return` statement; do not write

    ```
    PetscFunctionReturn( somefunction(...) ); /* Incorrect */
    ```

07. Do *not* put a blank line immediately after `PetscFunctionBegin;`
    or a blank line immediately before `PetscFunctionReturn(PETSC_SUCCESS);`.

08. Do not include `assert.h` in PETSc source code. Do not use
    `assert()`, it doesn’t play well in the parallel MPI world.
    You may use `PetscAssert()` where appropriate. See `PetscCall()` documentation
    for guidance of when to use `PetscCheck()` vs. `PetscAssert()`.

09. Make error messages short but informative. The user should be able to reasonably
    diagnose the greater problem from your error message.

10. Except in code that may be called before PETSc is fully initialized,
    always use `PetscMallocN()` (for example, `PetscMalloc1()`),
    `PetscCallocN()`, `PetscNew()`, and `PetscFree()`, not
    `malloc()` and `free()`.

11. MPI routines and macros that are not part of the 2.1 standard
    should not be used in PETSc without appropriate `configure`
    checks and `#if PetscDefined()` checks. Code should also be provided
    that works if the MPI feature is not available; for example,

    ```
    #if PetscDefined(HAVE_MPI_REDUCE_LOCAL)
      PetscCallMPI(MPI_Reduce_local(inbuf, inoutbuf, count, MPIU_INT, MPI_SUM));
    #else
      PetscCallMPI(MPI_Reduce(inbuf, inoutbuf, count, MPIU_INT, MPI_SUM, 0, PETSC_COMM_SELF);
    #endif
    ```

12. Do not introduce PETSc routines that provide essentially the same
    functionality as an available MPI routine. For example, do not write
    a routine `PetscGlobalSum()` that takes a scalar value and performs
    an `MPI_Allreduce()` on it. Instead, use the MPI routine
    `MPI_Allreduce()` directly in the code.

13. Never use a local variable counter such as `PetscInt flops = 0;` to
    accumulate flops and then call `PetscLogFlops();` *always* just
    call `PetscLogFlops()` directly when needed.

14. Library symbols meant to be directly usable by the user should be declared
    `PETSC_EXTERN` in their respective public header file. Symbols intended for internal use should instead be declared `PETSC_INTERN`. Note that doing so is
    unnecessary in the case of symbols local to a single translation unit; these should
    be declared `static`. PETSc can be configured to build a separate shared
    library for each top-level class (`Mat`, `Vec`, `KSP`, and so on), and that plugin
    implementations of these classes can be included as separate shared libraries; thus,
    otherwise private symbols may need to be marked `PETSC_SINGLE_LIBRARY_INTERN`. For
    example

    - `MatStashCreate_Private()` is marked `PETSC_INTERN` as it is used
      across compilation units, but only within the `Mat` package;
    - all functions, such as `KSPCreate()`, included in the public
      headers (`include/petsc*.h`) should be marked `PETSC_EXTERN`;
    - `VecLoad_Default()` is marked
      `PETSC_SINGLE_LIBRARY_INTERN` as it may be used across library boundaries, but is
      not intended to be visible to users;

15. Before removing or renaming an API function, type, or enumerator,
    `PETSC_DEPRECATED_XXX()` should be used in the relevant header file
    to indicate the new usage and the PETSc version number where the
    deprecation will first appear. The old function or type, with the
    deprecation warning, should remain for at least one major release. We do not remove support for the
    deprecated functionality unless there is a specific reason to remove it; it is not removed simply because
    it has been deprecated for "a long time."

    The function or type’s manual page should be updated (see {ref}`manual_page_format`).
    For example,

    ```
    typedef NewType OldType PETSC_DEPRECATED_TYPEDEF("Use NewType (since version 3.9)");

    PETSC_DEPRECATED_FUNCTION("Use NewFunction() (since version 3.9)") PetscErrorCode OldFunction();

    #define OLD_ENUMERATOR_DEPRECATED  OLD_ENUMERATOR PETSC_DEPRECATED_ENUM("Use NEW_ENUMERATOR (since version 3.9)")
    typedef enum {
      OLD_ENUMERATOR_DEPRECATED = 3,
      NEW_ENUMERATOR = 3
    } MyEnum;
    ```

    Note that after compiler preprocessing, the enum above would be transformed into something like

    ```
    typedef enum {
      OLD_ENUMERATOR __attribute__((deprecated)) = 3,
      NEW_ENUMERATOR = 3
    } MyEnum;
    ```

16. Before removing or renaming an options database key,
    `PetscOptionsDeprecated()` should be used for at least one major
    release. We do not remove support for the
    deprecated functionality unless there is a specific reason to remove it; it is not removed simply because
    it has been deprecated for "a long time."

17. The format strings in PETSc ASCII output routines, such as
    `PetscPrintf()`, take a `%" PetscInt_FMT "` for all PETSc variables of type `PetscInt`,
    not a `%d`.

18. All arguments of type `PetscReal` to PETSc ASCII output routines,
    such as `PetscPrintf`, must be cast to `double`, for example,

    ```
    PetscPrintf(PETSC_COMM_WORLD, "Norm %g\n", (double)norm);
    ```

## Formatted Comments

PETSc uses formatted comments and the Sowing packages {cite}`gropp1993sowing` {cite}`gropp1993sowing2`
to generate documentation (manual pages) and the Fortran interfaces. Documentation
for Sowing and the formatting may be found at
<http://wgropp.cs.illinois.edu/projects/software/sowing/>; in particular,
see the documentation for `doctext`. Currently, doctext produces Markdown files ending in `.md`, which
Sphinx later processes.

- `/*@`
  a formatted comment of a function that will be used for documentation and a Fortran interface.
- `/*@C`
  a formatted comment of a function that will be used only for documentation, not to generate a Fortran interface. Certain constructs and usages do not yet support automatically generating a Fortran interface. In general, such labeled C functions should have a custom Fortran interface provided.
- `/*E`
  a formatted comment of an enum used for documentation only.
  .
- `/*S`
  a formatted comment for a data type such as 
  `KSP`
  .
- `/*J`
  a formatted comment for a string type such as 
  `KSPType`
  .
- `/*MC`
  a formatted comment of a CPP macro or enum value for documentation.

The Fortran interface files supplied manually by the developer go into the 
directory `ftn-custom`, while those automatically generated
go into directories in the \$PETSC_ARCH/ftn\`\` directory tree.

Each include file that contains formatted comments needs to have a line of the form

> ```
> /* SUBMANSEC = submansec (for example Sys) */
> ```

preceded by and followed by a blank line. For source code, this information is found in the makefile in that source code's directory in the format

> ```
> MANSEC   = DM
> SUBMANSEC= DMPlex
> ```

(manual_page_format)=

### Manual Page Format

Each function, typedef, class, macro, enum, and so on in the public API
should include the following data, correctly formatted (see codes
section) to generate complete manual pages and (possibly) Fortran interfaces with
Sowing. All entries below should be separated by blank lines. Except
where noted, add a newline after the section headings.

01. The item’s name, followed by a dash and brief (one-sentence)
    description.

02. If documenting a function implemented with a preprocessor macro
    (e.g., `PetscOptionsBegin()`), an explicit `Synopsis:` section
    noting the required header and the function signature.

03. If documenting a function, a description of the function’s
    “collectivity”.

    - `Not Collective` if the function need not be called on multiple (or possibly all) MPI
      processes
    - `Collective` if the function is a collective operation.
    - `Logically Collective; yyy must contain common value]`
      if the function is collective but does not require any actual
      synchronization (e.g., setting class parameters uniformly). Any
      argument yyy, which must have the same value on all ranks of the
      MPI communicator should be noted here.

04. If the function is not supported in Fortran, then after the collective information, on the same line,
    one should provide `; No Fortran support`.

05. If documenting a function with input parameters, a list of input
    parameter descriptions in an `Input Parameter(s):` section.

06. If documenting a function with output parameters, a list of output
    parameter descriptions in an `Output Parameter(s):` section.

07. If any input or output parameters are function pointers, they should be documented in the style

    ```console
    Calling sequence of `func()`:
    . arg - the integer argument description
    ```

08. If documenting a function that interacts with the options database, a
    list of options database keys in an `Options Database Key(s):`
    section.

09. `Level:` (no newline) followed by `beginner`,
    `intermediate`, `advanced`, `developer`, or `deprecated`. This
    should be listed before the various `Note(s):` sections.

10. (Optional) a `Note(s):` section containing in-depth discussion,
    technical caveats, special cases, and so on. If it is ambiguous
    whether returned pointers/objects need to be freed/destroyed by the
    user or not, this information should be mentioned here.

11. (If applicable) a `Fortran Note(s):` section detailing any relevant
    differences in calling or using the item from Fortran.

12. (If applicable) a `Developer Note(s):` section detailing any relevant
    information about the code for developers, for example, why a
    particular algorithm was implemented.

13. (If applicable) references should be indicated inline with \{cite}\`Bibtex-key\` where
    Bibtex-key is in the file `doc/petsc.bib`, as in the manual page for `PCFIELDSPLIT`.

14. `.seealso:` (no newline, no spaces to the left of this text), followed by a list of related manual
    pages. These manual pages should usually also point back to this
    manual page in their `.seealso:` sections. This is the final entry in the
    comment. There should be no blank line after the `.seealso:` items.

15. All PETSc functions that appear in a manual page (except the one in the header at the top) should end with a `()` and be enclosed
    in single back tick marks. All PETSc enum types and macros etc, should also be enclosed in single back tick marks.
    This includes each item listed in the `.seealso:` lines.

[^footnote-1]: Type also refers to the string name of the subclass.

### Spelling and Capitalization

1. Proper nouns, including Unix, Linux, X Windows, and Microsoft Windows, should be fully written and capitalized. This includes all operating systems.
   The Apple computer operating system is written as macOS.
2. Company names and product names should be capitalized.
3. Company names and terms that are traditionally all capitalized, for example, MATLAB, NVIDIA, and CUDA, should be all capitalized.
4. ARM is a family of processor designs, while Arm is the company that licenses them.
5. Unix should not be all capitalized.
6. Microsoft Windows should always be written out with two words. That is, it should not be shortened to Windows or MS Win etc.
7. CMake should be capitalized as shown.
8. BLAS and LAPACK are written in full capitalization.
9. Open MPI is written as two words.

## References

```{bibliography} /petsc.bib
:filter: docname in docnames
```
