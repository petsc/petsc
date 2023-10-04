#include <petscsys.h>

typedef int testType;

/*@C
  testWellFormedFunctionDocString - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua.

  Not Collective, Synchronous

  Input Parameters:
+ viewer - a PetscViewer
- x      - an int

  Output Parameter:
+ viewer2 - a PetscViewer
- y       - a pointer

  Level: beginner

  Notes:
  Lorem ipsum dolor sit amet, for example\:

  References:
  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso: `testIllFormedFunctionDocString()`, `testType`
C@*/
PetscErrorCode testWellFormedFunctionDocString(PetscViewer viewer, PetscInt x, PetscViewer viewer2, PetscScalar *y)
{
  return 0;
}

/*@C Lorem ipsum dolor sit amet
  someOtherFunctionName - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
  eiusmod tempor incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat
  non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut
  perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
  totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae
  vitae dicta sunt explicabo.

  Not Collective, Synchronous

   Input Parameters:
+ viewer - a PetscViewer

  Output Parameter:
- y          - a pointer
+ cnd           - a boolean
. z - a nonexistent parameter

  level: Lorem ipsum dolor sit amet

  Level:
  Beginner

  Developer Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum as follows:

  Notes Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident,
  sunt in culpa qui officia deserunt mollit anim id est laborum example.

  Fortran Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum instance:

  References: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso:                                                  testNonExistentFunction(), testNonExistentType,
testIllFormedFunctionDocString(), `testNonExistentFunction()`, testIllFormedMinimalDocString()
@*/

PetscErrorCode testIllFormedFunctionDocString(PetscViewer viewer, PetscInt x, PetscScalar *y, PetscBool cond)
{
  return 0;
}

/*
  Not Collective, Synchronous

  input parms:
. foo

  Output params:
+ bar -

  References:
  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
   .seealso: testNonExistentFunction(), testNonExistentType,`testNonExistentFunction()
*/
PetscErrorCode testIllFormedMinimalDocString(void)
{
  return 0;
}

/*@C
  testTerbleSpelingDocstring - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
  eiusmod tempor incididunt ut labore et dolore magna aliqua.

  input prametirs:
+ viewer - a PetsViewer
- x - a PetscInt

  output Psrammmetrs:
. y - a PetscScalar pointer

  optnS dtaaSE:
- -option_a     - foo
- -option_b [filename][:[~]<foo,bar,baz>[:[~]bop]] - descr
  lvl: itnmediate

.zeeakso:
C@*/
PetscErrorCode testTerribleSpellingDocString(PetscViewer viewer, PetscInt x, PetscScalar *y)
{
  return 0;
}

/*@ asdadsadasdas
  testCustomFortranInterfaceDocString - Lorem ipsum dolor sit amet, consectetur adipiscing elit

  Input Parameters:
+ string -  a char pointer
- function_ptr - a function pointer

  Level:

.seealso: Lorem(), `ipsum()`, dolor(), `sit(), `amet()`, consectetur(), adipiscing(), elit()`
@*/
PetscErrorCode testCustomFortranInterfaceDocString(char *******string, PetscErrorCode (*function_ptr)(PetscInt))
{
  return 0;
}

/* a random comment above a function */
void function() { }

PETSC_INTERN PetscErrorCode testInternFunction();

/*@
  testInternFunction - an internal function

  Level: developer

.seealso: function()
@*/
PetscErrorCode testInternFunction()
{
  return 0;
}

/*@
  testStaticFunction - an internal function

  Level: developer

.seealso: function()
@*/
static PetscErrorCode testStaticFunction()
{
  return 0;
}

/*@
  testAllParamsUndocumented - lorem

  Level: beginner developer

  Example Usage:
.vb
  int a;
  double multiline;
  char codeBlock;
.ve

.seealso:
@*/
PetscErrorCode testAllParamsUndocumented(PetscInt a, PetscInt b)
{
  return testStaticFunction();
}

/*@
  testParameterGrouping ipsum

  Input parameters:
- a,b - some params
+ nonExistentParam - this param does not exist
. ... - variadic arguments

  Level dev

.see also: testStaticFunction()
@*/
PetscErrorCode testParameterGrouping(PetscInt a, PetscInt b, ...)
{
  return 0;
}

/*@
  testScatteredVerbatimBlocks - bla

  Input Parameters:
+ alpha - an alpha
.vb
  int a_code_block;
.ve
- beta - a beta

  Level: beginner

.seealso: `Foo()`
@*/
PetscErrorCode testScatteredVerbatimBlocks(PetscInt alpha, PetscInt beta)
{
  return 0;
}

/*@
  testBadParamListDescrSep - foo

  Input Parameters:
+ alpha, an alpha
- beta = a beta

  Level: beginner

.seealso: Foo()
@*/
PetscErrorCode testBadParamListDescrSep(PetscInt alpha, PetscInt beta)
{
  return 0;
}

/*@
  testBadMidSentenceColons - Lorem:

  Notes:
  Lorem ipsum dolor sit amet:, consectetur adipiscing elit: sed do: eiusmod tempor: incididunt ut
  labore et dolore: magna aliqua: Excepteur: sint occaecat cupidatat non proident, sunt: in culpa
  qui officia: deserunt mollit: anim id est: laborum as follows:

  Level: beginner

.seealso: `Foo()
@*/
PetscErrorCode testBadMidSentenceColons(void)
{
  return 0;
}

/*MC
  MYTYPE - MYTYPE = "mytype"

  Level: developer

.seealso: MATAIJ
MC*/

PetscErrorCode testFloatingDocstring(void)
{
  return 0;
}

/*@M
  testExplicitSynopsis - Lorem Ipsum

  Synopsis:
  #include "testheader.h"
  PetscErrorCode testExplicitSynopsis(PetscInt foo, PetscReal bar, void *baz)

  Collective

  Input Parameters:
+ foo - a foo
- bar - a bar

  Output Parameter:
. baz -                 a baz

  Level: beginner

.seealso: `testExplicitSynopsisBad()`
M@*/
PetscErrorCode testExplicitSynopsis_Private(PetscScalar unknown, PetscInt foo, PetscReal bar, void *baz)
{
  return 0;
}

/* testBadDocString - asdadsasd
*/
PetscErrorCode testBadDocString(PetscInt n)
{
  return 0;
}

/*C testBadDocStringMissingChar - asdadsasd

  Input Parameter:
. n - the n

  Level: beginner

.seealso: `testBadDocString()`
*/
PetscErrorCode testBadDocStringMissingChar(PetscInt n)
{
  return 0;
}

/*C@
  testBadDocStringCharOutOfOrder - asdadsasd

  Input Parameter:
. n - the n

  Level: beginner

.seealso: `testBadDocString()`
*/
PetscErrorCode testBadDocStringCharOutOfOrder(PetscInt n)
{
  return 0;
}

/*
  testInternalLinkageDocstring = This looks like a docstring, acts like a docstring, but it is
  not a docstring, no diagnostics should be emitted for this function

  Input parm:
  * hello - asdasd

  Notes:
  This is because the function has internal linkage, and hence it is developer documentation!

  level: dev

.seealso testWellFormedFunctionDocString()
*/
static PetscErrorCode testInternalLinkageDocstring(PetscInt param)
{
  return 0;
}

/*@
  testSingleFunctionArgNotFound - asdasdasdasd

  Input parm:
+ unrelated - A function arg

  lvel: dev

.seealso: `testBadDocString()

@*/
PetscErrorCode testSingleFunctionArgNotFound(PetscScalar some_function_arg)
{
  return 0;
}

extern PetscErrorCode testPredeclarationCursorIgnored(int, int *);

/*
  testPredeclarationCursorIgnored - the cursor above this will be ignored!

  Inp Paramet:
. asdasd - an arg

  Ouput Pameter:
. another_arg_asd22 - another arg

  Level: beg

.seealso: testPredeclarationCursorIgnored()`, foo()`, `Bar, baz()
*/

PetscErrorCode testPredeclarationCursorIgnored(int arg, int *another_arg)
{
  return 0;
}

/*@
  testFunctionPointerArguments - the first set of arguments are unnamed and should be errored

  Input Parameters:
+ foo - a foo
. bar - a bar
- baz - a baz

  Calling sequence of `foo`:
+ foo_parm1 - an int
. foo_parm2 - a double
- foo_parm3 - a float

  Calling sequence of `bar`:
+ bar_parm1 - an int
. bar_parm2 - a double
- bar_parm3 - a float

  Calling sequence of `baz`:
+ bop       - a bop
. blitz     - a blitz
. baz_parm1 - an int
. baz_parm2 - a double
- baz_parm3 - a float

  Level: developer

  Notes:
  But bars arguments should correctly match! Additionally, this function requires a 'C'
  interface marker!

.seealso: `testPredeclarationCursorIgnored()`
*/
PetscErrorCode testFunctionPointerArguments(int (*foo)(int, double, float), int (*bar)(int bar_parm1, double bar_parm2, float bar_parm3), void (*baz)(int (*bop)(void), void (*blitz)(void (*)(void)), int baz_parm1, double baz_parm2, float baz_parm3))
{
  return 0;
}

/*@
  testDeprecated - check that deprecated (since VERSION) works

  Level: deprecated (since 3.17)

.seealso: `testIllFormedDeprecated()`
*/
PetscErrorCode testDeprecated(void)
{
  return PETSC_SUCCESS;
}

/*@
  testIllFormedDeprecated - check that deprecated (since VERSION) works

  Input Parameters:
+ foo - a nonexistent foo
. bar - a nonexistent bar
- baz - a nonexistent baz

  Level: dpcrtd (since 3.18.5)

.seealso: [](ch_matrices), `testDeprecated()`, [Matrix Factorization](sec_matfactor)
*/
PetscErrorCode testIllFormedDeprecated(void)
{
  return PETSC_SUCCESS;
}

/*@
  testValidInOutParams - check that in-out params work

  Input Parameter:
. foo - the input description for an in-out param

  Output Parameter:
. foo - the output description for an in-out param

  Level: beginner

.seealso: `testWellFormedFunctionDocString()`
*/
PetscErrorCode testValidInOutParams(int *foo)
{
  return PETSC_SUCCESS;
}

/*@
  testInvalidInOutParams - check that in-out params work

  Input Parameter:
+ foo - the input description for an in-out param
+ baz - asdasdasd
- foo              - a duplicate description

  Output Parameters:
. bop = asdas
- foo    - the output description for an in-out param
- foo - a duplicate description2

  Level: beginner

.seealso: `testWellFormedFunctionDocString()`
*/
PetscErrorCode testInvalidInOutParams(int *foo)
{
  return PETSC_SUCCESS;
}

/*@C
  testFunctionParmsSameName - Sets the residual evaluation routine for least-square applications

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context
. res  - the res
. func - the residual evaluation routine
- ctx  - [optional] user-defined context for private data for the function evaluation
         routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. f   - function value vector
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSetObjective()`, `TaoSetJacobianRoutine()`
@*/
PetscErrorCode testFunctionParmsSameName(int tao, double res, PetscErrorCode (*func)(int tao, double x, double f, void *ctx), void *ctx)
{
  return PETSC_SUCCESS;
}

/*@C
  testFunctionParmsSameNameInOut - Sets the residual evaluation routine for least-square applications

  Logically Collective

  Input Parameters:
+ tao  - the `Tao` context (and in-out parm)
. res  - the res
. func - the residual evaluation routine
- ctx  - [optional] user-defined context for private data for the function evaluation
         routine (may be `NULL`)

  Calling sequence of `func`:
+ tao - the optimizer
. x   - input vector
. f   - function value vector
- ctx - [optional] user-defined function context

  Output Parameter:
. tao - the in-output parm

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoSetObjective()`, `TaoSetJacobianRoutine()`
@*/
PetscErrorCode testFunctionParmsSameNameInOut(int *tao, double res, PetscErrorCode (*func)(int tao, double x, double f, void *ctx), void *ctx)
{
  return PETSC_SUCCESS;
}

// PetscClangLinter pragma disable: -fdoc.*
/*@C Lorem ipsum dolor sit amet
  someOtherFunctionName - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
  eiusmod tempor incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat
  non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut
  perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
  totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae
  vitae dicta sunt explicabo.

  Not Collective, Synchronous

   Input Parameters:
+ viewer - a PetscViewer

  Output Parameter:
- y          - a pointer
+ cnd           - a boolean
. z - a nonexistent parameter

  level: Lorem ipsum dolor sit amet

  Level:
  Beginner

  Developer Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum as follows:

  Notes Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident,
  sunt in culpa qui officia deserunt mollit anim id est laborum example.

  Fortran Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum instance:

  References: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso:                                                  testNonExistentFunction(), testNonExistentType,
testIllFormedFunctionDocString(), `testNonExistentFunction()`, testIllFormedMinimalDocString()
@*/

PetscErrorCode testBadDocStringIgnoreAll()
{
  return PETSC_SUCCESS;
}

// A dummy comment
/* another dummy comment */
/*@C
  testIgnoringSpuriousComments - Insert point coordinates (defined over the reference cell)
  within each cell

  Level: beginner

  Notes:
  This verbatim section contains nested comments, but that's OK!
.vb
  // a nested comment
.ve

.seealso: `testBadDocString()`
@*/
PetscErrorCode testIgnoringSpuriousComments()
{
  return PETSC_SUCCESS;
}

/*@C
  testCheckingSectionIndentationAfterSwitch - the second section heading should be properly
  re-indented

  Input Parameter:
. foo - a foo

    OutputParameter:
    +     bar         -    a bar for
    example this is not a heading

  Level: intermediate

.seealso: `testBadDocString()`
@*/
PetscErrorCode testCheckingSectionIndentationAfterSwitch(int foo, double *bar)
{
  return PETSC_SUCCESS;
}

/*@C
  testReferencesFalsePositive - this should not pick up a references section

  Level: beginner

  Notes:
  A per- MPI communicator garbage dictionary is created to store
  references to objects destroyed using `PetscObjectDelayedDestroy()`.

.seealso: `testBadDocString()`
@*/
PetscErrorCode testReferencesFalsePositive()
{
  return PETSC_SUCCESS;
}

/*@C
  testOptionsDatabaseFalsePositive - this should not pick up an
  options database section

  Level: beginner

  Notes:
  A per- MPI communicator garbage dictionary is created to store an
  options database to objects destroyed using `PetscObjectDelayedDestroy()`.

.seealso: `testBadDocString()`
@*/
PetscErrorCode testOptionsDatabaseFalsePositive()
{
  return PETSC_SUCCESS;
}

/*@C
  testLeftFlushSeeAlsoFalsePositive - this should only indent the seealso once

  Level: beginner

.seealso:`thisShouldOnlyBeShiftedOverByOneSpace()`,
`andThisShouldBeLeftAlone()`
@*/
PetscErrorCode testLeftFlushSeeAlsoFalsePositive()
{
  return PETSC_SUCCESS;
}

/*@C
  testNoteFalsePositive - this is note a notes heading, note
  that there

  Level: beginner

.seealso: `Foo`
@*/
PetscErrorCode testNoteFalsePositive()
{
  return PETSC_SUCCESS;
}
