#pragma once

#include <petscmat.h>
#include <petsctaotypes.h>

/* MANSEC = Tao */
/* SUBMANSEC = TaoTerm */

/*S
  TaoTerm - Abstract PETSc object that manages individual terms whose sum forms the objective function of optimization solvers.

  Level: beginner

  Note:
  User can combine a user-defined `TaoTerm` using `TaoTermCreateShell()` and built-in `TaoTerm` to define the objective function.

.seealso: [](ch_tao), [](sec_tao_term),
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermView()`,
          `TaoTermComputeObjective()`,
          `TaoTermComputeGradient()`,
          `TaoTermComputeObjectiveAndGradient()`,
          `TaoTermComputeHessian()`,
          `TaoTermDestroy()`,
          `TaoTermCreateShell()`,
          `Tao`,
          `TaoAddTerm()`
S*/
typedef struct _p_TaoTerm *TaoTerm;

/*J
  TaoTermType - String with the name of a `TaoTerm` method

  Values:
+ `TAOTERMSHELL`         - uses user-provided callback functions, see `TaoTermCreateShell()`
. `TAOTERMSUM`           - a sum of multiple other `TaoTerm`s
. `TAOTERMHALFL2SQUARED` - $\tfrac{1}{2}\|x - p\|_2^2$
. `TAOTERML1`            - $\|x - p\|_1$
. `TAOTERMQUADRATIC`     - a quadratic form $\tfrac{1}{2}(x - p)^T A (x - p)$
- `TAOTERMCALLBACKS`     - uses the callback functions set in `TaoSetObjective()`, `TaoSetGradient()`, etc.

  Level: intermediate

.seealso: [](ch_tao), [](sec_tao_term), `TaoTerm`, `TaoTermCreate()`, `TaoTermSetType()`, `TaoTermCreateShell()`
J*/
typedef const char *TaoTermType;
#define TAOTERMCALLBACKS     "callbacks"
#define TAOTERMSHELL         "shell"
#define TAOTERMSUM           "sum"
#define TAOTERMHALFL2SQUARED "halfl2squared"
#define TAOTERML1            "l1"
#define TAOTERMQUADRATIC     "quadratic"

PETSC_EXTERN PetscErrorCode TaoTermRegister(const char[], PetscErrorCode (*)(TaoTerm));

PETSC_EXTERN PetscClassId      TAOTERM_CLASSID;
PETSC_EXTERN PetscFunctionList TaoTermList;

PETSC_EXTERN PetscErrorCode TaoTermCreate(MPI_Comm, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermDestroy(TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermView(TaoTerm, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoTermSetUp(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermGetType(TaoTerm, TaoTermType *);
PETSC_EXTERN PetscErrorCode TaoTermSetType(TaoTerm, TaoTermType);
PETSC_EXTERN PetscErrorCode TaoTermSetFromOptions(TaoTerm);

/*E
  TaoTermParametersMode - Ways a `TaoTerm` can accept parameter vectors in `TaoTermComputeObjective()` and related functions

  Values:
+ `TAOTERM_PARAMETERS_OPTIONAL` - the term has default parameters that will be used if parameters are omitted
. `TAOTERM_PARAMETERS_NONE`     - the term is not parametric, passing parameters is an error
- `TAOTERM_PARAMETERS_REQUIRED` - the term requires parameters, omitting parameters is an error

  Level: intermediate

  Note:
  Each `TaoTerm` represents a parametric real-valued function $f(x; p)$, where $x$ is the
  solution variable (the optimization variable) and $p$ is a parameter vector of fixed data
  that is not optimized over.  The solution space (the vector space of $x$) and the parameter
  space (the vector space of $p$) are set independently; see `TaoTermSetSolutionSizes()` and
  `TaoTermSetParametersSizes()`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermGetParametersMode()`, `TaoTermSetParametersMode()`
E*/
typedef enum {
  TAOTERM_PARAMETERS_OPTIONAL,
  TAOTERM_PARAMETERS_NONE,
  TAOTERM_PARAMETERS_REQUIRED
} TaoTermParametersMode;
PETSC_EXTERN const char *const TaoTermParametersModes[];

/*E
  TaoTermMask - Determine which evaluation operations are masked; that is, skipped (not used) by Tao when computing the objective function or its derivatives for a particular `TaoTerm`.

  Values:
+ `TAOTERM_MASK_NONE`      - do not mask any evaluation routines
. `TAOTERM_MASK_OBJECTIVE` - override the term's objective function and return 0 instead
. `TAOTERM_MASK_GRADIENT`  - override the term's gradient and return a zero vector instead
- `TAOTERM_MASK_HESSIAN`   - override the term's Hessian and return a zero matrix instead

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermSumSetTermMask()`, `TaoTermSumGetTermMask()`
E*/
typedef enum {
  TAOTERM_MASK_NONE      = 0, /* 0x0 */
  TAOTERM_MASK_OBJECTIVE = 1, /* 0x1 */
  TAOTERM_MASK_GRADIENT  = 2, /* 0x2 */
  TAOTERM_MASK_HESSIAN   = 4  /* 0x4 */
} TaoTermMask;

PETSC_EXTERN PetscErrorCode TaoTermSetParametersMode(TaoTerm, TaoTermParametersMode);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersMode(TaoTerm, TaoTermParametersMode *);

/*E
  TaoTermDuplicateOption - Aspects to preserve when duplicating a `TaoTerm`

  Values:
+ `TAOTERM_DUPLICATE_SIZEONLY` - duplicates size of the solution space only; user must set appropriate `TaoTermType`
- `TAOTERM_DUPLICATE_TYPE`     - `TaoTermType` preserved

  Level: intermediate

.seealso: `TaoTerm`, `TaoTermDuplicate()`
E*/
typedef enum {
  TAOTERM_DUPLICATE_SIZEONLY,
  TAOTERM_DUPLICATE_TYPE
} TaoTermDuplicateOption;

PETSC_EXTERN PetscErrorCode TaoTermDuplicate(TaoTerm, TaoTermDuplicateOption, TaoTerm *);

PETSC_EXTERN PetscErrorCode TaoTermSetSolutionSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetSolutionSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetSolutionVecType(TaoTerm, VecType);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersVecType(TaoTerm, VecType);
PETSC_EXTERN PetscErrorCode TaoTermGetSolutionVecType(TaoTerm, VecType *);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersVecType(TaoTerm, VecType *);
PETSC_EXTERN PetscErrorCode TaoTermSetSolutionLayout(TaoTerm, PetscLayout);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersLayout(TaoTerm, PetscLayout);
PETSC_EXTERN PetscErrorCode TaoTermGetSolutionLayout(TaoTerm, PetscLayout *);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersLayout(TaoTerm, PetscLayout *);
PETSC_EXTERN PetscErrorCode TaoTermCreateSolutionVec(TaoTerm, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermCreateParametersVec(TaoTerm, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatricesDefault(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSetCreateHessianMode(TaoTerm, PetscBool, MatType, MatType);
PETSC_EXTERN PetscErrorCode TaoTermGetCreateHessianMode(TaoTerm, PetscBool *, MatType *, MatType *);

/*S
  TaoTermObjectiveFn - A prototype of a `TaoTerm` function that would be passed to `TaoTermShellSetObjective()`

  Calling Sequence:
+ term   - a `TaoTerm`
. x      - the solution vector
. params - the parameters vector (for some `TaoTerm` this may be `NULL`, see `TaoTermGetParametersMode()`)
- value  - output, the value of the term

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSHELL`,
          `TaoTermShellSetObjective()`,
          `TaoTermObjectiveAndGradientFn`,
          `TaoTermGradientFn`,
          `TaoTermHessianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(TaoTermObjectiveFn)(TaoTerm term, Vec x, Vec params, PetscReal *value);

/*S
  TaoTermObjectiveAndGradientFn - A prototype of a `TaoTerm` function that would be passed to `TaoTermShellSetObjectiveAndGradient()`

  Calling Sequence:
+ term   - a `TaoTerm`
. x      - the solution vector
. params - the parameters vector (for some `TaoTerm` this may be `NULL`, see `TaoTermGetParametersMode()`)
. value  - output, the value of the term
- g      - output, the gradient of the term

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSHELL`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermObjectiveFn`,
          `TaoTermGradientFn`,
          `TaoTermHessianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(TaoTermObjectiveAndGradientFn)(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g);

/*S
  TaoTermGradientFn - A prototype of a `TaoTerm` function that would be passed to `TaoTermShellSetGradient()`

  Calling Sequence:
+ term   - a `TaoTerm`
. x      - the solution vector
. params - the parameters vector (for some `TaoTerm` this may be `NULL`, see `TaoTermGetParametersMode()`)
- g      - output, the gradient of the term

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSHELL`,
          `TaoTermShellSetGradient()`,
          `TaoTermObjectiveFn`,
          `TaoTermObjectiveAndGradientFn`,
          `TaoTermHessianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(TaoTermGradientFn)(TaoTerm term, Vec x, Vec params, Vec g);

/*S
  TaoTermHessianFn - A prototype of a `TaoTerm` function that would be passed to `TaoTermShellSetHessian()`

  Calling Sequence:
+ term   - a `TaoTerm`
. x      - the solution vector
. params - the parameters vector (for some `TaoTerm` this may be `NULL`, see `TaoTermGetParametersMode()`)
. H      - (optional) output, the Hessian of `term`
- Hpre   - (optional) output, the approximation of `H` from which a preconditioner may be built

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMSHELL`,
          `TaoTermShellSetHessian()`,
          `TaoTermObjectiveFn`,
          `TaoTermObjectiveAndGradientFn`,
          `TaoTermGradientFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(TaoTermHessianFn)(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre);

PETSC_EXTERN PetscErrorCode TaoTermComputeObjective(TaoTerm, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermComputeGradient(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermComputeObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoTermComputeHessian(TaoTerm, Vec, Vec, Mat, Mat);

PETSC_EXTERN PetscErrorCode TaoTermCreateShell(MPI_Comm, PetscCtx, PetscCtxDestroyFn *, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContext(TaoTerm, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoTermShellGetContext(TaoTerm, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContextDestroy(TaoTerm, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjective(TaoTerm, TaoTermObjectiveFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetGradient(TaoTerm, TaoTermGradientFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjectiveAndGradient(TaoTerm, TaoTermObjectiveAndGradientFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetHessian(TaoTerm, TaoTermHessianFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetView(TaoTerm, PetscErrorCode (*)(TaoTerm, PetscViewer));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateSolutionVec(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateParametersVec(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateHessianMatrices(TaoTerm, PetscErrorCode (*)(TaoTerm, Mat *, Mat *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetIsComputeHessianFDPossible(TaoTerm, PetscBool3);

PETSC_EXTERN PetscErrorCode TaoTermSumSetNumberTerms(TaoTerm, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermSumGetNumberTerms(TaoTerm, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetTerm(TaoTerm, PetscInt, const char[], PetscReal, TaoTerm, Mat);
PETSC_EXTERN PetscErrorCode TaoTermSumGetTerm(TaoTerm, PetscInt, const char **, PetscReal *, TaoTerm *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSumAddTerm(TaoTerm, const char[], PetscReal, TaoTerm, Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumParametersPack(TaoTerm, Vec[], Vec *);
PETSC_EXTERN PetscErrorCode TaoTermSumParametersUnpack(TaoTerm, Vec *, Vec[]);
PETSC_EXTERN PetscErrorCode VecNestGetTaoTermSumParameters(Vec, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermSumGetTermHessianMatrices(TaoTerm, PetscInt, Mat *, Mat *, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetTermHessianMatrices(TaoTerm, PetscInt, Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermSumGetTermMask(TaoTerm, PetscInt, TaoTermMask *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetTermMask(TaoTerm, PetscInt, TaoTermMask);
PETSC_EXTERN PetscErrorCode TaoTermSumGetLastTermObjectives(TaoTerm, const PetscReal *[]);

PETSC_EXTERN PetscErrorCode TaoTermCreateHalfL2Squared(MPI_Comm, PetscInt, PetscInt, TaoTerm *);

PETSC_EXTERN PetscErrorCode TaoTermCreateL1(MPI_Comm, PetscInt, PetscInt, PetscReal, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermL1SetEpsilon(TaoTerm, PetscReal);
PETSC_EXTERN PetscErrorCode TaoTermL1GetEpsilon(TaoTerm, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoTermCreateQuadratic(Mat, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermQuadraticGetMat(TaoTerm, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermQuadraticSetMat(TaoTerm, Mat);

PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsHessianDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsComputeHessianFDPossible(TaoTerm, PetscBool3 *);

PETSC_EXTERN PetscErrorCode TaoTermGetFDDelta(TaoTerm, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermSetFDDelta(TaoTerm, PetscReal);
PETSC_EXTERN PetscErrorCode TaoTermComputeGradientFD(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermComputeGradientSetUseFD(TaoTerm, PetscBool);
PETSC_EXTERN PetscErrorCode TaoTermComputeGradientGetUseFD(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermComputeHessianFD(TaoTerm, Vec, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermComputeHessianSetUseFD(TaoTerm, PetscBool);
PETSC_EXTERN PetscErrorCode TaoTermComputeHessianGetUseFD(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMFFD(TaoTerm, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermComputeHessianMFFD(TaoTerm, Vec, Vec, Mat, Mat);
