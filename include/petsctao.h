#pragma once

#include <petscsnes.h>

/* SUBMANSEC = Tao */

PETSC_EXTERN PetscErrorCode MatDSFischer(Mat, Vec, Vec, Vec, Vec, PetscReal, Vec, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoSoftThreshold(Vec, PetscReal, PetscReal, Vec);

/*E
  TaoSubsetType - Type representing the way the `Tao` solvers handle active sets

  Values:
+ `TAO_SUBSET_SUBVEC`     - Tao uses `MatCreateSubMatrix()` and `VecGetSubVector()`
. `TAO_SUBSET_MASK`       - Matrices are zeroed out corresponding to active set entries
- `TAO_SUBSET_MATRIXFREE` - Same as `TAO_SUBSET_MASK` but it can be applied to matrix-free operators

  Options database Key:
. -different_hessian - `Tao` will use a copy of the Hessian operator for masking.  By default `Tao` will directly alter the Hessian operator.

  Level: intermediate

.seealso: [](ch_tao), `TaoVecGetSubVec()`, `TaoMatGetSubMat()`, `Tao`, `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
E*/
typedef enum {
  TAO_SUBSET_SUBVEC,
  TAO_SUBSET_MASK,
  TAO_SUBSET_MATRIXFREE
} TaoSubsetType;
PETSC_EXTERN const char *const TaoSubsetTypes[];

/*S
   Tao - Abstract PETSc object that manages optimization solvers.

   Level: advanced

   Note:
   `Tao` is the object, while TAO, which stands for Toolkit for Advanced Optimization, is the software package.

.seealso: [](doc_taosolve), [](ch_tao), `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
S*/
typedef struct _p_Tao *Tao;

/*E
  TaoADMMUpdateType - Determine the spectral penalty update routine for the Lagrange augmented term for `TAOADMM`.

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoADMMSetUpdateType()`
E*/
typedef enum {
  TAO_ADMM_UPDATE_BASIC,
  TAO_ADMM_UPDATE_ADAPTIVE,
  TAO_ADMM_UPDATE_ADAPTIVE_RELAXED
} TaoADMMUpdateType;
PETSC_EXTERN const char *const TaoADMMUpdateTypes[];

/*MC
  TAO_ADMM_UPDATE_BASIC - Use same spectral penalty set at the beginning. This never performs an update to the penalty

  Level: advanced

  Note:
  Most basic implementation of `TAOADMM`. Generally slower than adaptive or adaptive relaxed version.

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoADMMSetUpdateType()`, `TAO_ADMM_UPDATE_ADAPTIVE`, `TAO_ADMM_UPDATE_ADAPTIVE_RELAXED`
M*/

/*MC
  TAO_ADMM_UPDATE_ADAPTIVE - Adaptively update the spectral penalty

  Level: advanced

  Note:
  Adaptively updates spectral penalty of `TAOADMM` by using both steepest descent and minimum gradient.

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoADMMSetUpdateType()`, `TAO_ADMM_UPDATE_BASIC`, `TAO_ADMM_UPDATE_ADAPTIVE_RELAXED`
M*/

/*MC
  ADMM_UPDATE_ADAPTIVE_RELAXED - Adaptively update spectral penalty, and relaxes parameter update

  Level: advanced

  Note:
  With adaptive spectral penalty update, it also relaxes the `x` vector update by a factor.

.seealso: [](ch_tao), `Tao`, `TaoADMMSetUpdateType()`, `TAO_ADMM_UPDATE_BASIC`, `TAO_ADMM_UPDATE_ADAPTIVE`
M*/

/*E
  TaoADMMRegularizerType - Determine regularizer routine - either user provided or soft threshold for `TAOADMM`

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoADMMSetRegularizerType()`
E*/
typedef enum {
  TAO_ADMM_REGULARIZER_USER,
  TAO_ADMM_REGULARIZER_SOFT_THRESH
} TaoADMMRegularizerType;
PETSC_EXTERN const char *const TaoADMMRegularizerTypes[];

/*MC
  TAO_ADMM_REGULARIZER_USER - User provided routines for regularizer part of `TAOADMM`

  Level: advanced

  Note:
  User needs to provided appropriate routines and type for regularizer solver

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoADMMSetRegularizerType()`, `TAO_ADMM_REGULARIZER_SOFT_THRESH`
M*/

/*MC
  TAO_ADMM_REGULARIZER_SOFT_THRESH - Soft threshold to solve regularizer part of `TAOADMM`

  Level: advanced

  Note:
  Utilizes built-in SoftThreshold routines

.seealso: [](ch_tao), `Tao`, `TAOADMM`, `TaoSoftThreshold()`, `TaoADMMSetRegularizerObjectiveAndGradientRoutine()`,
          `TaoADMMSetRegularizerHessianRoutine()`, `TaoADMMSetRegularizerType()`, `TAO_ADMM_REGULARIZER_USER`
M*/

/*E
   TaoALMMType - Determine the augmented Lagrangian formulation used in the `TAOALMM` subproblem.

   Values:
+  `TAO_ALMM_CLASSIC` - classic augmented Lagrangian definition including slack variables for inequality constraints
-  `TAO_ALMM_PHR`     - Powell-Hestenes-Rockafellar formulation without slack variables, uses pointwise `min()` for inequalities

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOALMM`, `TaoALMMSetType()`, `TaoALMMGetType()`
E*/
typedef enum {
  TAO_ALMM_CLASSIC,
  TAO_ALMM_PHR
} TaoALMMType;
PETSC_EXTERN const char *const TaoALMMTypes[];

/*E
  TaoBNCGType - Determine the conjugate gradient update formula used in the `TAOBNCG` algorithm.

  Values:
+  `TAO_BNCG_GD`         - basic gradient descent, no CG update
.  `TAO_BNCG_PCGD`       - preconditioned/scaled gradient descent
.  `TAO_BNCG_HS`         - Hestenes-Stiefel
.  `TAO_BNCG_FR`         - Fletcher-Reeves
.  `TAO_BNCG_PRP`        - Polak-Ribiere-Polyak (PRP)
.  `TAO_BNCG_PRP_PLUS`   - Polak-Ribiere-Polyak "plus" (PRP+)
.  `TAO_BNCG_DY`         - Dai-Yuan
.  `TAO_BNCG_HZ`         - Hager-Zhang (CG_DESCENT 5.3)
.  `TAO_BNCG_DK`         - Dai-Kou (2013)
.  `TAO_BNCG_KD`         - Kou-Dai (2015)
.  `TAO_BNCG_SSML_BFGS`  - Self-Scaling Memoryless BFGS (Perry-Shanno)
.  `TAO_BNCG_SSML_DFP`   - Self-Scaling Memoryless DFP
-  `TAO_BNCG_SSML_BRDN`  - Self-Scaling Memoryless (Symmetric) Broyden

  Level: advanced

.seealso: `Tao`, `TAOBNCG`, `TaoBNCGSetType()`, `TaoBNCGGetType()`
E*/

typedef enum {
  TAO_BNCG_GD,
  TAO_BNCG_PCGD,
  TAO_BNCG_HS,
  TAO_BNCG_FR,
  TAO_BNCG_PRP,
  TAO_BNCG_PRP_PLUS,
  TAO_BNCG_DY,
  TAO_BNCG_HZ,
  TAO_BNCG_DK,
  TAO_BNCG_KD,
  TAO_BNCG_SSML_BFGS,
  TAO_BNCG_SSML_DFP,
  TAO_BNCG_SSML_BRDN
} TaoBNCGType;
PETSC_EXTERN const char *const TaoBNCGTypes[];

/*J
  TaoType - String with the name of a `Tao` method

  Values:
+ `TAONLS`      - nls Newton's method with line search for unconstrained minimization
. `TAONTR`      - ntr Newton's method with trust region for unconstrained minimization
. `TAONTL`      - ntl Newton's method with trust region, line search for unconstrained minimization
. `TAOLMVM`     - lmvm Limited memory variable metric method for unconstrained minimization
. `TAOCG`       - cg Nonlinear conjugate gradient method for unconstrained minimization
. `TAONM`       - nm Nelder-Mead algorithm for derivate-free unconstrained minimization
. `TAOTRON`     - tron Newton Trust Region method for bound constrained minimization
. `TAOGPCG`     - gpcg Newton Trust Region method for quadratic bound constrained minimization
. `TAOBLMVM`    - blmvm Limited memory variable metric method for bound constrained minimization
. `TAOLCL`      - lcl Linearly constrained Lagrangian method for pde-constrained minimization
- `TAOPOUNDERS` - Pounders Model-based algorithm for nonlinear least squares

  Level: beginner

.seealso: [](doc_taosolve), [](ch_tao), `Tao`, `TaoCreate()`, `TaoSetType()`
J*/
typedef const char *TaoType;
#define TAOLMVM     "lmvm"
#define TAONLS      "nls"
#define TAONTR      "ntr"
#define TAONTL      "ntl"
#define TAOCG       "cg"
#define TAOTRON     "tron"
#define TAOOWLQN    "owlqn"
#define TAOBMRM     "bmrm"
#define TAOBLMVM    "blmvm"
#define TAOBQNLS    "bqnls"
#define TAOBNCG     "bncg"
#define TAOBNLS     "bnls"
#define TAOBNTR     "bntr"
#define TAOBNTL     "bntl"
#define TAOBQNKLS   "bqnkls"
#define TAOBQNKTR   "bqnktr"
#define TAOBQNKTL   "bqnktl"
#define TAOBQPIP    "bqpip"
#define TAOGPCG     "gpcg"
#define TAONM       "nm"
#define TAOPOUNDERS "pounders"
#define TAOBRGN     "brgn"
#define TAOLCL      "lcl"
#define TAOSSILS    "ssils"
#define TAOSSFLS    "ssfls"
#define TAOASILS    "asils"
#define TAOASFLS    "asfls"
#define TAOIPM      "ipm"
#define TAOPDIPM    "pdipm"
#define TAOSHELL    "shell"
#define TAOADMM     "admm"
#define TAOALMM     "almm"
#define TAOPYTHON   "python"
#define TAOSNES     "snes"

PETSC_EXTERN PetscClassId      TAO_CLASSID;
PETSC_EXTERN PetscFunctionList TaoList;

/*E
    TaoConvergedReason - reason a `Tao` optimizer was said to have converged or diverged

   Values:
+  `TAO_CONVERGED_GATOL`       - $||g(X)|| < gatol$
.  `TAO_CONVERGED_GRTOL`       - $||g(X)|| / f(X)  < grtol$
.  `TAO_CONVERGED_GTTOL`       - $||g(X)|| / ||g(X0)|| < gttol$
.  `TAO_CONVERGED_STEPTOL`     - step size smaller than tolerance
.  `TAO_CONVERGED_MINF`        - $F < F_min$
.  `TAO_CONVERGED_USER`        - the user indicates the optimization has succeeded
.  `TAO_DIVERGED_MAXITS`       - the maximum number of iterations allowed has been achieved
.  `TAO_DIVERGED_NAN`          - not a number appeared in the computations
.  `TAO_DIVERGED_MAXFCN`       - the maximum number of function evaluations has been computed
.  `TAO_DIVERGED_LS_FAILURE`   - a linesearch failed
.  `TAO_DIVERGED_TR_REDUCTION` - trust region failure
.  `TAO_DIVERGED_USER`         - the user has indicated the optimization has failed
-  `TAO_CONTINUE_ITERATING`    - the optimization is still running, `TaoSolve()`

   where
+  X            - current solution
.  X0           - initial guess
.  f(X)         - current function value
.  f(X*)        - true solution (estimated)
.  g(X)         - current gradient
.  its          - current iterate number
.  maxits       - maximum number of iterates
.  fevals       - number of function evaluations
-  max_funcsals - maximum number of function evaluations

   Level: beginner

   Note:
   The two most common reasons for divergence are  an incorrectly coded or computed gradient or Hessian failure or lack of convergence
   in the linear system solve (in this case we recommend testing with `-pc_type lu` to eliminate the linear solver as the cause of the problem).

   Developer Note:
   The names in `KSPConvergedReason`, `SNESConvergedReason`, and `TaoConvergedReason` should be uniformized

.seealso: [](ch_tao), `Tao`, `TaoSolve()`, `TaoGetConvergedReason()`, `KSPConvergedReason`, `SNESConvergedReason`
E*/
typedef enum {               /* converged */
  TAO_CONVERGED_GATOL   = 3, /* ||g(X)|| < gatol */
  TAO_CONVERGED_GRTOL   = 4, /* ||g(X)|| / f(X)  < grtol */
  TAO_CONVERGED_GTTOL   = 5, /* ||g(X)|| / ||g(X0)|| < gttol */
  TAO_CONVERGED_STEPTOL = 6, /* step size small */
  TAO_CONVERGED_MINF    = 7, /* F < F_min */
  TAO_CONVERGED_USER    = 8, /* User defined */
  /* diverged */
  TAO_DIVERGED_MAXITS       = -2,
  TAO_DIVERGED_NAN          = -4,
  TAO_DIVERGED_MAXFCN       = -5,
  TAO_DIVERGED_LS_FAILURE   = -6,
  TAO_DIVERGED_TR_REDUCTION = -7,
  TAO_DIVERGED_USER         = -8, /* User defined */
  /* keep going */
  TAO_CONTINUE_ITERATING = 0
} TaoConvergedReason;

PETSC_EXTERN const char **TaoConvergedReasons;

PETSC_EXTERN PetscErrorCode TaoInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TaoCreate(MPI_Comm, Tao *);
PETSC_EXTERN PetscErrorCode TaoSetFromOptions(Tao);
PETSC_EXTERN PetscErrorCode TaoSetUp(Tao);
PETSC_EXTERN PetscErrorCode TaoSetType(Tao, TaoType);
PETSC_EXTERN PetscErrorCode TaoGetType(Tao, TaoType *);
PETSC_EXTERN PetscErrorCode TaoSetApplicationContext(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoGetApplicationContext(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoDestroy(Tao *);
PETSC_EXTERN PetscErrorCode TaoParametersInitialize(Tao);

PETSC_EXTERN PetscErrorCode TaoSetOptionsPrefix(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoView(Tao, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoViewFromOptions(Tao, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode TaoSolve(Tao);

PETSC_EXTERN PetscErrorCode TaoRegister(const char[], PetscErrorCode (*)(Tao));
PETSC_EXTERN PetscErrorCode TaoRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TaoGetConvergedReason(Tao, TaoConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoGetSolutionStatus(Tao, PetscInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, TaoConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoSetConvergedReason(Tao, TaoConvergedReason);
PETSC_EXTERN PetscErrorCode TaoSetSolution(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoGetSolution(Tao, Vec *);

PETSC_EXTERN PetscErrorCode TaoSetObjective(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetObjective(Tao, PetscErrorCode (**)(Tao, Vec, PetscReal *, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetGradient(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetGradient(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetObjectiveAndGradient(Tao, Vec, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetObjectiveAndGradient(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, PetscReal *, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetHessian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetHessian(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, void *), void **);

PETSC_EXTERN PetscErrorCode TaoSetGradientNorm(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoGetGradientNorm(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoSetLMVMMatrix(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoGetLMVMMatrix(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoSetRecycleHistory(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoGetRecycleHistory(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoLMVMSetH0(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoLMVMGetH0(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoLMVMGetH0KSP(Tao, KSP *);
PETSC_EXTERN PetscErrorCode TaoLMVMRecycle(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoSetResidualRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetResidualWeights(Tao, Vec, PetscInt, PetscInt *, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetInequalityConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetInequalityConstraintsRoutine(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetEqualityConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetEqualityConstraintsRoutine(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetJacobianResidualRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetJacobianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetJacobianStateRoutine(Tao, Mat, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetJacobianDesignRoutine(Tao, Mat, PetscErrorCode (*)(Tao, Vec, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoSetJacobianInequalityRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetJacobianInequalityRoutine(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, void *), void **);
PETSC_EXTERN PetscErrorCode TaoSetJacobianEqualityRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoGetJacobianEqualityRoutine(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, void *), void **);

PETSC_EXTERN PetscErrorCode TaoPythonSetType(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoPythonGetType(Tao, const char *[]);

PETSC_EXTERN PetscErrorCode TaoShellSetSolve(Tao, PetscErrorCode (*)(Tao));
PETSC_EXTERN PetscErrorCode TaoShellSetContext(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoShellGetContext(Tao, void *);

PETSC_EXTERN PetscErrorCode TaoSetStateDesignIS(Tao, IS, IS);

PETSC_EXTERN PetscErrorCode TaoComputeObjective(Tao, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoComputeResidual(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTestGradient(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeGradient(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeObjectiveAndGradient(Tao, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeInequalityConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeEqualityConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeGradient(Tao, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveDefined(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoIsGradientDefined(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveAndGradientDefined(Tao, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoTestHessian(Tao);
PETSC_EXTERN PetscErrorCode TaoComputeHessian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeResidualJacobian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianState(Tao, Vec, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianEquality(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianInequality(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianDesign(Tao, Vec, Mat);

PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessian(Tao, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessianColor(Tao, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessianMFFD(Tao, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TaoComputeDualVariables(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoSetVariableBounds(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoGetVariableBounds(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoGetDualVariables(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoSetInequalityBounds(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoGetInequalityBounds(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoSetVariableBoundsRoutine(Tao, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoComputeVariableBounds(Tao);

PETSC_EXTERN PetscErrorCode TaoGetTolerances(Tao, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetTolerances(Tao, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoGetConstraintTolerances(Tao, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetConstraintTolerances(Tao, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetFunctionLowerBound(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetInitialTrustRegionRadius(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetMaximumIterations(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoSetMaximumFunctionEvaluations(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetFunctionLowerBound(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetInitialTrustRegionRadius(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetCurrentTrustRegionRadius(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetMaximumIterations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetCurrentFunctionEvaluations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetMaximumFunctionEvaluations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetIterationNumber(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoSetIterationNumber(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetTotalIterationNumber(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoSetTotalIterationNumber(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetResidualNorm(Tao, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoAppendOptionsPrefix(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoGetOptionsPrefix(Tao, const char *[]);
PETSC_EXTERN PetscErrorCode TaoResetStatistics(Tao);
PETSC_EXTERN PetscErrorCode TaoSetUpdate(Tao, PetscErrorCode (*)(Tao, PetscInt, void *), void *);

PETSC_EXTERN PetscErrorCode TaoGetKSP(Tao, KSP *);
PETSC_EXTERN PetscErrorCode TaoGetLinearSolveIterations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoKSPSetUseEW(Tao, PetscBool);

#include <petsctaolinesearch.h>

PETSC_EXTERN PetscErrorCode TaoGetLineSearch(Tao, TaoLineSearch *);

PETSC_EXTERN PetscErrorCode TaoSetConvergenceHistory(Tao, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscInt, PetscBool);
PETSC_EXTERN PetscErrorCode TaoGetConvergenceHistory(Tao, PetscReal **, PetscReal **, PetscReal **, PetscInt **, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoMonitorSet(Tao, PetscErrorCode (*)(Tao, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TaoMonitorCancel(Tao);
PETSC_EXTERN PetscErrorCode TaoMonitorDefault(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorGlobalization(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorDefaultShort(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorConstraintNorm(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorSolution(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorResidual(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorGradient(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorStep(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorSolutionDraw(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorStepDraw(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoMonitorGradientDraw(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoAddLineSearchCounts(Tao);

PETSC_EXTERN PetscErrorCode TaoDefaultConvergenceTest(Tao, void *);
PETSC_EXTERN PetscErrorCode TaoSetConvergenceTest(Tao, PetscErrorCode (*)(Tao, void *), void *);

PETSC_EXTERN PetscErrorCode          TaoLCLSetStateDesignIS(Tao, IS, IS);
PETSC_EXTERN PetscErrorCode          TaoMonitor(Tao, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal);
typedef struct _n_TaoMonitorDrawCtx *TaoMonitorDrawCtx;
PETSC_EXTERN PetscErrorCode          TaoMonitorDrawCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TaoMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode          TaoMonitorDrawCtxDestroy(TaoMonitorDrawCtx *);

/*E
  TaoBRGNRegularizationType - The regularization added in the `TAOBRGN` solver.

  Values:
+ TAOBRGN_REGULARIZATION_USER   - A user-defined regularizer
. TAOBRGN_REGULARIZATION_L2PROX - $\tfrac{1}{2}\|x - x_k\|_2^$, where $x_k$ is the latest solution
. TAOBRGN_REGULARIZATION_L2PURE - $\tfrac{1}{2}\|x\|_2^2$
. TAOBRGN_REGULARIZATION_L1DICT - $\|D x\|_1$, where $D$ is a dictionary matrix
- TAOBRGN_REGULARIZATION_LM     - Levenberg-Marquardt, $\tfrac{1}{2} x^T \mathrm{diag}(J^T J) x$, where $J$ is the Jacobian of the least-squares residual

  Options database Key:
. -tao_brgn_regularization_type <user,l2prox,l2pure,l1dict,lm> - one of the above regularization types

  Level: advanced

  Notes:
  If `TAOBRGN_REGULARIZATION_USER`, the regularizer is set either by calling
  `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()` and
  `TaoBRGNSetRegulazerHessianRoutine()` or by calling `TaoBRGNSetRegularizerTerm()`.

  If `TAOBRGN_REGULARIZATION_L1DICT`, the dictionary matrix is set with `TaoBRGNSetDictionaryMatrix()` and the smoothing parameter of the
  approximate $\ell_1$ norm is set with `TaoBRGNSetL1SmoothEpsilon()`.

  If `TAOBRGN_REGULARIZATION_LM`, the diagonal damping vector $\mathrm{diag}(J^T J)$ can be obtained with `TaoBRGNGetDampingVector()`.

.seealso: [](ch_tao), `Tao`, `TaoBRGNGetSubsolver()`, `TaoBRGNSetRegularizerWeight()`, `TaoBRGNSetL1SmoothEpsilon()`, `TaoBRGNSetDictionaryMatrix()`,
          `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`, `TaoBRGNSetRegularizerHessianRoutine()`,
          `TaoBRGNGetRegularizationType()`, `TaoBRGNSetRegularizationType()`
E*/
typedef enum {
  TAOBRGN_REGULARIZATION_USER,
  TAOBRGN_REGULARIZATION_L2PROX,
  TAOBRGN_REGULARIZATION_L2PURE,
  TAOBRGN_REGULARIZATION_L1DICT,
  TAOBRGN_REGULARIZATION_LM,
} TaoBRGNRegularizationType;

PETSC_EXTERN const char *const TaoBRGNRegularizationTypes[];

PETSC_EXTERN PetscErrorCode TaoBRGNGetSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoBRGNGetRegularizationType(Tao, TaoBRGNRegularizationType *);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizationType(Tao, TaoBRGNRegularizationType);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(Tao, Mat, PetscErrorCode (*)(Tao, Vec, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerWeight(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoBRGNSetL1SmoothEpsilon(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoBRGNGetDampingVector(Tao, Vec *);
PETSC_EXTERN PetscErrorCode TaoBNCGSetType(Tao, TaoBNCGType);
PETSC_EXTERN PetscErrorCode TaoBNCGGetType(Tao, TaoBNCGType *);

PETSC_EXTERN PetscErrorCode TaoADMMGetMisfitSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizationSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMGetDualVector(Tao, Vec *);
PETSC_EXTERN PetscErrorCode TaoADMMGetSpectralPenalty(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoADMMSetSpectralPenalty(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoGetADMMParentTao(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMSetConstraintVectorRHS(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerCoefficient(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizerCoefficient(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitConstraintJacobian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerConstraintJacobian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerHessianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitHessianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitHessianChangeStatus(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegHessianChangeStatus(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoADMMSetMinimumSpectralPenalty(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerType(Tao, TaoADMMRegularizerType);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizerType(Tao, TaoADMMRegularizerType *);
PETSC_EXTERN PetscErrorCode TaoADMMSetUpdateType(Tao, TaoADMMUpdateType);
PETSC_EXTERN PetscErrorCode TaoADMMGetUpdateType(Tao, TaoADMMUpdateType *);

PETSC_EXTERN PetscErrorCode TaoALMMGetType(Tao, TaoALMMType *);
PETSC_EXTERN PetscErrorCode TaoALMMSetType(Tao, TaoALMMType);
PETSC_EXTERN PetscErrorCode TaoALMMGetSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoALMMSetSubsolver(Tao, Tao);
PETSC_EXTERN PetscErrorCode TaoALMMGetMultipliers(Tao, Vec *);
PETSC_EXTERN PetscErrorCode TaoALMMSetMultipliers(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoALMMGetPrimalIS(Tao, IS *, IS *);
PETSC_EXTERN PetscErrorCode TaoALMMGetDualIS(Tao, IS *, IS *);

PETSC_EXTERN PetscErrorCode TaoVecGetSubVec(Vec, IS, TaoSubsetType, PetscReal, Vec *);
PETSC_EXTERN PetscErrorCode TaoMatGetSubMat(Mat, IS, Vec, TaoSubsetType, Mat *);
PETSC_EXTERN PetscErrorCode TaoGradientNorm(Tao, Vec, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoEstimateActiveBounds(Vec, Vec, Vec, Vec, Vec, Vec, PetscReal, PetscReal *, IS *, IS *, IS *, IS *, IS *);
PETSC_EXTERN PetscErrorCode TaoBoundStep(Vec, Vec, Vec, IS, IS, IS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TaoBoundSolution(Vec, Vec, Vec, PetscReal, PetscInt *, Vec);

PETSC_EXTERN PetscErrorCode MatCreateSubMatrixFree(Mat, IS, IS, Mat *);

#include <petsctao_deprecations.h>
