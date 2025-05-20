#pragma once

#include <petscdmlabel.h>

/* MANSEC = DM */
/* SUBMANSEC = DT */

/*S
  PetscDS - PETSc object that manages a discrete system, which is a set of discretizations + continuum equations from a `PetscWeakForm`

  Level: intermediate

.seealso: `PetscDSCreate()`, `PetscDSSetType()`, `PetscDSType`, `PetscWeakForm`, `PetscFECreate()`, `PetscFVCreate()`
S*/
typedef struct _p_PetscDS *PetscDS;

/*S
  PetscWeakForm - PETSc object that manages a sets of pointwise functions defining a system of equations

  Level: intermediate

.seealso: `PetscWeakFormCreate()`, `PetscDS`, `PetscFECreate()`, `PetscFVCreate()`
S*/
typedef struct _p_PetscWeakForm *PetscWeakForm;

/*S
  PetscFormKey - This key indicates how to use a set of pointwise functions defining part of a system of equations

  The subdomain on which to integrate is specified by (label, value), the test function field by (field), and the
  piece of the equation by (part). For example, LHS = 0 and RHS = 1 in IMEX methods. More pieces can be present for
  operator splitting methods.

  Level: intermediate

  Note:
  This is a struct, not a `PetscObject`

.seealso: `DMPlexSNESComputeResidualFEM()`, `DMPlexSNESComputeJacobianFEM()`, `DMPlexSNESComputeBoundaryFEM()`
S*/
typedef struct {
  DMLabel  label; /* The (label, value) select a subdomain */
  PetscInt value;
  PetscInt field; /* Selects the field for the test function */
  PetscInt part;  /* Selects the equation part. For example, LHS = 0 and RHS = 1 in IMEX methods. More pieces can be present for operator splitting methods. */
} PetscFormKey;

/*E
  PetscWeakFormKind - The kind of weak form. The specific forms are given in the documentation for the integraton functions.

  Values:
+ OBJECTIVE                  - Objective form
. F0, F1                     - Residual forms
. G0, G1, G2, G3             - Jacobian forms
. GP0, GP1, GP2, GP3         - Jacobian forms used to construct the preconditioner
. GT0, GT1, GT2, GT3         - Dynamic Jacobian matrix forms
. BDF0, BDF1                 - Boundary Residual forms
. BDG0, BDG1, BDG2, BDG3     - Jacobian forms
. BDGP0, BDGP1, BDGP2, BDGP3 - Jacobian forms used to construct the preconditioner
. R                          - Riemann solver
- CEED                       - libCEED QFunction

  Level: beginner

.seealso: `PetscWeakForm`, `PetscFEIntegrateResidual()`, `PetscFEIntegrateJacobian()`, `PetscFEIntegrateBdResidual()`, `PetscFEIntegrateBdJacobian()`,
          `PetscFVIntegrateRHSFunction()`, `PetscWeakFormSetIndexResidual()`, `PetscWeakFormClearIndex()`
E*/
typedef enum {
  PETSC_WF_OBJECTIVE,
  PETSC_WF_F0,
  PETSC_WF_F1,
  PETSC_WF_G0,
  PETSC_WF_G1,
  PETSC_WF_G2,
  PETSC_WF_G3,
  PETSC_WF_GP0,
  PETSC_WF_GP1,
  PETSC_WF_GP2,
  PETSC_WF_GP3,
  PETSC_WF_GT0,
  PETSC_WF_GT1,
  PETSC_WF_GT2,
  PETSC_WF_GT3,
  PETSC_WF_BDF0,
  PETSC_WF_BDF1,
  PETSC_WF_BDG0,
  PETSC_WF_BDG1,
  PETSC_WF_BDG2,
  PETSC_WF_BDG3,
  PETSC_WF_BDGP0,
  PETSC_WF_BDGP1,
  PETSC_WF_BDGP2,
  PETSC_WF_BDGP3,
  PETSC_WF_R,
  PETSC_WF_CEED,
  PETSC_NUM_WF
} PetscWeakFormKind;
PETSC_EXTERN const char *const PetscWeakFormKinds[];

/*S
  PetscPointFn - A prototype of a pointwise function that can be passed to, for example, `PetscDSSetObjective()`

  Calling Sequence:
+ dim          - the coordinate dimension
. Nf           - the number of fields
. NfAux        - the number of auxiliary fields
. uOff         - the offset into `u`[] and `u_t`[] for each field
. uOff_x       - the offset into `u_x`[] for each field
. u            - each field evaluated at the current point
. u_t          - the time derivative of each field evaluated at the current point
. u_x          - the gradient of each field evaluated at the current point
. aOff         - the offset into `a`[] and `a_t`[] for each auxiliary field
. aOff_x       - the offset into `a_x`[] for each auxiliary field
. a            - each auxiliary field evaluated at the current point
. a_t          - the time derivative of each auxiliary field evaluated at the current point
. a_x          - the gradient of auxiliary each field evaluated at the current point
. t            - current time
. x            - coordinates of the current point
. numConstants - number of constant parameters
. constants    - constant parameters
- obj          - output values at the current point

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetObjective()`, `PetscDSGetObjective()`, PetscDSGetResidual()`, `PetscDSSetResidual()`,
          `PetscDSGetRHSResidual()`, `PetscDSGetRHSResidual()`, `PetscDSSetUpdate()`, `PetscDSGetUpdate()`, `DMPlexSetCoordinateMap()`
S*/
typedef void PetscPointFn(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar result[]);

/*S
  PetscPointJacFn - A prototype of a pointwise function that can be passed to, for example, `PetscDSSetJacobian()` for computing Jacobians

  Calling Sequence:
+ dim          - the coordinate dimension
. Nf           - the number of fields
. NfAux        - the number of auxiliary fields
. uOff         - the offset into `u`[] and `u_t`[] for each field
. uOff_x       - the offset into `u_x`[] for each field
. u            - each field evaluated at the current point
. u_t          - the time derivative of each field evaluated at the current point
. u_x          - the gradient of each field evaluated at the current point
. aOff         - the offset into `a`[] and `a_t`[] for each auxiliary field
. aOff_x       - the offset into a_`x`[] for each auxiliary field
. a            - each auxiliary field evaluated at the current point
. a_t          - the time derivative of each auxiliary field evaluated at the current point
. a_x          - the gradient of auxiliary each field evaluated at the current point
. t            - current time
. u_tShift     - the multiplier `a` for $dF/dU_t$
. x            - coordinates of the current point
. numConstants - number of constant parameters
. constants    - constant parameters
- g            - output values at the current point

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetJacobian()`, `PetscDSGetJacobian()`, PetscDSSetJacobianPreconditioner()`, `PetscDSGetJacobianPreconditioner()`,
          `PetscDSSetDynamicJacobian()`, `PetscDSGetDynamicJacobian()`
S*/
typedef void PetscPointJacFn(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g[]);

/*S
  PetscBdPointFn - A prototype of a pointwise boundary function that can be passed to, for example, `PetscDSSetBdResidual()`

  Calling Sequence:
+ dim          - the coordinate dimension
. Nf           - the number of fields
. NfAux        - the number of auxiliary fields
. uOff         - the offset into `u`[] and `u_t`[] for each field
. uOff_x       - the offset into `u_x`[] for each field
. u            - each field evaluated at the current point
. u_t          - the time derivative of each field evaluated at the current point
. u_x          - the gradient of each field evaluated at the current point
. aOff         - the offset into `a`[] and `a_t`[] for each auxiliary field
. aOff_x       - the offset into `a_x`[] for each auxiliary field
. a            - each auxiliary field evaluated at the current point
. a_t          - the time derivative of each auxiliary field evaluated at the current point
. a_x          - the gradient of auxiliary each field evaluated at the current point
. t            - current time
. x            - coordinates of the current point
. n            - unit normal at the current point
. numConstants - number of constant parameters
. constants    - constant parameters
- f            - output values at the current point

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetBdResidual()`, `PetscDSGetBdResidual()`, `PetscDSSetObjective()`, `PetscDSGetObjective()`, PetscDSGetResidual()`,
          `PetscDSGetRHSResidual()`, `PetscDSGetRHSResidual()`, `PetscDSSetUpdate()`, `PetscDSGetUpdate()`, `DMPlexSetCoordinateMap()`,
          `PetscDSSetResidual()`, `PetscPointJacFn`
S*/
typedef void PetscBdPointFn(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

/*S
  PetscBdPointJacFn - A prototype of a pointwise boundary function that can be passed to, for example, `PetscDSSetBdJacobian()`

  Calling Sequence:
+ dim          - the coordinate dimension
. Nf           - the number of fields
. NfAux        - the number of auxiliary fields
. uOff         - the offset into `u`[] and `u_t`[] for each field
. uOff_x       - the offset into `u_x`[] for each field
. u            - each field evaluated at the current point
. u_t          - the time derivative of each field evaluated at the current point
. u_x          - the gradient of each field evaluated at the current point
. aOff         - the offset into `a`[] and `a_t`[] for each auxiliary field
. aOff_x       - the offset into `a_x`[] for each auxiliary field
. a            - each auxiliary field evaluated at the current point
. a_t          - the time derivative of each auxiliary field evaluated at the current point
. a_x          - the gradient of auxiliary each field evaluated at the current point
. t            - current time
. u_tShift     - the multiplier `a` for $dF/dU_t$
. x            - coordinates of the current point
. n            - normal at the current point
. numConstants - number of constant parameters
. constants    - constant parameters
- g            - output values at the current point

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetBdJacobian()`, PetscDSGetBdJacobian()`, `PetscDSSetBdJacobianPreconditioner()`, `PetscDSGetBdJacobianPreconditioner()`,
          `PetscDSSetBdResidual()`, `PetscDSGetBdResidual()`, `PetscDSSetObjective()`, `PetscDSGetObjective()`, PetscDSGetResidual()`,
          `PetscDSGetRHSResidual()`, `PetscDSGetRHSResidual()`, `PetscDSSetUpdate()`, `PetscDSGetUpdate()`, `DMPlexSetCoordinateMap()`,
          `PetscDSSetResidual()`, `PetscPointJacFn`
S*/
typedef void PetscBdPointJacFn(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

/*S
  PetscPointExactSolutionFn - A prototype of a pointwise function that computes the exact solution to a PDE. Used with, for example,
  `PetscDSSetExactSolution()`

  Calling Sequence:
+ dim - the coordinate dimension
. t   - current time
. x   - coordinates of the current point
. Nc  - the number of field components
. u   - the solution field evaluated at the current point
- ctx - a user context, set with `PetscDSSetExactSolution()` or `PetscDSSetExactSolutionTimeDerivative()`

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetExactSolution()`, `PetscDSGetExactSolution()`, `PetscDSSetExactSolutionTimeDerivative()`, `PetscDSGetExactSolutionTimeDerivative()`
S*/
typedef PetscErrorCode PetscPointExactSolutionFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);

/*S
  PetscRiemannFn - A prototype of a pointwise function that can be passed to, for example, `PetscDSSetRiemannSolver()`

  Calling Sequence:
+ dim          - the coordinate dimension
. Nf           - The number of fields
. x            - The coordinates at a point on the interface
. n            - The normal vector to the interface
. uL           - The state vector to the left of the interface
. uR           - The state vector to the right of the interface
. numConstants - number of constant parameters
. constants    - constant parameters
. flux         - output array of flux through the interface
- ctx          - optional user context

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetRiemannSolver()`, `PetscDSGetRiemannSolver()`
S*/
typedef void PetscRiemannFn(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx);

/*S
  PetscSimplePointFn - A prototype of a simple pointwise function that can be passed to, for example, `DMPlexTransformExtrudeSetNormalFunction()`

  Calling Sequence:
+ dim  - The coordinate dimension of the original mesh (usually a surface)
. time - The current time, or 0.
. x    - The location of the current normal, in the coordinate space of the original mesh
. r    - The layer number of this point
. u    - The user provides the computed normal on output
- ctx  - An optional user context, this context may be obtained by the calling code with `DMGetApplicationContext()`

  Level: beginner

  Developer Note:
  The handling of `ctx` in the use of such functions may not be ideal since the context is not provided when the function pointer is provided with, for example, `DMSwarmSetCoordinateFunction()`

.seealso: `PetscPointFn`, `DMPlexTransformExtrudeSetNormalFunction()`, `DMSwarmSetCoordinateFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscSimplePointFn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx);

PETSC_EXTERN_TYPEDEF typedef PetscSimplePointFn *PetscSimplePointFunc PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscSimplePointFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscPointFn       *PetscPointFunc PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscPointFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscPointJacFn    *PetscPointJac PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscPointJacFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscBdPointFn     *PetscBdPointFunc PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscBdPointFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscBdPointJacFn  *PetscBdPointJac PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscBdPointJacFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscRiemannFn     *PetscRiemannFunc PETSC_DEPRECATED_TYPEDEF(3, 24, 0, "PetscRiemannFn*", );

/*S
  PetscPointBoundFn - A prototype of a pointwise function that can be passed to, for example, `PetscDSSetLowerBound()`

  Calling Sequence:
+ dim - the coordinate dimension
. t   - current time
. x   - coordinates of the current point
. Nc  - the number of field components
. u   - the lower bound evaluated at the current point
- ctx - a user context, passed in with, for example, `PetscDSSetLowerBound()`

  Level: beginner

.seealso: `PetscPointFn`, `PetscDSSetLowerBound()`, `PetscDSSetUpperBound()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscPointBoundFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
