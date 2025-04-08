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
. GP0, GP1, GP2, GP3         - Jacobian preconditioner matrix forms
. GT0, GT1, GT2, GT3         - Dynamic Jacobian matrix forms
. BDF0, BDF1                 - Boundary Residual forms
. BDG0, BDG1, BDG2, BDG3     - Jacobian forms
. BDGP0, BDGP1, BDGP2, BDGP3 - Jacobian preconditioner matrix forms
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

typedef void (*PetscPointFunc)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
typedef void (*PetscPointJac)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
typedef void (*PetscBdPointFunc)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
typedef void (*PetscBdPointJac)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
typedef void (*PetscRiemannFunc)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);

/*S
  PetscSimplePointFn - A prototype of a simple pointwise function that can be passed to, for example, `DMPlexTransformExtrudeSetNormalFunction()`

  Calling Sequence:
+ dim  - The coordinate dimension of the original mesh (usually a surface)
. time - The current time, or 0.
. x    - The location of the current normal, in the coordinate space of the original mesh
. r    - The layer number of this point
. u    - The user provides the computed normal on output
- ctx  - An optional user context

  Level: beginner

  Note:
  The deprecated `PetscSimplePointFunc` works as a replacement for `PetscSimplePointFn` *

.seealso: `DMPlexTransformExtrudeSetNormalFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(PetscSimplePointFn)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx);

PETSC_EXTERN_TYPEDEF typedef PetscSimplePointFn *PetscSimplePointFunc;
