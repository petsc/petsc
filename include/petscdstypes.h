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
