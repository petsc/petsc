#ifndef PETSC_GSIMPL_H
#define PETSC_GSIMPL_H

#include <petsc/private/snesimpl.h> /*I "petscsnes.h"  I*/
#include <petscdm.h>

typedef struct {
  PetscInt   sweeps;     /* number of sweeps through the local subdomain before neighbor communication */
  PetscInt   max_its;    /* maximum iterations of the inner pointblock solver */
  PetscReal  rtol;       /* relative tolerance of the inner pointblock solver */
  PetscReal  abstol;     /* absolute tolerance of the inner pointblock solver */
  PetscReal  stol;       /* step tolerance of the inner pointblock solver */
  PetscReal  h;          /* differencing for secant variants */
  PetscBool  secant_mat; /* use the Jacobian to get the coloring for the secant */
  ISColoring coloring;
} SNES_NGS;

PETSC_EXTERN PetscErrorCode SNESComputeNGSDefaultSecant(SNES, Vec, Vec, void *);

#endif // PETSC_GSIMPL_H
