#ifndef __APPROXMAT_H
#define __APPROXMAT_H

#include "private/matimpl.h"
#include "tao_sys.h"

typedef struct {
  PetscBool allocated;
  PetscInt lm;
  PetscInt lmnow;
  PetscInt iter;
  PetscInt nupdates;
  PetscInt nrejects;

  Vec *S;
  Vec *Y;
  Vec *Bs;
  Vec Gprev;
  Vec Xprev;

  Vec W; /* work vector */
  PetscReal *rho;

} MatApproxCtx;

extern PetscErrorCode MatCreateAPPROX(MPI_Comm, PetscInt, PetscInt, Mat*);

/* Petsc Mat overrides */
extern PetscErrorCode MatView_APPROX(Mat, PetscViewer);
extern PetscErrorCode MatDestroy_APPROX(Mat);
extern PetscErrorCode MatMult_APPROX(Mat, Vec, Vec);

/* Functions used by TAO */
PetscErrorCode MatApproxReset(Mat);
PetscErrorCode MatApproxUpdate(Mat,Vec, Vec);
PetscErrorCode MatApproxAllocateVectors(Mat m, Vec v);

#endif



