#ifndef __TAO_BQPIP_H
#define __TAO_BQPIP_H

#include <petsc/private/taoimpl.h>

typedef struct{

  /* Parameters */
  PetscInt predcorr;
  PetscReal ksp_tol;

  /* Problem variables, vectors and index sets */
  PetscInt n;   /* Dimension of the Problem */
  PetscInt m;  /* Number of constraints */

  /* Problem statistics */
  PetscReal dinfeas;
  PetscReal pinfeas;
  PetscReal pobj;
  PetscReal dobj;
  PetscReal gap;
  PetscReal rgap;
  PetscReal mu;
  PetscReal sigma;
  PetscReal pathnorm;
  PetscReal pre_sigma;
  PetscReal psteplength;
  PetscReal dsteplength;
  PetscReal rnorm;

  /* Variable Vectors */
  Vec G;
  Vec DG;
  Vec T;
  Vec DT;
  Vec Z;
  Vec DZ;
  Vec S;
  Vec DS;
  Vec GZwork;
  Vec TSwork;
  Vec XL,XU;

  /* Work Vectors */
  Vec R3;
  Vec R5;
  Vec HDiag;
  Vec Work;

  Vec DiagAxpy;
  Vec RHS;
  Vec RHS2;


  /* Data */
  Vec B;
  Vec C0;
  PetscReal c;

}TAO_BQPIP;

static PetscErrorCode QPIPSetInitialPoint(TAO_BQPIP *, Tao);
static PetscErrorCode QPComputeStepDirection(TAO_BQPIP *, Tao);
static PetscErrorCode QPIPComputeResidual(TAO_BQPIP *, Tao);
static PetscErrorCode QPStepLength(TAO_BQPIP *);
static PetscErrorCode QPIPComputeNormFromCentralPath(TAO_BQPIP *,PetscReal *);

#endif












