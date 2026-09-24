/*
   Private data structure used by the EKSM method
*/
#pragma once

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <../src/mat/impls/nest/matnestimpl.h>

typedef struct {
  /* User data */
  PetscScalar shift;
  PetscBool   shift_set;
  KSP         ksps; /* linear solver for K + shift M */
  KSP         kspm; /* linear solver for M */

  /* Hessenberg matrix and orthogonalization information. */
  PetscScalar *hh_origin;     /* holds the upper Hessenberg matrix */
  PetscScalar *kk_origin;     /* holds the second matrix of the Krylov relation */
  PetscScalar *tt_origin;     /* temporary matrix with the coefficient of the projected problem */
  PetscScalar *yy_origin;     /* solutions of the projected problem */
  PetscScalar *rs_origin;     /* holds the right-hand side of the Hessenberg system */
  PetscInt     ldh, ldk, ldt; /* leading dimensions */
  PetscInt     factor;        /* defaults to 1, will be 2 in case of complex-conjugate pairs */
  PetscScalar *work;          /* LAPACK workspace */
  PetscInt     lwork;         /* length of workspace */

  PetscReal haptol; /* tolerance for happy breakdown */
  PetscReal v0norm; /* norm of first basis vector (M^{-1}b) */

  Vec      *vecs;           /* the work vectors */
  PetscInt  delta_allocate; /* number of vectors to preallocate in each block if not preallocated */
  PetscInt  vv_allocated;   /* number of allocated Krylov vectors */
  PetscInt  vecs_allocated; /* total number of vecs available */
  Vec     **user_work;
  PetscInt *mwork_alloc; /* number of work vectors allocated as part of a work-vector chunk */
  PetscInt  nwork_alloc; /* number of work vector chunks allocated */
} KSP_EKSM;

#define HH(a, b) (eksm->hh_origin + (b) * eksm->ldh + (a))
#define KK(a, b) (eksm->kk_origin + (b) * eksm->ldk + (a))
#define TT(a, b) (eksm->tt_origin + (b) * eksm->ldt + (a))
#define YY(a, b) (eksm->yy_origin + (b) * eksm->ldk + (a))

#define SHIFT_IS_COMPLEX(cmplx, i) (!PetscDefined(USE_COMPLEX) && (cmplx)[(i)])

/* vector names */
#define VEC_OFFSET 1
#define VEC_TEMP   eksm->vecs[0]
#define VEC_VV(i)  eksm->vecs[VEC_OFFSET + (i)]
