#if !defined(__LMVM_H)
#define __LMVM_H
#include <petscksp.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>


/*
  MATLMVM format - a base matrix-type that represents Limited-Memory 
  Variable Metric (LMVM) approximations of a Jacobian.
  
  LMVM approximations can be symmetric, symmetric positive-definite, 
  rectangular, or otherwise square with no determinable properties. 
  Each derived LMVM type should automatically set its matrix properties 
  if its construction can guarantee symmetry (MAT_SYMMETRIC) or symmetric 
  positive-definiteness (MAT_SPD).
*/

typedef struct {
  /* Core data structures for stored updates */
  PetscBool allocated, allocatedP, prev_set;
  PetscInt m, k, nupdates, nrejects;
  PetscReal phi;
  Vec *S, *Y, *P;
  Vec Q, R;
  Vec Xprev, Fprev;
  
  /* User-defined initial Jacobian tools */
  PetscBool user_pc, user_ksp, user_scale, square;
  PetscReal ksp_rtol, ksp_atol;
  PetscReal scale;
  PetscInt ksp_max_it;
  Mat J0;
  PC J0pc;
  KSP J0ksp;
  Vec diag_scale;
  
  /* Miscellenous parameters */
  PetscReal eps; /* (default: PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0)) */
} Mat_LMVM;

/* Shared internal functions for LMVM matrices
   NOTE: MATLMVM is not a registered matrix type */
PETSC_INTERN PetscErrorCode MatGetVecs_LMVM(Mat, Vec*, Vec*);
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject, Mat);
PETSC_INTERN PetscErrorCode MatSetUp_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatView_LMVM(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatDestroy_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatCreate_LMVM(Mat);

/* Create functions for derived LMVM types
   NOTE: MatCreateXYZ() declarations for subtypes live under petsctao.h */
PETSC_EXTERN PetscErrorCode MatCreate_LDFP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LBFGS(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LSR1(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LBRDN(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMBRDN(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LSBRDN(Mat);

/* Internal solve routines for derived LMVM types */
PETSC_INTERN PetscErrorCode MatSolve_LDFP(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LBFGS(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LSR1(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LBRDN(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LMBRDN(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LSBRDN(Mat, Vec, Vec);

#endif