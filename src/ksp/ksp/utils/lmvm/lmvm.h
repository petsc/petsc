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

typedef struct _MatOps_LMVM *MatOps_LMVM;
struct _MatOps_LMVM {
  PetscErrorCode (*update)(Mat,Vec,Vec);
  PetscErrorCode (*allocate)(Mat,Vec,Vec);
  PetscErrorCode (*reset)(Mat,PetscBool);
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*copy)(Mat,Mat,MatStructure);
};

typedef struct {
  /* Core data structures for stored updates */
  PETSCHEADER(struct _MatOps_LMVM);
  PetscBool allocated, prev_set;
  PetscInt m_old, m, k, nupdates, nrejects, nresets;
  Vec *S, *Y;
  Vec Xprev, Fprev;
  
  /* User-defined initial Jacobian tools */
  PetscBool user_pc, user_ksp, user_scale;
  PetscReal ksp_rtol, ksp_atol;
  PetscInt ksp_max_it;
  PetscReal J0scalar;
  Vec J0diag;
  Mat J0;
  PC J0pc;
  KSP J0ksp;
  
  /* Data structures to support common Mat functions */
  PetscReal shift;
  
  /* Miscellenous parameters */
  PetscBool square; /* flag for defining the LMVM approximation as a square matrix */
  PetscReal eps; /* (default: PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0)) */
  void *ctx; /* implementation specific context */
} Mat_LMVM;

/* Shared internal functions for LMVM matrices */
PETSC_INTERN PetscErrorCode MatUpdateKernel_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatUpdate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatAllocate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatReset_LMVM(Mat, PetscBool);

/* LMVM implementations of core Mat functionality */
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject, Mat);
PETSC_INTERN PetscErrorCode MatSetUp_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatView_LMVM(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatDestroy_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatCreate_LMVM(Mat);

/* Create functions for derived LMVM types
   NOTE: MatCreateXYZ() declarations for subtypes live under petsctao.h */
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDFP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBFGS(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSR1(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDiagBrdn(Mat);

/* Solve functions for derived LMVM types (necessary only for DFP and BFGS for re-use under SymBrdn) */
PETSC_INTERN PetscErrorCode MatSolve_LMVMDFP(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LMVMBFGS(Mat, Vec, Vec);

/* Mult functions for derived LMVM types (necessary only for DFP and BFGS for re-use under SymBrdn) */
PETSC_INTERN PetscErrorCode MatMult_LMVMDFP(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatMult_LMVMBFGS(Mat, Vec, Vec);

#endif
