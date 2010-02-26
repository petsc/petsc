/*
      Demonstrates the use of the "extra", polymorphic versions of many functions
*/
#include "petscsys.h"
#include "petscvec.h"



int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscReal      norm;
#if defined(__cplusplus) && !defined(PETSC_USE_EXTERN_CXX)
  PetscScalar    dot;
#endif

#if defined(__cplusplus) && !defined(PETSC_USE_EXTERN_CXX)
  PetscInitialize(&argc,&args);
#else
  PetscInitialize(&argc,&args,0,0);
#endif

#if defined(__cplusplus) && !defined(PETSC_USE_EXTERN_CXX)
  PetscSequentialPhaseBegin();
  PetscSequentialPhaseEnd();
#endif

#if defined(__cplusplus) && !defined(PETSC_USE_EXTERN_CXX)
  ierr = VecCreate(&x);CHKERRQ(ierr);
#else
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
#endif
  ierr = VecSetSizes(x,2,2);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_EXTERN_CXX)
  norm = VecNorm(x);
  norm = VecNorm(x,NORM_2);CHKERRQ(ierr);
  ierr = VecNormBegin(x,NORM_2);CHKERRQ(ierr);
  norm = VecNormEnd(x,NORM_2);CHKERRQ(ierr);
  ierr = VecDotBegin(x,x);CHKERRQ(ierr);
  dot  = VecDotEnd(x,x);CHKERRQ(ierr);
#else
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(x);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
