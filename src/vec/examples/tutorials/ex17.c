/*
      Demonstrates the use of the "extra", polymorphic versions of many functions
*/
#include "petsc.h"
#include "petscvec.h"



int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscReal      norm;
  PetscMap       map;
  PetscInt       s;
  PetscMapType   t;

#if defined(__cplusplus)
  PetscInitialize(&argc,&args);
#else
  PetscInitialize(&argc,&args,0,0);
#endif

#if defined(__cplusplus)
  PetscSequentialPhaseBegin();
  PetscSequentialPhaseEnd();
#endif

#if defined(__cplusplus)
  ierr = VecCreate(&x);CHKERRQ(ierr);
#else
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
#endif
  ierr = VecSetSizes(x,2,2);CHKERRQ(ierr);
  ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
#if defined(__cplusplus)
  norm = VecNorm(x);
#else
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(x);CHKERRQ(ierr);

#if defined(__cplusplus)
  map = PetscMapCreate();
  s   = PetscMapGetLocalSize(map);
  s   = PetscMapGetSize(map);
  t   = PetscMapGetType(map);
#else
  ierr = PetscMapCreate(PETSC_COMM_SELF,&map);CHKERRQ(ierr);
#endif

  ierr = PetscMapDestroy(map);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
