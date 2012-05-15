static char help[] = "Test Basic vector routines.\n\n";

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>
#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscScalar    dot=0.0,v;
  Vec            x,y;
  PetscInt       N=8;
  PetscScalar    one=1.0,two=2.0,alpha=2.0;

  PetscInitialize(&argc,&argv,(char *)0,help);

#if defined(PETSC_THREADCOMM_ACTIVE)
  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"x = %lf\n",one);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"y = %lf\n",two);CHKERRQ(ierr);

  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  v = two+alpha*one;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"x+%lfy = %lf\n",alpha,v);CHKERRQ(ierr);
  
  ierr = VecDot(x,y,&dot);CHKERRQ(ierr);

#if defined(PETSC_THREADCOMM_ACTIVE)
  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#endif

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Dot product %d*(%lf*%lf) is %lf\n",N,one,v,dot);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
