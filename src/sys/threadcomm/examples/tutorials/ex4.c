static char help[] = "Tests basic vector routines.\n\n";

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

  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"x = %lf\n",PetscRealPart(one));CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"y = %lf\n",PetscRealPart(two));CHKERRQ(ierr);

  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  v = two+alpha*one;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"y+%lfx = %lf\n",alpha,PetscRealPart(v));CHKERRQ(ierr);
  
  ierr = VecDot(x,y,&dot);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Dot product %d*(%lf*%lf) is %lf\n",N,PetscRealPart(one),PetscRealPart(v),PetscRealPart(dot));CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
