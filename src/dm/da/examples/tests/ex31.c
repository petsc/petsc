static char help[] = "Tests MAIJ matrix for large DOF\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  Mat            M;
  Vec            x,y;
  PetscErrorCode ierr;
  DA             da,daf;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,4,5,PETSC_DECIDE,PETSC_DECIDE,41,1,0,0,&da);CHKERRQ(ierr);
  ierr = DARefine(da,PETSC_COMM_WORLD,&daf);CHKERRQ(ierr);
  ierr = DAGetInterpolation(da,daf,&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(daf,&y);CHKERRQ(ierr);

  ierr = MatMult(M,x,y);CHKERRQ(ierr);
  ierr = MatMultTranspose(M,y,x);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = DADestroy(daf);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = MatDestroy(M);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
