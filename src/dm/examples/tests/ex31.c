static char help[] = "Tests MAIJ matrix for large DOF\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  Mat            M;
  Vec            x,y;
  PetscErrorCode ierr;
  DM             da,daf;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,5,PETSC_DECIDE,PETSC_DECIDE,41,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMRefine(da,PETSC_COMM_WORLD,&daf);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da,daf,&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf,&y);CHKERRQ(ierr);

  ierr = MatMult(M,x,y);CHKERRQ(ierr);
  ierr = MatMultTranspose(M,y,x);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
