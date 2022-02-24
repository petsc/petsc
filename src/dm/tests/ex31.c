static char help[] = "Tests MAIJ matrix for large DOF\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char *argv[])
{
  Mat            M;
  Vec            x,y;
  PetscErrorCode ierr;
  DM             da,daf;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,5,PETSC_DECIDE,PETSC_DECIDE,41,1,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMRefine(da,PETSC_COMM_WORLD,&daf));
  CHKERRQ(DMCreateInterpolation(da,daf,&M,NULL));
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(DMCreateGlobalVector(daf,&y));

  CHKERRQ(MatMult(M,x,y));
  CHKERRQ(MatMultTranspose(M,y,x));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(DMDestroy(&daf));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&M));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
