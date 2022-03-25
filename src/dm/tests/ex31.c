static char help[] = "Tests MAIJ matrix for large DOF\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char *argv[])
{
  Mat            M;
  Vec            x,y;
  DM             da,daf;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,5,PETSC_DECIDE,PETSC_DECIDE,41,1,0,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMRefine(da,PETSC_COMM_WORLD,&daf));
  PetscCall(DMCreateInterpolation(da,daf,&M,NULL));
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(DMCreateGlobalVector(daf,&y));

  PetscCall(MatMult(M,x,y));
  PetscCall(MatMultTranspose(M,y,x));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&daf));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&M));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
