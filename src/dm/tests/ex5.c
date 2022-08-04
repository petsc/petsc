/*
  Test DMCreateMatrix() for structure_only
*/

#include <petscdmda.h>

int main(int argc, char *argv[])
{
  PetscInt       nx=6,ny=6,nz=6,dim=1,dof=2;
  DM             da;
  Mat            A;
  PetscBool      struct_only=PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  switch (dim) {
  case 1:
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,nx,dof,1,NULL,&da));
    break;
  case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,ny,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&da));
    break;
  default:
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nx,ny,nz,
                           PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,2,NULL,NULL,NULL,&da));
  }

  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMView(da,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-struct_only",&struct_only,NULL));
  PetscCall(DMSetMatrixStructureOnly(da,struct_only));
  PetscCall(DMCreateMatrix(da,&A));
  /* Set da->structure_only to default PETSC_FALSE in case da is being used to create new matrices */
  PetscCall(DMSetMatrixStructureOnly(da,PETSC_FALSE));

  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -dm_mat_type baij -dim 2

TEST*/
