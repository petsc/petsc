/*
  Test DMCreateMatrix() for structure_only
*/

#include <petscdmda.h>

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscInt       nx=6,ny=6,nz=6,dim=1,dof=2;
  DM             da;
  Mat            A;
  PetscBool      struct_only=PETSC_TRUE;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  switch (dim) {
  case 1:
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,nx,dof,1,NULL,&da));
    break;
  case 2:
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,ny,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&da));
    break;
  default:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nx,ny,nz,
                      PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,2,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  }

  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMView(da,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-struct_only",&struct_only,NULL));
  CHKERRQ(DMSetMatrixStructureOnly(da,struct_only));
  CHKERRQ(DMCreateMatrix(da,&A));
  /* Set da->structure_only to default PETSC_FALSE in case da is being used to create new matrices */
  CHKERRQ(DMSetMatrixStructureOnly(da,PETSC_FALSE));

  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -dm_mat_type baij -dim 2

TEST*/
