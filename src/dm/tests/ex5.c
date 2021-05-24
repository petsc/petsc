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

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  switch (dim) {
  case 1:
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,nx,dof,1,NULL,&da);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,ny,PETSC_DECIDE,PETSC_DECIDE,dof,1,NULL,NULL,&da);CHKERRQ(ierr);
    break;
  default:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,nx,ny,nz,
                      PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,2,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  }

  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMView(da,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-struct_only",&struct_only,NULL);CHKERRQ(ierr);
  ierr = DMSetMatrixStructureOnly(da,struct_only);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  /* Set da->structure_only to default PETSC_FALSE in case da is being used to create new matrices */
  ierr = DMSetMatrixStructureOnly(da,PETSC_FALSE);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -dm_mat_type baij -dim 2

TEST*/
