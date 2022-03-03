static char help[] = "Test DMStag explicit coordinate routines";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscBool      flg;
  DM             dm;
  Vec            coord;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option with value 1, 2, or 3");

  if (dim == 1) {
    CHKERRQ(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,2,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm));
  } else if (dim == 2) {
    CHKERRQ(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,PETSC_DECIDE,PETSC_DECIDE,2,3,4,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
  } else if (dim == 3) {
    CHKERRQ(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option with value 1, 2, or 3");

  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMStagSetUniformCoordinatesExplicit(dm,1.0,3.0,1.0,3.0,1.0,3.0));
  CHKERRQ(DMGetCoordinates(dm,&coord));
  CHKERRQ(VecView(coord,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1d_1
      nsize: 1
      args: -dim 1

   test:
      suffix: 1d_2
      nsize: 2
      args: -dim 1

   test:
      suffix: 2d_1
      nsize: 1
      args: -dim 2

   test:
      suffix: 2d_2
      nsize: 4
      args: -dim 2

   test:
      suffix: 3d_1
      nsize: 1
      args: -dim 3

   test:
      suffix: 3d_2
      nsize: 8
      args: -dim 3

TEST*/
