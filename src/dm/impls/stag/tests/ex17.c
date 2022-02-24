static char help[] = "Test DMStag IS computation\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm;
  PetscInt        dim,dof0,dof1,dof2,dof3;
  PetscBool       flg;

  /* Create a DMStag object */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option\n");CHKERRQ(ierr);
    return 1;
  }
  if (dim == 1) {
    ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,6,PETSC_DECIDE,PETSC_DECIDE,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n");CHKERRQ(ierr);
    return 1;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMView(dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3);CHKERRQ(ierr);

  {
    IS is;
    DMStagStencil s;
    s.c = 0; s.loc = DMSTAG_ELEMENT;
    ierr = DMStagCreateISFromStencils(dm,1,&s,&is);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test 1\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  {
    IS is;
    DMStagStencil s;
    s.c = 0; s.loc = DMSTAG_RIGHT;
    ierr = DMStagCreateISFromStencils(dm,1,&s,&is);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test 2\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  if (dim > 1) {
    IS is;
    DMStagStencil s[2];
    s[0].c = 0; s[0].loc = DMSTAG_DOWN;
    s[1].c = 0; s[1].loc = DMSTAG_LEFT;
    ierr = DMStagCreateISFromStencils(dm,2,s,&is);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test 3\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  if (dim == 2 && dof1 > 1) {
    IS is;
    DMStagStencil s[5];
    s[0].c = 0; s[0].loc = DMSTAG_DOWN;
    s[1].c = 0; s[1].loc = DMSTAG_DOWN; /* redundant, should be ignored */
    s[2].c = 0; s[2].loc = DMSTAG_LEFT;
    s[3].c = 0; s[3].loc = DMSTAG_RIGHT; /* redundant, should be ignored */
    s[4].c = 1; s[4].loc = DMSTAG_RIGHT;
    ierr = DMStagCreateISFromStencils(dm,5,s,&is);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test 4\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  if (dim == 3 && dof0 > 1) {
    IS is;
    DMStagStencil s[3];
    s[0].c = 0; s[0].loc = DMSTAG_BACK_DOWN_LEFT;
    s[1].c = 0; s[1].loc = DMSTAG_FRONT_UP_RIGHT; /* redundant, should be ignored */
    s[2].c = 1; s[2].loc = DMSTAG_FRONT_DOWN_RIGHT;
    ierr = DMStagCreateISFromStencils(dm,3,s,&is);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test 5\n");CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -dim 2

   test:
      suffix: 2
      nsize: 1
      args: -dim 2

   test:
      suffix: 3
      nsize: 2
      args: -dim 2 -stag_dof_1 2

   test:
      suffix: 4
      nsize: 1
      args: -dim 3 -stag_dof_0 3

   test:
      suffix: 5
      nsize: 2
      args: -dim 3 -stag_dof_0 2

   test:
      suffix: 6
      nsize: 1
      args: -dim 1

   test:
      suffix: 7
      nsize: 2
      args: -dim 1

TEST*/
