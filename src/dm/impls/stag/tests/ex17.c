static char help[] = "Test DMStag IS computation\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM              dm;
  PetscInt        dim,dof0,dof1,dof2,dof3;
  PetscBool       flg;

  /* Create a DMStag object */
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option\n"));
    return 1;
  }
  if (dim == 1) {
    CHKERRQ(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
  } else if (dim == 2) {
    CHKERRQ(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,6,PETSC_DECIDE,PETSC_DECIDE,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
  } else if (dim == 3) {
    CHKERRQ(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n"));
    return 1;
  }
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3));

  {
    IS is;
    DMStagStencil s;
    s.c = 0; s.loc = DMSTAG_ELEMENT;
    CHKERRQ(DMStagCreateISFromStencils(dm,1,&s,&is));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test 1\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&is));
  }
  {
    IS is;
    DMStagStencil s;
    s.c = 0; s.loc = DMSTAG_RIGHT;
    CHKERRQ(DMStagCreateISFromStencils(dm,1,&s,&is));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test 2\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&is));
  }
  if (dim > 1) {
    IS is;
    DMStagStencil s[2];
    s[0].c = 0; s[0].loc = DMSTAG_DOWN;
    s[1].c = 0; s[1].loc = DMSTAG_LEFT;
    CHKERRQ(DMStagCreateISFromStencils(dm,2,s,&is));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test 3\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&is));
  }
  if (dim == 2 && dof1 > 1) {
    IS is;
    DMStagStencil s[5];
    s[0].c = 0; s[0].loc = DMSTAG_DOWN;
    s[1].c = 0; s[1].loc = DMSTAG_DOWN; /* redundant, should be ignored */
    s[2].c = 0; s[2].loc = DMSTAG_LEFT;
    s[3].c = 0; s[3].loc = DMSTAG_RIGHT; /* redundant, should be ignored */
    s[4].c = 1; s[4].loc = DMSTAG_RIGHT;
    CHKERRQ(DMStagCreateISFromStencils(dm,5,s,&is));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test 4\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&is));
  }
  if (dim == 3 && dof0 > 1) {
    IS is;
    DMStagStencil s[3];
    s[0].c = 0; s[0].loc = DMSTAG_BACK_DOWN_LEFT;
    s[1].c = 0; s[1].loc = DMSTAG_FRONT_UP_RIGHT; /* redundant, should be ignored */
    s[2].c = 1; s[2].loc = DMSTAG_FRONT_DOWN_RIGHT;
    CHKERRQ(DMStagCreateISFromStencils(dm,3,s,&is));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test 5\n"));
    CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&is));
  }

  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
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
