
static char help[] = "Tests DMDA interpolation.\n\n";

#include "petscdm.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M1 = 3,M2,dof = 1,s = 1,ratio = 2,dim = 1;
  PetscErrorCode ierr;
  DM             da_c,da_f;
  Vec            v_c,v_f;
  Mat            I;
  PetscScalar    one = 1.0;
  DMDAPeriodicType pt = DMDA_NONPERIODIC;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dim",&dim,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_width",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ratio",&ratio,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-periodic",(PetscBool*)&pt,PETSC_NULL);CHKERRQ(ierr);

  if (pt != DMDA_NONPERIODIC) {
    if (dim == 1) pt = DMDA_XPERIODIC;
    if (dim == 2) pt = DMDA_XYPERIODIC;
    if (dim == 3) pt = DMDA_XYZPERIODIC;
  }
  if (pt == DMDA_NONPERIODIC) {
    M2   = ratio*(M1-1) + 1;
  } else {
    M2 = ratio*M1;
  }

  /* Set up the array */ 
  if (dim == 1) {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,pt,M1,dof,s,PETSC_NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate1d(PETSC_COMM_WORLD,pt,M2,dof,s,PETSC_NULL,&da_f);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,pt,DMDA_STENCIL_BOX,M1,M1,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,pt,DMDA_STENCIL_BOX,M2,M2,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da_f);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,pt,DMDA_STENCIL_BOX,M1,M1,M1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate3d(PETSC_COMM_WORLD,pt,DMDA_STENCIL_BOX,M2,M2,M2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da_f);CHKERRQ(ierr);
  }

  ierr = DMCreateGlobalVector(da_c,&v_c);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da_f,&v_f);CHKERRQ(ierr);

  ierr = VecSet(v_c,one);CHKERRQ(ierr);
  ierr = DMGetInterpolation(da_c,da_f,&I,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMult(I,v_c,v_f);CHKERRQ(ierr);
  ierr = VecView(v_f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMultTranspose(I,v_f,v_c);CHKERRQ(ierr);
  ierr = VecView(v_c,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(I);CHKERRQ(ierr);
  ierr = VecDestroy(v_c);CHKERRQ(ierr);
  ierr = DMDestroy(da_c);CHKERRQ(ierr);
  ierr = VecDestroy(v_f);CHKERRQ(ierr);
  ierr = DMDestroy(da_f);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 



