
static char help[] = "Tests DMDA interpolation.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         M1 = 3,M2,dof = 1,s = 1,ratio = 2,dim = 1;
  PetscErrorCode   ierr;
  DM               da_c,da_f;
  Vec              v_c,v_f;
  Mat              Interp;
  PetscScalar      one = 1.0;
  PetscBool        pt;
  DMBoundaryType   bx = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_width",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ratio",&ratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-periodic",(PetscBool*)&pt,NULL);CHKERRQ(ierr);

  if (pt) {
    if (dim > 0) bx = DM_BOUNDARY_PERIODIC;
    if (dim > 1) by = DM_BOUNDARY_PERIODIC;
    if (dim > 2) bz = DM_BOUNDARY_PERIODIC;
  }
  if (bx == DM_BOUNDARY_NONE) {
    M2 = ratio*(M1-1) + 1;
  } else {
    M2 = ratio*M1;
  }

  /* Set up the array */
  if (dim == 1) {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M1,dof,s,NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M2,dof,s,NULL,&da_f);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,DMDA_STENCIL_BOX,M1,M1,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,DMDA_STENCIL_BOX,M2,M2,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_f);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,DMDA_STENCIL_BOX,M1,M1,M1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,NULL,&da_c);CHKERRQ(ierr);
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,DMDA_STENCIL_BOX,M2,M2,M2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,NULL,&da_f);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"dim must be 1,2, or 3");
  ierr = DMSetFromOptions(da_c);CHKERRQ(ierr);
  ierr = DMSetUp(da_c);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da_f);CHKERRQ(ierr);
  ierr = DMSetUp(da_f);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da_c,&v_c);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da_f,&v_f);CHKERRQ(ierr);

  ierr = VecSet(v_c,one);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da_c,da_f,&Interp,NULL);CHKERRQ(ierr);
  ierr = MatMult(Interp,v_c,v_f);CHKERRQ(ierr);
  ierr = VecView(v_f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatMultTranspose(Interp,v_f,v_c);CHKERRQ(ierr);
  ierr = VecView(v_c,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&Interp);CHKERRQ(ierr);
  ierr = VecDestroy(&v_c);CHKERRQ(ierr);
  ierr = DMDestroy(&da_c);CHKERRQ(ierr);
  ierr = VecDestroy(&v_f);CHKERRQ(ierr);
  ierr = DMDestroy(&da_f);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

