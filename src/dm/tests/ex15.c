
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
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&s,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ratio",&ratio,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-periodic",(PetscBool*)&pt,NULL));

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
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M1,dof,s,NULL,&da_c));
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M2,dof,s,NULL,&da_f));
  } else if (dim == 2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,DMDA_STENCIL_BOX,M1,M1,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_c));
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,DMDA_STENCIL_BOX,M2,M2,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da_f));
  } else if (dim == 3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,DMDA_STENCIL_BOX,M1,M1,M1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,NULL,&da_c));
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,DMDA_STENCIL_BOX,M2,M2,M2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,NULL,&da_f));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"dim must be 1,2, or 3");
  CHKERRQ(DMSetFromOptions(da_c));
  CHKERRQ(DMSetUp(da_c));
  CHKERRQ(DMSetFromOptions(da_f));
  CHKERRQ(DMSetUp(da_f));

  CHKERRQ(DMCreateGlobalVector(da_c,&v_c));
  CHKERRQ(DMCreateGlobalVector(da_f,&v_f));

  CHKERRQ(VecSet(v_c,one));
  CHKERRQ(DMCreateInterpolation(da_c,da_f,&Interp,NULL));
  CHKERRQ(MatMult(Interp,v_c,v_f));
  CHKERRQ(VecView(v_f,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatMultTranspose(Interp,v_f,v_c));
  CHKERRQ(VecView(v_c,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&Interp));
  CHKERRQ(VecDestroy(&v_c));
  CHKERRQ(DMDestroy(&da_c));
  CHKERRQ(VecDestroy(&v_f));
  CHKERRQ(DMDestroy(&da_f));
  ierr = PetscFinalize();
  return ierr;
}
