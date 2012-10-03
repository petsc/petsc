
static char help[] = "Tests VecView()/VecLoad() for DMDA vectors (this tests DMDAGlobalToNatural()).\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt      size;
  PetscInt         N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE,M=8,dof=1,stencil_width=1,P=5,pt = 0,st = 0;
  PetscErrorCode   ierr;
  PetscBool        flg2,flg3;
  DMDABoundaryType bx = DMDA_BOUNDARY_NONE,by = DMDA_BOUNDARY_NONE,bz = DMDA_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_STAR;
  DM               da;
  Vec              global1,global2,global3,global4;
  PetscScalar      mone = -1.0;
  PetscReal        norm;
  PetscViewer      viewer;
  PetscRandom      rdm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_width",&stencil_width,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic",&pt,PETSC_NULL);CHKERRQ(ierr);
  if (pt == 1) bx = DMDA_BOUNDARY_PERIODIC;
  if (pt == 2) by = DMDA_BOUNDARY_PERIODIC;
  if (pt == 3) {bx = DMDA_BOUNDARY_PERIODIC; by = DMDA_BOUNDARY_PERIODIC;}
  if (pt == 4) bz = DMDA_BOUNDARY_PERIODIC;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-stencil_type",&st,PETSC_NULL);CHKERRQ(ierr);
  stencil_type = (DMDAStencilType) st;

  ierr = PetscOptionsHasName(PETSC_NULL,"-one",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-two",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-three",&flg3);CHKERRQ(ierr);
  if (flg2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,m,n,dof,stencil_width,0,0,&da);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
  }
  else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,PETSC_NULL,&da);CHKERRQ(ierr);
  }

  ierr = DMCreateGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global4);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecSetRandom(global1,rdm);CHKERRQ(ierr);
  ierr = VecView(global1,viewer);CHKERRQ(ierr);
  ierr = VecSetRandom(global3,rdm);CHKERRQ(ierr);
  ierr = VecView(global3,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(global2,viewer);CHKERRQ(ierr);
  ierr = VecLoad(global4,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecAXPY(global2,mone,global1);CHKERRQ(ierr);
  ierr = VecNorm(global2,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %G should be zero\n",norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)bx);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }
  ierr = VecAXPY(global4,mone,global3);CHKERRQ(ierr);
  ierr = VecNorm(global4,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %G should be zero\n",norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)bx);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }


  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&global1);CHKERRQ(ierr);
  ierr = VecDestroy(&global2);CHKERRQ(ierr);
  ierr = VecDestroy(&global3);CHKERRQ(ierr);
  ierr = VecDestroy(&global4);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
