
static char help[] = "Tests VecView()/VecLoad() for DMDA vectors (this tests DMDAGlobalToNatural()).\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt      size;
  PetscInt         N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE,M=8,dof=1,stencil_width=1,P=5,pt = 0,st = 0;
  PetscErrorCode   ierr;
  PetscBool        flg2,flg3,native = PETSC_FALSE;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_STAR;
  DM               da;
  Vec              global1,global2,global3,global4;
  PetscScalar      mone = -1.0;
  PetscReal        norm;
  PetscViewer      viewer;
  PetscRandom      rdm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-native",&native,NULL));
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 3) {bx = DM_BOUNDARY_PERIODIC; by = DM_BOUNDARY_PERIODIC;}
  if (pt == 4) bz = DM_BOUNDARY_PERIODIC;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL));
  stencil_type = (DMDAStencilType) st;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-one",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-two",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-three",&flg3));
  if (flg2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,m,n,dof,stencil_width,0,0,&da));
  } else if (flg3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&da));
  } else {
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da));
  }
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&global1));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(DMCreateGlobalVector(da,&global2));
  CHKERRQ(DMCreateGlobalVector(da,&global3));
  CHKERRQ(DMCreateGlobalVector(da,&global4));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer));
  if (native) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE));
  CHKERRQ(VecSetRandom(global1,rdm));
  CHKERRQ(VecView(global1,viewer));
  CHKERRQ(VecSetRandom(global3,rdm));
  CHKERRQ(VecView(global3,viewer));
  if (native) CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
  if (native) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE));
  CHKERRQ(VecLoad(global2,viewer));
  CHKERRQ(VecLoad(global4,viewer));
  if (native) CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  if (native) {
    Vec filenative;
    PetscBool same;
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
    CHKERRQ(DMDACreateNaturalVector(da,&filenative));
    /* DMDA "natural" Vec does not commandeer VecLoad.  The following load will only work when run on the same process
     * layout, where as the standard VecView/VecLoad (using DMDA and not PETSC_VIEWER_NATIVE) can be read on a different
     * number of processors. */
    CHKERRQ(VecLoad(filenative,viewer));
    CHKERRQ(VecEqual(global2,filenative,&same));
    if (!same) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ex23: global vector does not match contents of file\n"));
      CHKERRQ(VecView(global2,0));
      CHKERRQ(VecView(filenative,0));
    }
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(VecDestroy(&filenative));
  }

  CHKERRQ(VecAXPY(global2,mone,global1));
  CHKERRQ(VecNorm(global2,NORM_MAX,&norm));
  if (norm != 0.0) {
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)bx));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3));
  }
  CHKERRQ(VecAXPY(global4,mone,global3));
  CHKERRQ(VecNorm(global4,NORM_MAX,&norm));
  if (norm != 0.0) {
    CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)bx));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3));
  }

  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(VecDestroy(&global1));
  CHKERRQ(VecDestroy(&global2));
  CHKERRQ(VecDestroy(&global3));
  CHKERRQ(VecDestroy(&global4));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1  3}}
      args: -one -dof {{1 2 3}} -stencil_type {{0 1}}

   test:
      suffix: 3
      nsize: {{2 4}}
      args: -two -dof {{1 3}} -stencil_type {{0 1}}

   test:
      suffix: 4
      nsize: {{1 4}}
      args: -three -dof {{2 3}} -stencil_type {{0 1}}

   test:
      suffix: 2
      nsize: 2
      args: -two -native

TEST*/
