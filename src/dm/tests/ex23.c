
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-native",&native,NULL);CHKERRQ(ierr);
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 3) {bx = DM_BOUNDARY_PERIODIC; by = DM_BOUNDARY_PERIODIC;}
  if (pt == 4) bz = DM_BOUNDARY_PERIODIC;

  ierr         = PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL);CHKERRQ(ierr);
  stencil_type = (DMDAStencilType) st;

  ierr = PetscOptionsHasName(NULL,NULL,"-one",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-two",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-three",&flg3);CHKERRQ(ierr);
  if (flg2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,m,n,dof,stencil_width,0,0,&da);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,m,n,p,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global4);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  if (native) {ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);}
  ierr = VecSetRandom(global1,rdm);CHKERRQ(ierr);
  ierr = VecView(global1,viewer);CHKERRQ(ierr);
  ierr = VecSetRandom(global3,rdm);CHKERRQ(ierr);
  ierr = VecView(global3,viewer);CHKERRQ(ierr);
  if (native) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  if (native) {ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);}
  ierr = VecLoad(global2,viewer);CHKERRQ(ierr);
  ierr = VecLoad(global4,viewer);CHKERRQ(ierr);
  if (native) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (native) {
    Vec filenative;
    PetscBool same;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = DMDACreateNaturalVector(da,&filenative);CHKERRQ(ierr);
    /* DMDA "natural" Vec does not commandeer VecLoad.  The following load will only work when run on the same process
     * layout, where as the standard VecView/VecLoad (using DMDA and not PETSC_VIEWER_NATIVE) can be read on a different
     * number of processors. */
    ierr = VecLoad(filenative,viewer);CHKERRQ(ierr);
    ierr = VecEqual(global2,filenative,&same);CHKERRQ(ierr);
    if (!same) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: global vector does not match contents of file\n");CHKERRQ(ierr);
      ierr = VecView(global2,0);CHKERRQ(ierr);
      ierr = VecView(filenative,0);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&filenative);CHKERRQ(ierr);
  }

  ierr = VecAXPY(global2,mone,global1);CHKERRQ(ierr);
  ierr = VecNorm(global2,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)bx);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }
  ierr = VecAXPY(global4,mone,global3);CHKERRQ(ierr);
  ierr = VecNorm(global4,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm);CHKERRQ(ierr);
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
