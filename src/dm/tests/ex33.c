
static char help[] = "Tests VecView()/VecLoad() for DMDA vectors (this tests DMDAGlobalToNatural()).\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank,size;
  PetscInt         N            = 6,M=8,P=5,dof=1;
  PetscInt         stencil_width=1,pt=0,st=0;
  PetscErrorCode   ierr;
  PetscBool        flg2,flg3,isbinary,mpiio;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_STAR;
  DM               da,da2;
  Vec              global1,global2;
  PetscScalar      mone = -1.0;
  PetscReal        norm;
  PetscViewer      viewer;
  PetscRandom      rdm;
#if defined(PETSC_HAVE_HDF5)
  PetscBool ishdf5;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL);CHKERRQ(ierr);
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 4) {bx = DM_BOUNDARY_PERIODIC; by = DM_BOUNDARY_PERIODIC;}

  ierr         = PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL);CHKERRQ(ierr);
  stencil_type = (DMDAStencilType) st;

  ierr = PetscOptionsHasName(NULL,NULL,"-oned",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-twod",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-threed",&flg3);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-binary",&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscOptionsHasName(NULL,NULL,"-hdf5",&ishdf5);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsHasName(NULL,NULL,"-mpiio",&mpiio);CHKERRQ(ierr);
  if (flg2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global1,"Test_Vec");CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(global1,rdm);CHKERRQ(ierr);
  if (isbinary) {
    if (mpiio) {
      ierr = PetscOptionsSetValue(NULL,"-viewer_binary_mpiio","");CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option\n");
  ierr = VecView(global1,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Global vector written to temp file is \n");CHKERRQ(ierr);
  ierr = VecView(global1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (flg2) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da2);CHKERRQ(ierr);
  } else if (flg3) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da2);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da2);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);

  if (isbinary) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option\n");

  ierr = DMCreateGlobalVector(da2,&global2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global2,"Test_Vec");CHKERRQ(ierr);
  ierr = VecLoad(global2,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Global vector read from temp file is \n");CHKERRQ(ierr);
  ierr = VecView(global2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecAXPY(global2,mone,global1);CHKERRQ(ierr);
  ierr = VecNorm(global2,NORM_MAX,&norm);CHKERRQ(ierr);
  if (norm != 0.0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)pt);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3);CHKERRQ(ierr);
  }

  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = VecDestroy(&global1);CHKERRQ(ierr);
  ierr = VecDestroy(&global2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
