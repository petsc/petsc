
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL));
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 4) {bx = DM_BOUNDARY_PERIODIC; by = DM_BOUNDARY_PERIODIC;}

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL));
  stencil_type = (DMDAStencilType) st;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-oned",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-twod",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-threed",&flg3));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-binary",&isbinary));
#if defined(PETSC_HAVE_HDF5)
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-hdf5",&ishdf5));
#endif
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mpiio",&mpiio));
  if (flg2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da));
  } else if (flg3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da));
  } else {
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da));
  }
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&global1));
  CHKERRQ(PetscObjectSetName((PetscObject)global1,"Test_Vec"));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecSetRandom(global1,rdm));
  if (isbinary) {
    if (mpiio) {
      CHKERRQ(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));
    }
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option");
  CHKERRQ(VecView(global1,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Global vector written to temp file is \n"));
  CHKERRQ(VecView(global1,PETSC_VIEWER_STDOUT_WORLD));

  if (flg2) {
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da2));
  } else if (flg3) {
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da2));
  } else {
    CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da2));
  }
  CHKERRQ(DMSetFromOptions(da2));
  CHKERRQ(DMSetUp(da2));

  if (isbinary) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option");

  CHKERRQ(DMCreateGlobalVector(da2,&global2));
  CHKERRQ(PetscObjectSetName((PetscObject)global2,"Test_Vec"));
  CHKERRQ(VecLoad(global2,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Global vector read from temp file is \n"));
  CHKERRQ(VecView(global2,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecAXPY(global2,mone,global1));
  CHKERRQ(VecNorm(global2,NORM_MAX,&norm));
  if (norm != 0.0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %D %D %D %D\n",M,N,P,dof));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %D stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)pt));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3));
  }

  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(DMDestroy(&da2));
  CHKERRQ(VecDestroy(&global1));
  CHKERRQ(VecDestroy(&global2));
  ierr = PetscFinalize();
  return ierr;
}
