
static char help[] = "Tests VecView()/VecLoad() for DMDA vectors (this tests DMDAGlobalToNatural()).\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  PetscMPIInt      rank,size;
  PetscInt         N            = 6,M=8,P=5,dof=1;
  PetscInt         stencil_width=1,pt=0,st=0;
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-periodic",&pt,NULL));
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 4) {bx = DM_BOUNDARY_PERIODIC; by = DM_BOUNDARY_PERIODIC;}

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-stencil_type",&st,NULL));
  stencil_type = (DMDAStencilType) st;

  PetscCall(PetscOptionsHasName(NULL,NULL,"-oned",&flg2));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-twod",&flg2));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-threed",&flg3));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-binary",&isbinary));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscOptionsHasName(NULL,NULL,"-hdf5",&ishdf5));
#endif
  PetscCall(PetscOptionsHasName(NULL,NULL,"-mpiio",&mpiio));
  if (flg2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da));
  } else if (flg3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da));
  } else {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da));
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da,&global1));
  PetscCall(PetscObjectSetName((PetscObject)global1,"Test_Vec"));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecSetRandom(global1,rdm));
  if (isbinary) {
    if (mpiio) {
      PetscCall(PetscOptionsSetValue(NULL,"-viewer_binary_mpiio",""));
    }
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_WRITE,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option");
  PetscCall(VecView(global1,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Global vector written to temp file is \n"));
  PetscCall(VecView(global1,PETSC_VIEWER_STDOUT_WORLD));

  if (flg2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stencil_type,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,&da2));
  } else if (flg3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da2));
  } else {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,bx,M,dof,stencil_width,NULL,&da2));
  }
  PetscCall(DMSetFromOptions(da2));
  PetscCall(DMSetUp(da2));

  if (isbinary) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"temp",FILE_MODE_READ,&viewer));
#endif
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Invalid Viewer : Run with -binary or -hdf5 option");

  PetscCall(DMCreateGlobalVector(da2,&global2));
  PetscCall(PetscObjectSetName((PetscObject)global2,"Test_Vec"));
  PetscCall(VecLoad(global2,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Global vector read from temp file is \n"));
  PetscCall(VecView(global2,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecAXPY(global2,mone,global1));
  PetscCall(VecNorm(global2,NORM_MAX,&norm));
  if (norm != 0.0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ex23: Norm of difference %g should be zero\n",(double)norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Number of processors %d\n",size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  M,N,P,dof %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",M,N,P,dof));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  stencil_width %" PetscInt_FMT " stencil_type %d periodic %d\n",stencil_width,(int)stencil_type,(int)pt));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  dimension %d\n",1 + (int) flg2 + (int) flg3));
  }

  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&da2));
  PetscCall(VecDestroy(&global1));
  PetscCall(VecDestroy(&global2));
  PetscCall(PetscFinalize());
  return 0;
}
