
static char help[] = "Tests DMDAGlobalToNaturalAllCreate() using contour plotting for 2d DMDAs.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt         i,j,M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE;
  PetscMPIInt      rank;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  PetscViewer      viewer;
  Vec              localall,global;
  PetscScalar      value,*vlocal;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  VecScatter       tolocalall,fromlocalall;
  PetscInt         start,end;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,m,n,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,M*N,&localall));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(VecGetOwnershipRange(global,&start,&end));
  for (i=start; i<end; i++) {
    value = 5.0*rank;
    PetscCall(VecSetValues(global,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecView(global,viewer));

  /*
     Create Scatter from global DMDA parallel vector to local vector that
   contains all entries
  */
  PetscCall(DMDAGlobalToNaturalAllCreate(da,&tolocalall));
  PetscCall(DMDANaturalAllToGlobalCreate(da,&fromlocalall));

  PetscCall(VecScatterBegin(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecGetArray(localall,&vlocal));
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      *vlocal++ += i + j*M;
    }
  }
  PetscCall(VecRestoreArray(localall,&vlocal));

  /* scatter back to global vector */
  PetscCall(VecScatterBegin(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecView(global,viewer));

  /* Free memory */
  PetscCall(VecScatterDestroy(&tolocalall));
  PetscCall(VecScatterDestroy(&fromlocalall));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&localall));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !complex

   test:
      nsize: 3

TEST*/
