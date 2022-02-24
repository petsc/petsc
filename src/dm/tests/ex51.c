
static char help[] = "Tests DMDAGlobalToNaturalAllCreate() using contour plotting for 2d DMDAs.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt         i,j,M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE;
  PetscMPIInt      rank;
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  PetscViewer      viewer;
  Vec              localall,global;
  PetscScalar      value,*vlocal;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  VecScatter       tolocalall,fromlocalall;
  PetscInt         start,end;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,m,n,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,M*N,&localall));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(VecGetOwnershipRange(global,&start,&end));
  for (i=start; i<end; i++) {
    value = 5.0*rank;
    CHKERRQ(VecSetValues(global,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecView(global,viewer));

  /*
     Create Scatter from global DMDA parallel vector to local vector that
   contains all entries
  */
  CHKERRQ(DMDAGlobalToNaturalAllCreate(da,&tolocalall));
  CHKERRQ(DMDANaturalAllToGlobalCreate(da,&fromlocalall));

  CHKERRQ(VecScatterBegin(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecGetArray(localall,&vlocal));
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      *vlocal++ += i + j*M;
    }
  }
  CHKERRQ(VecRestoreArray(localall,&vlocal));

  /* scatter back to global vector */
  CHKERRQ(VecScatterBegin(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecView(global,viewer));

  /* Free memory */
  CHKERRQ(VecScatterDestroy(&tolocalall));
  CHKERRQ(VecScatterDestroy(&fromlocalall));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&localall));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !complex

   test:
      nsize: 3

TEST*/
