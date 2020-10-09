
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
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL);CHKERRQ(ierr);
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,m,n,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M*N,&localall);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = 5.0*rank;
    ierr  = VecSetValues(global,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecView(global,viewer);CHKERRQ(ierr);

  /*
     Create Scatter from global DMDA parallel vector to local vector that
   contains all entries
  */
  ierr = DMDAGlobalToNaturalAllCreate(da,&tolocalall);CHKERRQ(ierr);
  ierr = DMDANaturalAllToGlobalCreate(da,&fromlocalall);CHKERRQ(ierr);

  ierr = VecScatterBegin(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(tolocalall,global,localall,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecGetArray(localall,&vlocal);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      *vlocal++ += i + j*M;
    }
  }
  ierr = VecRestoreArray(localall,&vlocal);CHKERRQ(ierr);

  /* scatter back to global vector */
  ierr = VecScatterBegin(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fromlocalall,localall,global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecView(global,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecScatterDestroy(&tolocalall);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&fromlocalall);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&localall);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   build:
     requires: !complex

   test:
      nsize: 3

TEST*/
