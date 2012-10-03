
static char help[] = "Tests DMDAGlobalToNaturalAllCreate() using contour plotting for 2d DMDAs.\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
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
  DMDABoundaryType bx = DMDA_BOUNDARY_NONE,by = DMDA_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;
  VecScatter       tolocalall,fromlocalall;
  PetscInt         start,end;


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-star_stencil",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,
                    M,N,m,n,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
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
  return 0;
}

