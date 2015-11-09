
static char help[] = "Tests DMCreateColoring() in 3d.\n\n";

#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt        i,M = 3,N = 5,P=3,s=1,w=2,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode  ierr;
  PetscInt        *lx = NULL,*ly = NULL,*lz = NULL;
  DM              da;
  PetscBool       flg = PETSC_FALSE,test_order = PETSC_FALSE;
  ISColoring      coloring;
  Mat             mat;
  DMDAStencilType stencil_type = DMDA_STENCIL_BOX;
  Vec             lvec,dvec;
  MatFDColoring   fdcoloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL);CHKERRQ(ierr);
  if (flg) stencil_type =  DMDA_STENCIL_STAR;
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_order",&test_order,NULL);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,1,"Must set -m option with -distribute option");
    ierr = PetscMalloc1(m,&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) lx[i] = 4;
    lx[m-1] = M - 4*(m-1);

    if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,1,"Must set -n option with -distribute option");
    ierr = PetscMalloc1(n,&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) ly[i] = 2;
    ly[n-1] = N - 2*(n-1);

    if (p == PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,1,"Must set -p option with -distribute option");
    ierr = PetscMalloc1(p,&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) lz[i] = 2;
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da);CHKERRQ(ierr);
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);
  ierr = PetscFree(lz);CHKERRQ(ierr);

  ierr = DMSetMatType(da,MATMPIAIJ);CHKERRQ(ierr);
  ierr = DMCreateColoring(da,IS_COLORING_GLOBAL,&coloring);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATMPIAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&mat);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(mat,coloring,&fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(mat,coloring,fdcoloring);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&dvec);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&lvec);CHKERRQ(ierr);

  /* Free memory */
  ierr = MatFDColoringDestroy(&fdcoloring);CHKERRQ(ierr);
  ierr = VecDestroy(&dvec);CHKERRQ(ierr);
  ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}





















