      
static char help[] = "Tests DAGetColoring() in 3d.\n\n";

#include "petscmat.h"
#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       i,M = 3,N = 5,P=3,s=1,w=2,m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode ierr;
  PetscInt       *lx = PETSC_NULL,*ly = PETSC_NULL,*lz = PETSC_NULL;
  DA             da;
  PetscTruth     flg = PETSC_FALSE,test_order = PETSC_FALSE;
  ISColoring     coloring;
  Mat            mat;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  Vec            lvec,dvec;
  MatFDColoring  fdcoloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Read options */  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-test_order",&test_order,PETSC_NULL);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-distribute",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRQ(1,"Must set -m option with -distribute option");
    ierr = PetscMalloc(m*sizeof(PetscInt),&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRQ(1,"Must set -n option with -distribute option");
    ierr = PetscMalloc(n*sizeof(PetscInt),&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRQ(1,"Must set -p option with -distribute option");
    ierr = PetscMalloc(p*sizeof(PetscInt),&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) { lz[i] = 2;}
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,stencil_type,M,N,P,m,n,p,w,s,lx,ly,lz,&da);CHKERRQ(ierr);
  if (lx) {
    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);
    ierr = PetscFree(lz);CHKERRQ(ierr);
  }

  ierr = DAGetColoring(da,IS_COLORING_GLOBAL,MATMPIAIJ,&coloring);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATMPIAIJ,&mat);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(mat,coloring,&fdcoloring);CHKERRQ(ierr); 

  ierr = DACreateGlobalVector(da,&dvec);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&lvec);CHKERRQ(ierr);

  /* Free memory */
  ierr = MatFDColoringDestroy(fdcoloring);CHKERRQ(ierr);
  ierr = VecDestroy(dvec);CHKERRQ(ierr);
  ierr = VecDestroy(lvec);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr); 
  ierr = ISColoringDestroy(coloring);CHKERRQ(ierr); 
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
  




















