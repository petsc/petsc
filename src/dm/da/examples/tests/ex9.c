/*$Id: ex9.c,v 1.13 2001/01/15 21:49:08 bsmith Exp balay $*/
      
static char help[] = "Tests DAGetColoring() in 3d.\n\n";

#include "petscmat.h"
#include "petscda.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            i,M = 3,N = 5,P=3,s=1,w=2;
  int            m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,ierr;
  int            *lx = PETSC_NULL,*ly = PETSC_NULL,*lz = PETSC_NULL;
  DA             da;
  PetscTruth     flg,test_order;
  ISColoring     coloring;
  Mat            mat;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  Vec            lvec,dvec;
  MatFDColoring  fdcoloring;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-star",&flg);CHKERRA(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = PetscOptionsHasName(PETSC_NULL,"-test_order",&test_order);CHKERRA(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-distribute",&flg);CHKERRA(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRA(1,"Must set -m option with -distribute option");
    ierr = PetscMalloc(m*sizeof(int),&lx);CHKERRQ(ierr);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRA(1,"Must set -n option with -distribute option");
    ierr = PetscMalloc(n*sizeof(int),&ly);CHKERRQ(ierr);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRA(1,"Must set -p option with -distribute option");
    ierr = PetscMalloc(p*sizeof(int),&lz);CHKERRQ(ierr);
    for (i=0; i<p-1; i++) { lz[i] = 2;}
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,stencil_type,M,N,P,m,n,p,w,s,
                    lx,ly,lz,&da);CHKERRA(ierr);
  if (lx) {
    ierr = PetscFree(lx);CHKERRA(ierr);
    ierr = PetscFree(ly);CHKERRA(ierr);
    ierr = PetscFree(lz);CHKERRA(ierr);
  }

  ierr = DAGetColoring(da,&coloring,&mat);CHKERRA(ierr);
  ierr = MatFDColoringCreate(mat,coloring,&fdcoloring);CHKERRA(ierr); 

  ierr = DACreateGlobalVector(da,&dvec);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&lvec);CHKERRA(ierr);

  /* Free memory */
  ierr = MatFDColoringDestroy(fdcoloring);CHKERRA(ierr);
  ierr = VecDestroy(dvec);CHKERRA(ierr);
  ierr = VecDestroy(lvec);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr); 
  ierr = ISColoringDestroy(coloring);CHKERRA(ierr); 
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















