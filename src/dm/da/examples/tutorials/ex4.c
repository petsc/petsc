
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates various vector routines.\n\n";

/*T
   Concepts: mathematical functions
   Processors: n
T*/

/* 
  Include "petscpf.h" so that we can use pf functions and "petscda.h" so
 we can use the PETSc distributed arrays
*/

#include "petscpf.h"
#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "myfunction"
PetscErrorCode myfunction(void *ctx,PetscInt n,PetscScalar *xy,PetscScalar *u)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    u[2*i] = xy[2*i];
    u[2*i+1] = xy[2*i+1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            u,xy;
  DA             da;
  PetscErrorCode ierr;
  PetscInt       m = 10, n = 10, dof = 2;
  PF             pf;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
 
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&xy);CHKERRQ(ierr);

  ierr = DACreatePF(da,&pf);CHKERRQ(ierr);
  ierr = PFSet(pf,myfunction,0,0,0,0);CHKERRQ(ierr);
  ierr = PFSetFromOptions(pf);CHKERRQ(ierr);

  ierr = PFApplyVec(pf,xy,u);CHKERRQ(ierr);

  ierr = VecView(u,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(xy);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
