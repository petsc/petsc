/*$Id: ex1.c,v 1.4 2000/09/22 20:47:35 bsmith Exp bsmith $*/

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

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

#undef __FUNC__
#define __FUNC__ "myfunction"
int myfunction(void *ctx,int n,Scalar *xy,Scalar *u)
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    u[2*i] = xy[2*i];
    u[2*i+1] = xy[2*i+1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Vec        u,xy;
  DA         da;
  int        ierr, m = 10, n = 10, dof = 2;
  PF         pf;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRA(ierr);
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&u);CHKERRA(ierr);
  ierr = DAGetCoordinates(da,&xy);CHKERRA(ierr);

  ierr = DACreatePF(da,&pf);CHKERRA(ierr);
  ierr = PFSet(pf,myfunction,0,0,0,0);CHKERRA(ierr);

  ierr = PFApplyVec(pf,xy,u);CHKERRA(ierr);

  ierr = VecView(u,PETSC_VIEWER_DRAW_WORLD);CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = PFDestroy(pf);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
