/*$Id: ex18.c,v 1.24 2001/01/15 21:45:13 bsmith Exp bsmith $*/

/* np = 1 */

static char help[] = "Compares BLAS dots on different machines. Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int          n = 15,ierr,i;
  Scalar       v;
  Vec          x,y;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  if (n < 5) n = 5;


  /* create two vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    v = ((double)i) + 1.0/(((double)i) + .35);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v += 1.375547826473644376;
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  ierr = VecDot(x,y,&v);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Vector inner product %16.12e\n",v);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
 
