#ifndef lint
static char vcid[] = "$Id: petscadic.c,v 1.1 1997/03/28 04:08:42 bsmith Exp bsmith $";
#endif

#include "petscadic.h"

struct _PetscADICFunction {
  Vec din,dout;
  int (*Function)(Vec,Vec);
};

/*
   PetscADICFunctionCreate - Creates a data structure to manage the evaluate
                             of a PETSc function and its derivative.
*/
int PetscADICFunctionCreate(Vec in,Vec out,int (*Function)(Vec,Vec),PetscADICFunction*ctx)
{
  int ierr,n,m;
 
  *ctx = PetscNew(struct _PetscADICFunction); CHKPTRQ(*ctx);

  ierr = VecGetSize(in,&m); CHKERRQ(ierr);
  ierr = VecGetSize(out,&n); CHKERRQ(ierr);

  ierr = ad_PetscADICFunctionCreate(*ctx,m,n); CHKERRQ(ierr);
  return 0;
}

/*
    PetscADICFunctionEvaluate - Evaluates a given PETSc function and its derivative
*/
int PetscADICFunctionEvaluate(Vec in,Vec out,Mat grad,PetscADICFunction ctx)
{
  int    ierr,m,n;
  Scalar *inx,*outx;

  ierr = VecGetArray(in,&inx); CHKERRQ(ierr);
  ierr = VecGetArray(out,&outx); CHKERRQ(ierr);

  ierr = ad_PetscADICFunctionEvaluate(inx,outx,m,n,ctx); CHKERRQ(ierr);

  ierr = VecRestoreArray(in,&inx); CHKERRQ(ierr);
  ierr = VecRestoreArray(out,&outx); CHKERRQ(ierr);
  return 0;
}
