#ifndef lint
static char vcid[] = "$Id: mmdense.c,v 1.4 1995/10/23 23:14:27 curfman Exp bsmith $";
#endif

/*
   Support for the parallel dense matrix vector multiply
*/
#include "mpidense.h"
#include "vec/vecimpl.h"

int MatSetUpMultiply_MPIDense(Mat mat)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr,n;
  IS           tofrom;
  Vec          gvec;

  /* Create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,mdn->N,&mdn->lvec); CHKERRQ(ierr);

  /* Create temporary index set for building scatter gather */
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,mdn->N,0,1,&tofrom); CHKERRQ(ierr);

  /* Create temporary global vector to generate scatter context */
  n    = mdn->cowners[mdn->rank+1] - mdn->cowners[mdn->rank];
  ierr = VecCreateMPI(mat->comm,n,mdn->N,&gvec); CHKERRQ(ierr);

  /* Generate the scatter context */
  ierr = VecScatterCreate(gvec,tofrom,mdn->lvec,tofrom,&mdn->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,mdn->Mvctx);
  PLogObjectParent(mat,mdn->lvec);
  PLogObjectParent(mat,tofrom);
  PLogObjectParent(mat,gvec);

  ierr = ISDestroy(tofrom); CHKERRQ(ierr);
  ierr = VecDestroy(gvec); CHKERRQ(ierr);
  return 0;
}



