#ifndef lint
static char vcid[] = "$Id: mmbdiag.c,v 1.7 1995/05/26 19:23:48 curfman Exp bsmith $";
#endif

/*
   Support for the MPIBDIAG matrix-vector multiply
*/
#include "mpibdiag.h"
#include "vec/vecimpl.h"
#include "../seq/bdiag.h"

int MatSetUpMultiply_MPIBDiag(Mat mat)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr, N = mbd->N;
  IS           from, to;
  Vec          gvec;

  /* create local vector that is used to scatter into */
  ierr = VecCreateSequential(MPI_COMM_SELF,N,&mbd->lvec); CHKERRQ(ierr);

  /* create two temporary Index sets for building scatter-gather */
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&from); CHKERRQ(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&to); CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/

  /* We should really associate a vector with the matrix!! */
  ierr = VecCreateMPI(mat->comm,PETSC_DECIDE,mbd->N,&gvec); CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCtxCreate(gvec,from,mbd->lvec,to,&mbd->Mvctx); 
  CHKERRQ(ierr);
  PLogObjectParent(mat,mbd->Mvctx);
  PLogObjectParent(mat,mbd->lvec);
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);

  ierr = VecScatterBegin(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);

  ierr = VecDestroy(gvec); CHKERRQ(ierr);

  return 0;
}

