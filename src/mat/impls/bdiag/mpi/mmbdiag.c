#ifndef lint
static char vcid[] = "$Id: mmbdiag.c,v 1.6 1995/05/18 22:14:30 curfman Exp curfman $";
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
  ierr = VecCreateSequential(MPI_COMM_SELF,N,&mbd->lvec); CHKERR(ierr);

  /* create two temporary Index sets for building scatter-gather */
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&from); CHKERR(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,N,0,1,&to); CHKERR(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/

  /* We should really associate a vector with the matrix!! */
  ierr = VecCreateMPI(mat->comm,PETSC_DECIDE,mbd->N,&gvec); CHKERR(ierr);

  /* generate the scatter context */
  ierr = VecScatterCtxCreate(gvec,from,mbd->lvec,to,&mbd->Mvctx); 
  CHKERR(ierr);
  PLogObjectParent(mat,mbd->Mvctx);
  PLogObjectParent(mat,mbd->lvec);
  ierr = ISDestroy(from); CHKERR(ierr);
  ierr = ISDestroy(to); CHKERR(ierr);

  ierr = VecScatterBegin(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERR(ierr);

  ierr = VecDestroy(gvec); CHKERR(ierr);

  return 0;
}

