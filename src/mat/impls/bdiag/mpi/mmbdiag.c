#ifndef lint
static char vcid[] = "$Id: mmbdiag.c,v 1.3 1995/05/11 18:50:04 curfman Exp bsmith $";
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
  Mat_BDiag    *A = (Mat_BDiag *) (mbd->A->data);  
  int          ierr, N = mbd->N;
  IS           from, to;
  Vec          gvec;

  int i, high, low, iglobal, lsize, mytid;
  Scalar zero = 0.0, value;

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

  MPI_Comm_rank(mat->comm,&mytid);
  ierr = VecSet(&zero,mbd->lvec); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(gvec,&low,&high); CHKERRA(ierr);
  ierr = VecGetSize(gvec,&lsize); CHKERRA(ierr);
  printf("[%d] low=%d, high=%d \n", mytid, low, high);
  for ( i=0; i<lsize; i++ ) {
    iglobal = i + low; value = (Scalar) (i + 100*mytid);
    ierr = VecSetValues(gvec,1,&iglobal,&value,INSERTVALUES); CHKERRA(ierr);
  }

  ierr = VecScatterBegin(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(gvec,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERR(ierr);

  printf("processor %d\n", mytid);
  ierr = VecView(mbd->lvec,STDOUT_VIEWER); CHKERR(ierr);

  ierr = VecDestroy(gvec); CHKERR(ierr);

  return 0;
}

