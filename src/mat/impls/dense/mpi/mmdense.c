#ifndef lint
static char vcid[] = "$Id: mmdense.c,v 1.2 1995/10/23 21:11:46 curfman Exp curfman $";
#endif

/*
   Support for the parallel dense matrix vector multiply
*/
#include "mpidense.h"
#include "vec/vecimpl.h"

int MatSetUpMultiply_MPIDense(Mat mat)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;
  IS           to, from;
  Vec          gvec;

  int mytid, low, high, lsize, i, iglobal;
   Scalar  zero = 0.0, value;

  /* Create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,mdn->N,&mdn->lvec); CHKERRQ(ierr);


  /* Create temporary index set for building scatter gather */
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,mdn->N,0,1,&to); CHKERRQ(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,mdn->N,0,1,&from); CHKERRQ(ierr);

  /* Create temporary global vector to generate scatter context */
  ierr = VecCreateMPI(mat->comm,PETSC_DECIDE,mdn->N,&gvec); CHKERRQ(ierr);

  MPI_Comm_rank(mat->comm,&mytid);
  ierr = VecSet(&zero,mdn->lvec); CHKERRA(ierr);
  ierr = VecGetOwnershipRange(gvec,&low,&high); CHKERRA(ierr);
  ierr = VecGetLocalSize(gvec,&lsize); CHKERRA(ierr);
  printf("[%d] low=%d, high=%d, mdn->n=%d, mdn->N=%d \n", 
           mytid, low, high,mdn->n,mdn->N);
  for ( i=0; i<lsize; i++ ) {
    iglobal = i + low; value = (Scalar) (i + 100*mytid);
    printf("[%d] i=%d, ig=%d, val=%g\n",mytid,i,iglobal,value);
    ierr = VecSetValues(gvec,1,&iglobal,&value,INSERT_VALUES); CHKERRA(ierr);
  }


  /* Generate the scatter context */
  ierr = VecScatterCreate(gvec,from,mdn->lvec,to,&mdn->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,mdn->Mvctx);
  PLogObjectParent(mat,mdn->lvec);
  PLogObjectParent(mat,from);
  PLogObjectParent(mat,to);

  ierr = VecScatterBegin(gvec,mdn->lvec,ADD_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(gvec,mdn->lvec,ADD_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);

  printf("processor %d\n", mytid);
  ierr = VecView(mdn->lvec,STDOUT_VIEWER_SELF); CHKERRQ(ierr);
  MPI_Barrier(MPI_COMM_WORLD);
  PetscFinalize();
  exit(0);

  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecDestroy(gvec);
  return 0;
}



