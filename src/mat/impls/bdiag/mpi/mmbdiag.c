#ifndef lint
static char vcid[] = "$Id: mmbdiag.c,v 1.12 1995/09/11 18:48:48 bsmith Exp bsmith $";
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
  Mat_SeqBDiag    *lmbd = (Mat_SeqBDiag *) mbd->A->data;
  int          ierr, N = mbd->N, *indices, *garray, ec=0;
  int          nb = lmbd->nb, d, i, j, diag;
  IS           to_from;
  Vec          gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in mbd->A */
  indices = (int *) PETSCMALLOC( N*sizeof(int) ); CHKPTRQ(indices);
  PETSCMEMSET(indices,0,N*sizeof(int));

  if (nb == 1) {
    for (d=0; d<lmbd->nd; d++) {
      diag = lmbd->diag[d];
      if (diag > 0) { /* col = loc */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[j]) ec++; 
          indices[j] = 1;
        }
      } else { /* col = loc-diag */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[j-diag]) ec++; 
          indices[j-diag] = 1;
        }
      }
    }
  } else {
    for (d=0; d<lmbd->nd; d++) {
      diag = lmbd->diag[d];
      if (diag > 0) { /* col = loc */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[nb*j]) ec += nb; 
          for (i=0; i<nb; i++) indices[nb*j+i] = 1;
        }
      } else { /* col = loc-diag */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[nb*(j-diag)]) ec += nb; 
          for (i=0; i<nb; i++) indices[nb*(j-diag)+i] = 1;
        }
      }
    }
  }

  /* form array of columns we need */
  garray = (int *) PETSCMALLOC( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }
  PETSCFREE(indices);

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&mbd->lvec); CHKERRQ(ierr);

  /* create temporary index set for building scatter-gather */
  ierr = ISCreateSeq(MPI_COMM_SELF,ec,garray,&to_from); CHKERRQ(ierr);
  CHKERRQ(ierr);
  PETSCFREE(garray);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/

  ierr = VecCreateMPI(mat->comm,PETSC_DECIDE,mbd->N,&gvec); CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCtxCreate(gvec,to_from,mbd->lvec,to_from,&mbd->Mvctx); 
  PLogObjectParent(mat,to_from);
  CHKERRQ(ierr);
  PLogObjectParent(mat,mbd->Mvctx);
  PLogObjectParent(mat,mbd->lvec);
  ierr = ISDestroy(to_from); CHKERRQ(ierr);
  ierr = VecDestroy(gvec); CHKERRQ(ierr);
  return 0;
}
