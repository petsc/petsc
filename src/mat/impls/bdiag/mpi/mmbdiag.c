#define PETSCMAT_DLL

/*
   Support for the MPIBDIAG matrix-vector multiply
*/
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPIBDiag"
PetscErrorCode MatSetUpMultiply_MPIBDiag(Mat mat)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  Mat_SeqBDiag   *lmbd = (Mat_SeqBDiag*)mbd->A->data;
  PetscErrorCode ierr;
  PetscInt       N = mat->N,*indices,*garray,ec=0;
  PetscInt       bs = mat->bs,d,i,j,diag;
  IS             to,from;
  Vec            gvec;

  PetscFunctionBegin;
  /* We make an array as long as the number of columns */
  /* mark those columns that are in mbd->A */
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,N*sizeof(PetscInt));CHKERRQ(ierr);

  if (bs == 1) {
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
          if (!indices[bs*j]) ec += bs; 
          for (i=0; i<bs; i++) indices[bs*j+i] = 1;
        }
      } else { /* col = loc-diag */
        for (j=0; j<lmbd->bdlen[d]; j++) {
          if (!indices[bs*(j-diag)]) ec += bs; 
          for (i=0; i<bs; i++) indices[bs*(j-diag)+i] = 1;
        }
      }
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ec   = 0;
  for (i=0; i<N; i++) {
    if (indices[i]) garray[ec++] = i;
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&mbd->lvec);CHKERRQ(ierr);

  /* create temporary index set for building scatter-gather */
  ierr = ISCreateGeneral(mat->comm,ec,garray,&from);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ec,garray,&to);CHKERRQ(ierr);
  ierr = PetscFree(garray);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  /*
     This is not correct for a rectangular matrix mbd->m? 
  */
  ierr = VecCreateMPI(mat->comm,mat->m,mat->N,&gvec);CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,from,mbd->lvec,to,&mbd->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,mbd->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,mbd->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,gvec);CHKERRQ(ierr);

  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = VecDestroy(gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
