#ifndef lint
static char vcid[] = "$Id: mmaij.c,v 1.29 1996/08/08 14:42:52 bsmith Exp bsmith $";
#endif


/*
   Support for the parallel AIJ matrix vector multiply
*/
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/vec/vecimpl.h"

int MatSetUpMultiply_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *B = (Mat_SeqAIJ *) (aij->B->data);  
  int        N = aij->N,i,j,*indices,*aj = B->j,ierr,ec = 0,*garray;
  int        shift = B->indexshift;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in aij->B */
  indices = (int *) PetscMalloc( N*sizeof(int) ); CHKPTRQ(indices);
  PetscMemzero(indices,N*sizeof(int));
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      if (!indices[aj[B->i[i] +shift + j] + shift]) ec++; 
      indices[aj[B->i[i] + shift + j] + shift] = 1;
    }
  }

  /* form array of columns we need */
  garray = (int *) PetscMalloc( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i-shift;
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      aj[B->i[i] + shift + j] = indices[aj[B->i[i] + shift + j]+shift];
    }
  }
  B->n = ec;
  PetscFree(indices);
  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,ec,&aij->lvec); CHKERRQ(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateGeneral(MPI_COMM_SELF,ec,garray,&from); CHKERRQ(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,ec,0,1,&to); CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,aij->n,aij->N,&gvec); CHKERRQ(ierr);

  /* generate the scatter context */
  ierr = VecScatterCreate(gvec,from,aij->lvec,to,&aij->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->Mvctx);
  PLogObjectParent(mat,aij->lvec);
  PLogObjectParent(mat,from);
  PLogObjectParent(mat,to);
  aij->garray = garray;
  PLogObjectMemory(mat,(ec+1)*sizeof(int));
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecDestroy(gvec);
  return 0;
}


/*
     Takes the local part of an already assembled MPIAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
int DisAssemble_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) A->data;
  Mat        B = aij->B,Bnew;
  Mat_SeqAIJ *Baij = (Mat_SeqAIJ*)B->data;
  int        ierr,i,j,m=Baij->m,n = aij->N,col,ct = 0,*garray = aij->garray;
  int        *nz,ec,shift = Baij->indexshift;
  Scalar     v;

  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(aij->lvec,&ec); /* needed for PLogObjectMemory below */
  ierr = VecDestroy(aij->lvec); CHKERRQ(ierr); aij->lvec = 0;
  ierr = VecScatterDestroy(aij->Mvctx); CHKERRQ(ierr); aij->Mvctx = 0;
  if (aij->colmap) {
    PetscFree(aij->colmap); aij->colmap = 0;
    PLogObjectMemory(A,-Baij->n*sizeof(int));
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  nz = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRQ(nz);
  for ( i=0; i<m; i++ ) {
    nz[i] = Baij->i[i+1]-Baij->i[i];
  }
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,m,n,0,nz,&Bnew); CHKERRQ(ierr);
  PetscFree(nz);
  for ( i=0; i<m; i++ ) {
    for ( j=Baij->i[i]+shift; j<Baij->i[i+1]+shift; j++ ) {
      col = garray[Baij->j[ct]+shift];
      v = Baij->a[ct++];
      ierr = MatSetValues(Bnew,1,&i,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  PetscFree(aij->garray); aij->garray = 0;
  PLogObjectMemory(A,-ec*sizeof(int));
  ierr = MatDestroy(B); CHKERRQ(ierr);
  PLogObjectParent(A,Bnew);
  aij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  return 0;
}


