#ifndef lint
static char vcid[] = "$Id: mmbaij.c,v 1.3 1996/06/19 23:03:42 balay Exp bsmith $";
#endif


/*
   Support for the parallel BAIJ matrix vector multiply
*/
#include "mpibaij.h"
#include "src/vec/vecimpl.h"
#include "../seq/baij.h"

int MatSetUpMultiply_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) mat->data;
  Mat_SeqBAIJ *B = (Mat_SeqBAIJ *) (baij->B->data);  
  int        Nbs = baij->Nbs,i,j,*indices,*aj = B->j,ierr,ec = 0,*garray;
  int        col,bs = baij->bs,*tmp;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  indices = (int *) PetscMalloc( Nbs*sizeof(int) ); CHKPTRQ(indices);
  PetscMemzero(indices,Nbs*sizeof(int));
  for ( i=0; i<B->mbs; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j] ] = 1;
    }
  }

  /* form array of columns we need */
  garray = (int *) PetscMalloc( (ec+1)*sizeof(int) ); CHKPTRQ(garray);
  tmp    = (int *) PetscMalloc( (ec*bs+1)*sizeof(int) ); CHKPTRQ(tmp)
  ec = 0;
  for ( i=0; i<Nbs; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->mbs; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
    }
  }
  B->nbs = ec;
  B->n   = ec*B->bs;
  PetscFree(indices);
  
  for ( i=0,col=0; i<ec; i++ ) {
    for ( j=0; j<bs; j++,col++) tmp[col] = garray[i]*bs+j;
  }
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(MPI_COMM_SELF,ec*bs,&baij->lvec); CHKERRQ(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateSeq(MPI_COMM_SELF,ec*bs,tmp,&from); CHKERRQ(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,ec*bs,0,1,&to); CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,baij->n,baij->N,&gvec); CHKERRQ(ierr);

  /* gnerate the scatter context */
  ierr = VecScatterCreate(gvec,from,baij->lvec,to,&baij->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,baij->Mvctx);
  PLogObjectParent(mat,baij->lvec);
  PLogObjectParent(mat,from);
  PLogObjectParent(mat,to);
  baij->garray = garray;
  PLogObjectMemory(mat,(ec+1)*sizeof(int));
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  ierr = VecDestroy(gvec);
  PetscFree(tmp);
  return 0;
}


/*
     Takes the local part of an already assembled MPIBAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
int DisAssemble_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *) A->data;
  Mat        B = baij->B,Bnew;
  Mat_SeqBAIJ *Bbaij = (Mat_SeqBAIJ*)B->data;
  int        ierr,i,j,mbs=Bbaij->mbs,n = baij->N,col,*garray=baij->garray;
  int        k,bs=baij->bs,bs2=baij->bs2,*rvals,*nz,ec,m=Bbaij->m;
  Scalar     *a=Bbaij->a;

  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(baij->lvec,&ec); /* needed for PLogObjectMemory below */
  ierr = VecDestroy(baij->lvec); CHKERRQ(ierr); baij->lvec = 0;
  ierr = VecScatterDestroy(baij->Mvctx); CHKERRQ(ierr); baij->Mvctx = 0;
  if (baij->colmap) {
    PetscFree(baij->colmap); baij->colmap = 0;
    PLogObjectMemory(A,-Bbaij->nbs*sizeof(int));
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  nz = (int *) PetscMalloc( mbs*sizeof(int) ); CHKPTRQ(nz);
  for ( i=0; i<mbs; i++ ) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,baij->bs,m,n,0,nz,&Bnew); CHKERRQ(ierr);
  PetscFree(nz);
  
  rvals = (int *) PetscMalloc(bs*sizeof(int)); CHKPTRQ(rvals);
  for ( i=0; i<mbs; i++ ) {
    rvals[0] = bs*i;
    for ( j=1; j<bs; j++ ) { rvals[j] = rvals[j-1] + 1; }
    for ( j=Bbaij->i[i]; j<Bbaij->i[i+1]; j++ ) {
      col = garray[Bbaij->j[j]]*bs;
      for (k=0; k<bs; k++ ) {
        ierr = MatSetValues(Bnew,bs,rvals,1,&col,a+j*bs2,INSERT_VALUES);CHKERRQ(ierr);
        col++;
      }
    }
  }
  PetscFree(baij->garray); baij->garray = 0;
  PetscFree(rvals);
  PLogObjectMemory(A,-ec*sizeof(int));
  ierr = MatDestroy(B); CHKERRQ(ierr);
  PLogObjectParent(A,Bnew);
  baij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  return 0;
}


