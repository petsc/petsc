
#include <../src/mat/impls/baij/seq/baij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqAIJ     *b;
  PetscBool      roworiented;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       bs = A->rmap->bs,*ai = a->i,*aj = a->j,n = A->rmap->N/bs,i,j,k;
  PetscInt       *rowlengths,*rows,*cols,maxlen = 0,ncols;
  MatScalar      *aa = a->a;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    for (i=0; i<n; i++) {
      maxlen = PetscMax(maxlen,(ai[i+1] - ai[i]));
    }
  } else {
    ierr = PetscMalloc1(n*bs,&rowlengths);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      maxlen = PetscMax(maxlen,(ai[i+1] - ai[i]));
      for (j=0; j<bs; j++) {
        rowlengths[i*bs+j] = bs*(ai[i+1] - ai[i]);
      }
    }
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,0,rowlengths);CHKERRQ(ierr);
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  }
  b = (Mat_SeqAIJ*)B->data;
  roworiented = b->roworiented;

  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs*maxlen,&cols);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      rows[j] = i*bs+j;
    }
    ncols = ai[i+1] - ai[i];
    for (k=0; k<ncols; k++) {
      for (j=0; j<bs; j++) {
        cols[k*bs+j] = bs*(*aj) + j;
      }
      aj++;
    }
    ierr = MatSetValues(B,bs,rows,bs*ncols,cols,aa,INSERT_VALUES);CHKERRQ(ierr);
    aa  += ncols*bs*bs;
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED,roworiented);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else *newmat = B;
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/aij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,m=A->rmap->N,n=A->cmap->N,i,*rowlengths,bs=PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc1(m/bs,&rowlengths);CHKERRQ(ierr);
    for (i=0; i<m/bs; i++) {
      rowlengths[i] = (ai[i*bs+1] - ai[i*bs])/bs;
    }
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
    ierr = MatSetType(B,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(B,bs,0,rowlengths);CHKERRQ(ierr);
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  } else B = *newmat;

  if (bs == 1) {
    b = (Mat_SeqBAIJ*)(B->data);

    ierr = PetscArraycpy(b->i,a->i,m+1);CHKERRQ(ierr);
    ierr = PetscArraycpy(b->ilen,a->ilen,m);CHKERRQ(ierr);
    ierr = PetscArraycpy(b->j,a->j,a->nz);CHKERRQ(ierr);
    ierr = PetscArraycpy(b->a,a->a,a->nz);CHKERRQ(ierr);

    ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
    /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
    /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
    ierr = MatConvert_Basic(A,newtype,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else *newmat = B;
  PetscFunctionReturn(0);
}
