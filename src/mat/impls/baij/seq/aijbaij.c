
#include <../src/mat/impls/baij/seq/baij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqAIJ     *b;
  PetscBool      roworiented;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
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
    CHKERRQ(PetscMalloc1(n*bs,&rowlengths));
    for (i=0; i<n; i++) {
      maxlen = PetscMax(maxlen,(ai[i+1] - ai[i]));
      for (j=0; j<bs; j++) {
        rowlengths[i*bs+j] = bs*(ai[i+1] - ai[i]);
      }
    }
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&B));
    CHKERRQ(MatSetType(B,MATSEQAIJ));
    CHKERRQ(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    CHKERRQ(MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs));
    CHKERRQ(MatSeqAIJSetPreallocation(B,0,rowlengths));
    CHKERRQ(PetscFree(rowlengths));
  }
  b = (Mat_SeqAIJ*)B->data;
  roworiented = b->roworiented;

  CHKERRQ(MatSetOption(B,MAT_ROW_ORIENTED,PETSC_FALSE));
  CHKERRQ(PetscMalloc1(bs,&rows));
  CHKERRQ(PetscMalloc1(bs*maxlen,&cols));
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
    CHKERRQ(MatSetValues(B,bs,rows,bs*ncols,cols,aa,INSERT_VALUES));
    aa  += ncols*bs*bs;
  }
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(B,MAT_ROW_ORIENTED,roworiented));

  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(A,&B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/aij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqBAIJ    *b;
  PetscInt       *ai=a->i,m=A->rmap->N,n=A->cmap->N,i,*rowlengths,bs=PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    CHKERRQ(PetscMalloc1(m/bs,&rowlengths));
    for (i=0; i<m/bs; i++) {
      rowlengths[i] = (ai[i*bs+1] - ai[i*bs])/bs;
    }
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&B));
    CHKERRQ(MatSetSizes(B,m,n,m,n));
    CHKERRQ(MatSetType(B,MATSEQBAIJ));
    CHKERRQ(MatSeqBAIJSetPreallocation(B,bs,0,rowlengths));
    CHKERRQ(PetscFree(rowlengths));
  } else B = *newmat;

  if (bs == 1) {
    b = (Mat_SeqBAIJ*)(B->data);

    CHKERRQ(PetscArraycpy(b->i,a->i,m+1));
    CHKERRQ(PetscArraycpy(b->ilen,a->ilen,m));
    CHKERRQ(PetscArraycpy(b->j,a->j,a->nz));
    CHKERRQ(PetscArraycpy(b->a,a->a,a->nz));

    CHKERRQ(MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE));
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  } else {
    /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
    /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
    /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
    CHKERRQ(MatConvert_Basic(A,newtype,MAT_REUSE_MATRIX,&B));
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(A,&B));
  } else *newmat = B;
  PetscFunctionReturn(0);
}
