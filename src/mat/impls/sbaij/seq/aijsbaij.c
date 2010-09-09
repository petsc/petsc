#define PETSCMAT_DLL

#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/impls/sbaij/seq/sbaij.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqSBAIJ_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqSBAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  Mat            B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  Mat_SeqAIJ     *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,*aj=a->j,m=A->rmap->N,n=A->cmap->n,i,j,k,*bi,*bj,*rowlengths,nz,*rowstart,itmp;
  PetscInt       bs=A->rmap->bs,bs2=bs*bs,mbs=A->rmap->N/bs,diagcnt=0;
  MatScalar      *av,*bv;

  PetscFunctionBegin;
  /* compute rowlengths of newmat */
  ierr = PetscMalloc2(m,PetscInt,&rowlengths,m+1,PetscInt,&rowstart);CHKERRQ(ierr);
  
  for (i=0; i<mbs; i++) rowlengths[i*bs] = 0;
  aj = a->j;
  k = 0;
  for (i=0; i<mbs; i++) {
    nz = ai[i+1] - ai[i];
    if (nz) {
      rowlengths[k] += nz;   /* no. of upper triangular blocks */
      if (*aj == i) {aj++;diagcnt++;nz--;} /* skip diagonal */
      for (j=0; j<nz; j++) { /* no. of lower triangular blocks */
	rowlengths[(*aj)*bs]++; aj++;
      }
    }
    rowlengths[k] *= bs;
    for (j=1; j<bs; j++) {
      rowlengths[k+j] = rowlengths[k];
    }
    k += bs;
    /* printf(" rowlengths[%d]: %d\n",i, rowlengths[i]); */
  }
  
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,rowlengths);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  B->rmap->bs = A->rmap->bs;

  b  = (Mat_SeqAIJ*)(B->data);
  bi = b->i;
  bj = b->j; 
  bv = b->a; 

  /* set b->i */
  bi[0] = 0; rowstart[0] = 0;
  for (i=0; i<mbs; i++){
    for (j=0; j<bs; j++){
      b->ilen[i*bs+j]  = rowlengths[i*bs];
      rowstart[i*bs+j+1] = rowstart[i*bs+j] + rowlengths[i*bs];
    }
    bi[i+1]     = bi[i] + rowlengths[i*bs]/bs; 
  }
  if (bi[mbs] != 2*a->nz - diagcnt) SETERRQ2(PETSC_ERR_PLIB,"bi[mbs]: %D != 2*a->nz-diagcnt: %D\n",bi[mbs],2*a->nz - diagcnt);

  /* set b->j and b->a */
  aj = a->j; av = a->a;
  for (i=0; i<mbs; i++) {
    nz = ai[i+1] - ai[i];
    /* diagonal block */
    if (nz && *aj == i) {
      nz--;
      for (j=0; j<bs; j++){   /* row i*bs+j */
	itmp = i*bs+j;
	for (k=0; k<bs; k++){ /* col i*bs+k */
	  *(bj + rowstart[itmp]) = (*aj)*bs+k;
	  *(bv + rowstart[itmp]) = *(av+k*bs+j); 
	  rowstart[itmp]++;
	}
      }
      aj++; av += bs2; 
    }
    
    while (nz--){
      /* lower triangular blocks */
      for (j=0; j<bs; j++){   /* row (*aj)*bs+j */
        itmp = (*aj)*bs+j;
        for (k=0; k<bs; k++){ /* col i*bs+k */
          *(bj + rowstart[itmp]) = i*bs+k;
          *(bv + rowstart[itmp]) = *(av+k*bs+j); 
          rowstart[itmp]++;
        }
      }
      /* upper triangular blocks */
      for (j=0; j<bs; j++){   /* row i*bs+j */
        itmp = i*bs+j;
        for (k=0; k<bs; k++){ /* col (*aj)*bs+k */
          *(bj + rowstart[itmp]) = (*aj)*bs+k;
          *(bv + rowstart[itmp]) = *(av+k*bs+j); 
          rowstart[itmp]++;
        }
      }
      aj++; av += bs2;
    }
  }
  ierr = PetscFree2(rowlengths,rowstart);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqAIJ_SeqSBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_SeqSBAIJ(Mat A,const MatType newtype,MatReuse reuse,Mat *newmat) {
  Mat            B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  Mat_SeqSBAIJ   *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,*aj,m=A->rmap->N,n=A->cmap->N,i,j,*bi,*bj,*rowlengths;
  MatScalar      *av,*bv;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");

  ierr = PetscMalloc(m*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    rowlengths[i] = ai[i+1] - a->diag[i];
  }
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(B,1,0,rowlengths);CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  
  b  = (Mat_SeqSBAIJ*)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  bi[0] = 0;
  for (i=0; i<m; i++) {
    aj = a->j + a->diag[i];
    av = a->a + a->diag[i];    
    for (j=0; j<rowlengths[i]; j++){
      *bj = *aj; bj++; aj++;
      *bv = *av; bv++; av++;
    }
    bi[i+1]    = bi[i] + rowlengths[i];
    b->ilen[i] = rowlengths[i];
  }
 
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A->hermitian){
    ierr = MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqSBAI_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqSBAIJ_SeqBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  Mat            B;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  Mat_SeqBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,*aj=a->j,m=A->rmap->N,n=A->cmap->n,i,k,*bi,*bj,*browlengths,nz,*browstart,itmp;
  PetscInt       bs=A->rmap->bs,bs2=bs*bs,mbs=m/bs;
  MatScalar      *av,*bv;

  PetscFunctionBegin;
  /* compute browlengths of newmat */
  ierr = PetscMalloc2(mbs,PetscInt,&browlengths,mbs,PetscInt,&browstart);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) browlengths[i] = 0;
  aj = a->j;
  for (i=0; i<mbs; i++) {
    nz = ai[i+1] - ai[i];
    aj++; /* skip diagonal */
    for (k=1; k<nz; k++) { /* no. of lower triangular blocks */
      browlengths[*aj]++; aj++;
    }
    browlengths[i] += nz;   /* no. of upper triangular blocks */
  }
  
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,browlengths);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  
  b  = (Mat_SeqBAIJ*)(B->data);
  bi = b->i;
  bj = b->j; 
  bv = b->a; 

  /* set b->i */
  bi[0] = 0;
  for (i=0; i<mbs; i++){
    b->ilen[i]    = browlengths[i];
    bi[i+1]       = bi[i] + browlengths[i]; 
    browstart[i]  = bi[i];
  }
  if (bi[mbs] != 2*a->nz - mbs) SETERRQ2(PETSC_ERR_PLIB,"bi[mbs]: %D != 2*a->nz - mbs: %D\n",bi[mbs],2*a->nz - mbs);
  
  /* set b->j and b->a */
  aj = a->j; av = a->a;
  for (i=0; i<mbs; i++) {
    /* diagonal block */
    *(bj + browstart[i]) = *aj; aj++;
    itmp = bs2*browstart[i];
    for (k=0; k<bs2; k++){
      *(bv + itmp + k) = *av; av++; 
    } 
    browstart[i]++;
    
    nz = ai[i+1] - ai[i] -1;
    while (nz--){
      /* lower triangular blocks */   
      *(bj + browstart[*aj]) = i;
      itmp = bs2*browstart[*aj];
      for (k=0; k<bs2; k++){
        *(bv + itmp + k) = *(av + k);
      }
      browstart[*aj]++;

      /* upper triangular blocks */
      *(bj + browstart[i]) = *aj; aj++;
      itmp = bs2*browstart[i];
      for (k=0; k<bs2; k++){
        *(bv + itmp + k) = *av; av++;
      }
      browstart[i]++;
    }
  }
  ierr = PetscFree2(browlengths,browstart);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SeqBAIJ_SeqSBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqBAIJ_SeqSBAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat) 
{
  Mat            B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  Mat_SeqSBAIJ   *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,*aj,m=A->rmap->N,n=A->cmap->n,i,j,k,*bi,*bj,*browlengths;
  PetscInt       bs=A->rmap->bs,bs2=bs*bs,mbs=m/bs,dd;
  MatScalar      *av,*bv;
  PetscTruth     flg;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  ierr = MatMissingDiagonal_SeqBAIJ(A,&flg,&dd);CHKERRQ(ierr); /* check for missing diagonals, then mark diag */
  if (flg) SETERRQ1(PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal %D",dd);
  
  ierr = PetscMalloc(mbs*sizeof(PetscInt),&browlengths);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    browlengths[i] = ai[i+1] - a->diag[i];
  }

  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,newtype);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(B,bs,0,browlengths);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  
  b  = (Mat_SeqSBAIJ*)(B->data);
  bi = b->i;
  bj = b->j;
  bv = b->a;

  bi[0] = 0;
  for (i=0; i<mbs; i++) {
    aj = a->j + a->diag[i];
    av = a->a + (a->diag[i])*bs2;   
    for (j=0; j<browlengths[i]; j++){
      *bj = *aj; bj++; aj++; 
      for (k=0; k<bs2; k++){
        *bv = *av; bv++; av++;
      }
    }
    bi[i+1]    = bi[i] + browlengths[i];
    b->ilen[i] = browlengths[i];
  }
  ierr = PetscFree(browlengths);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
