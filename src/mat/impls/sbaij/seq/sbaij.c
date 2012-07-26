
/*
    Defines the basic matrix operations for the SBAIJ (compressed row)
  matrix storage format.
*/
#include <../src/mat/impls/baij/seq/baij.h>         /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petscblaslapack.h>

#include <../src/mat/impls/sbaij/seq/relax.h>
#define USESHORT
#include <../src/mat/impls/sbaij/seq/relax.h>

extern PetscErrorCode MatSeqSBAIJSetNumericFactorization_inplace(Mat,PetscBool );

/*
     Checks for missing diagonals
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal_SeqSBAIJ"
PetscErrorCode MatMissingDiagonal_SeqSBAIJ(Mat A,PetscBool  *missing,PetscInt *dd)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       *diag,*jj = a->j,i;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqSBAIJ(A);CHKERRQ(ierr);
  *missing = PETSC_FALSE;
  if(A->rmap->n > 0 && !jj) {
    *missing = PETSC_TRUE;
    if (dd) *dd = 0;
    PetscInfo(A,"Matrix has no entries therefore is missing diagonal");
  } else {
    diag = a->diag;
    for (i=0; i<a->mbs; i++) {
      if (jj[diag[i]] != i) {
	*missing    = PETSC_TRUE;
	if (dd) *dd = i;
	break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMarkDiagonal_SeqSBAIJ"
PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr = PetscMalloc(a->mbs*sizeof(PetscInt),&a->diag);CHKERRQ(ierr); 
    ierr = PetscLogObjectMemory(A,a->mbs*sizeof(PetscInt));CHKERRQ(ierr);
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<a->mbs; i++) a->diag[i] = a->i[i];  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_SeqSBAIJ"
static PetscErrorCode MatGetRowIJ_SeqSBAIJ(Mat A,PetscInt oshift,PetscBool  symmetric,PetscBool  blockcompressed,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscBool  *done)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     i,j,n = a->mbs,nz = a->i[n],bs = A->rmap->bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (!blockcompressed) {
    /* malloc & create the natural set of indices */
    ierr = PetscMalloc2((n+1)*bs,PetscInt,ia,nz*bs,PetscInt,ja);CHKERRQ(ierr);
    for (i=0; i<n+1; i++) {
      for (j=0; j<bs; j++) {
        *ia[i*bs+j] = a->i[i]*bs+j+oshift;
      }
    }
    for (i=0; i<nz; i++) {
      for (j=0; j<bs; j++) {
        *ja[i*bs+j] = a->j[i]*bs+j+oshift;
      }
    }
  } else { /* blockcompressed */
    if (oshift == 1) {
      /* temporarily add 1 to i and j indices */
      for (i=0; i<nz; i++) a->j[i]++;
      for (i=0; i<n+1; i++) a->i[i]++;
    }
    *ia = a->i; *ja = a->j;
  }

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_SeqSBAIJ" 
static PetscErrorCode MatRestoreRowIJ_SeqSBAIJ(Mat A,PetscInt oshift,PetscBool  symmetric,PetscBool  blockcompressed,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscBool  *done)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     i,n = a->mbs,nz = a->i[n];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);

  if (!blockcompressed) {
    ierr = PetscFree2(*ia,*ja);CHKERRQ(ierr);
  } else if (oshift == 1) { /* blockcompressed */
    for (i=0; i<nz; i++) a->j[i]--;
    for (i=0; i<n+1; i++) a->i[i]--;
  }

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqSBAIJ"
PetscErrorCode MatDestroy_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, NZ=%D",A->rmap->N,a->nz);
#endif
  ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
  if (a->free_diag){ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  ierr = ISDestroy(&a->row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr);
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = PetscFree(a->idiag);CHKERRQ(ierr);
  ierr = PetscFree(a->inode.size);CHKERRQ(ierr);
  if (a->free_imax_ilen) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = PetscFree(a->sor_work);CHKERRQ(ierr);
  ierr = PetscFree(a->solves_work);CHKERRQ(ierr);
  ierr = PetscFree(a->mult_work);CHKERRQ(ierr);
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);
  ierr = PetscFree(a->xtoy);CHKERRQ(ierr);
  if (a->free_jshort) {ierr = PetscFree(a->jshort);CHKERRQ(ierr);}
  ierr = PetscFree(a->inew);CHKERRQ(ierr);
  ierr = MatDestroy(&a->parent);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqSBAIJSetColumnIndices_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_seqbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_seqsbstrm_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqSBAIJ"
PetscErrorCode MatSetOption_SeqSBAIJ(Mat A,MatOption op,PetscBool  flg)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused = (flg ? -1 : 0);
    break;
  case MAT_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_HERMITIAN:
    if (!A->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatAssemblyEnd() first");
    if (A->cmap->n < 65536 && A->cmap->bs == 1) {
      A->ops->mult = MatMult_SeqSBAIJ_1_Hermitian_ushort;
    } else if (A->cmap->bs == 1) {
      A->ops->mult = MatMult_SeqSBAIJ_1_Hermitian;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for Hermitian with block size greater than 1");
    break;
  case MAT_SPD:
    A->spd_set                         = PETSC_TRUE;
    A->spd                             = flg;
    if (flg) {
      A->symmetric                     = PETSC_TRUE;
      A->structurally_symmetric        = PETSC_TRUE;
      A->symmetric_set                 = PETSC_TRUE;
      A->structurally_symmetric_set    = PETSC_TRUE;
    }
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_SYMMETRY_ETERNAL:
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix must be symmetric");
    ierr = PetscInfo1(A,"Option %s not relevent\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_LOWER_TRIANGULAR:
    a->ignore_ltriangular = flg;
    break;
  case MAT_ERROR_LOWER_TRIANGULAR:
    a->ignore_ltriangular = flg;
    break;
  case MAT_GETROW_UPPERTRIANGULAR:
    a->getrow_utriangular = flg;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqSBAIJ"
PetscErrorCode MatGetRow_SeqSBAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **v)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       itmp,i,j,k,M,*ai,*aj,bs,bn,bp,*cols_i,bs2;
  MatScalar      *aa,*aa_i;
  PetscScalar    *v_i;

  PetscFunctionBegin;
  if (A && !a->getrow_utriangular) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatGetRow is not supported for SBAIJ matrix format. Getting the upper triangular part of row, run with -mat_getrow_uppertriangular, call MatSetOption(mat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE) or MatGetRowUpperTriangular()");
  /* Get the upper triangular part of the row */
  bs  = A->rmap->bs;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  bs2 = a->bs2;
  
  if (row < 0 || row >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Row %D out of range", row);
  
  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block position */
  M   = ai[bn+1] - ai[bn];
  *ncols = bs*M;
  
  if (v) {
    *v = 0;
    if (*ncols) {
      ierr = PetscMalloc((*ncols+row)*sizeof(PetscScalar),v);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        v_i  = *v + i*bs;
        aa_i = aa + bs2*(ai[bn] + i);
        for (j=bp,k=0; j<bs2; j+=bs,k++) {v_i[k] = aa_i[j];}
      }
    }
  }
  
  if (cols) {
    *cols = 0;
    if (*ncols) {
      ierr = PetscMalloc((*ncols+row)*sizeof(PetscInt),cols);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        cols_i = *cols + i*bs;
        itmp  = bs*aj[ai[bn] + i];
        for (j=0; j<bs; j++) {cols_i[j] = itmp++;}
      }
    }
  }
  
  /*search column A(0:row-1,row) (=A(row,0:row-1)). Could be expensive! */
  /* this segment is currently removed, so only entries in the upper triangle are obtained */
#ifdef column_search
  v_i    = *v    + M*bs;
  cols_i = *cols + M*bs;
  for (i=0; i<bn; i++){ /* for each block row */
    M = ai[i+1] - ai[i];
    for (j=0; j<M; j++){
      itmp = aj[ai[i] + j];    /* block column value */
      if (itmp == bn){ 
        aa_i   = aa    + bs2*(ai[i] + j) + bs*bp;
        for (k=0; k<bs; k++) {
          *cols_i++ = i*bs+k; 
          *v_i++    = aa_i[k]; 
        }
        *ncols += bs;
        break;
      }
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqSBAIJ"
PetscErrorCode MatRestoreRow_SeqSBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowUpperTriangular_SeqSBAIJ"
PetscErrorCode MatGetRowUpperTriangular_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  a->getrow_utriangular = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowUpperTriangular_SeqSBAIJ"
PetscErrorCode MatRestoreRowUpperTriangular_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  a->getrow_utriangular = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqSBAIJ"
PetscErrorCode MatTranspose_SeqSBAIJ(Mat A,MatReuse reuse,Mat *B)
{ 
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || *B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqSBAIJ_ASCII"
static PetscErrorCode MatView_SeqSBAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,bs = A->rmap->bs,k,l,bs2=a->bs2;
  PetscViewerFormat format;
  PetscInt          *diag;
  
  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size is %D\n",bs);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    Mat aij;
    if (A->factortype && bs>1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"Warning: matrix is factored with bs>1. MatView() with PETSC_VIEWER_ASCII_MATLAB is not supported and ignored!\n");CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
    ierr = MatView(aij,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&aij);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) { 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",bs*a->j[k]+l,
                                            PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %G i) ",bs*a->j[k]+l,
                                            PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
            }
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
     PetscFunctionReturn(0);
  } else { 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
    if (A->factortype){ /* for factored matrix */
      if (bs>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"matrix is factored with bs>1. Not implemented yet");

      diag=a->diag;
      for (i=0; i<a->mbs; i++) { /* for row block i */
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        /* diagonal entry */
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[diag[i]]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",a->j[diag[i]],PetscRealPart(1.0/a->a[diag[i]]),PetscImaginaryPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[diag[i]]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %G i) ",a->j[diag[i]],PetscRealPart(1.0/a->a[diag[i]]),-PetscImaginaryPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",a->j[diag[i]],PetscRealPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",a->j[diag[i]],1.0/a->a[diag[i]]);CHKERRQ(ierr);
#endif
        /* off-diagonal entries */
        for (k=a->i[i]; k<a->i[i+1]-1; k++) { 
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[k]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",bs*a->j[k],PetscRealPart(a->a[k]),PetscImaginaryPart(a->a[k]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->a[k]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %G i) ",bs*a->j[k],PetscRealPart(a->a[k]),-PetscImaginaryPart(a->a[k]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k],PetscRealPart(a->a[k]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",a->j[k],a->a[k]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      
    } else { /* for non-factored matrix */
      for (i=0; i<a->mbs; i++) { /* for row block i */
        for (j=0; j<bs; j++) {   /* for row bs*i + j */
          ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i*bs+j);CHKERRQ(ierr);
          for (k=a->i[i]; k<a->i[i+1]; k++) { /* for column block */
            for (l=0; l<bs; l++) {            /* for column */
#if defined(PETSC_USE_COMPLEX)
              if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",bs*a->j[k]+l,
                                            PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %G i) ",bs*a->j[k]+l,
                                            PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              }
#else
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
#endif
            } 
          } 
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        } 
      } 
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqSBAIJ_Draw_Zoom"
static PetscErrorCode MatView_SeqSBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat            A = (Mat) Aa;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,mbs=a->mbs,color,bs=A->rmap->bs,bs2=a->bs2;
  PetscMPIInt    rank;
  PetscReal      xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar      *aa;
  MPI_Comm       comm;
  PetscViewer    viewer;
  
  PetscFunctionBegin; 
  /*
    This is nasty. If this is called from an originally parallel matrix
    then all processes call this,but only the first has the matrix so the
    rest should return immediately.
  */
  ierr = PetscObjectGetComm((PetscObject)draw,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);
  
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr); 
  
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  PetscDrawString(draw, .3*(xl+xr), .3*(yl+yr), PETSC_DRAW_BLACK, "symmetric");
  
  /* loop over matrix elements drawing boxes */
  color = PETSC_DRAW_BLUE;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) >=  0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  color = PETSC_DRAW_CYAN;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) != 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  
  color = PETSC_DRAW_RED;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) <= 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqSBAIJ_Draw"
static PetscErrorCode MatView_SeqSBAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      xl,yl,xr,yr,w,h;
  PetscDraw      draw;
  PetscBool      isnull;
  
  PetscFunctionBegin; 
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->rmap->N; yr = A->rmap->N; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqSBAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqSBAIJ"
PetscErrorCode MatView_SeqSBAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw;
  FILE           *file = 0;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii){
    ierr = MatView_SeqSBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqSBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    Mat B;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file) {
      fprintf(file,"-matload_block_size %d\n",(int)A->rmap->bs);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqSBAIJ"
PetscErrorCode MatGetValues_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  PetscInt     *ai = a->i,*ailen = a->ilen;
  PetscInt     brow,bcol,ridx,cidx,bs=A->rmap->bs,bs2=a->bs2;
  MatScalar    *ap,*aa = a->a;
  
  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) {v += n; continue;} /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",row); */
    if (row >= A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
    rp   = aj + ai[brow] ; ap = aa + bs2*ai[brow] ;
    nrow = ailen[brow]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) {v++; continue;} /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",in[l]); */
      if (in[l] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap->n-1);
      col  = in[l] ; 
      bcol = col/bs;
      cidx = col%bs; 
      ridx = row%bs;
      high = nrow; 
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          *v++ = ap[bs2*i+bs*cidx+ridx];
          goto finished;
        }
      } 
      *v++ = 0.0;
      finished:;
    }
  }
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_SeqSBAIJ"
PetscErrorCode MatSetValuesBlocked_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt          *aj=a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs,stepval;
  PetscBool         roworiented=a->roworiented; 
  const PetscScalar *value = v;
  MatScalar         *ap,*aa = a->a,*bap;
  
  PetscFunctionBegin;
  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= a->mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,a->mbs-1);
#endif
    rp   = aj + ai[row]; 
    ap   = aa + bs2*ai[row];
    rmax = imax[row]; 
    nrow = ailen[row]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      col = in[l]; 
#if defined(PETSC_USE_DEBUG)  
      if (col >= a->nbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",col,a->nbs-1);
#endif
      if (col < row) {
        if (a->ignore_ltriangular) {
          continue; /* ignore lower triangular block */
        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
        }
      }
      if (roworiented) { 
        value = v + k*(stepval+bs)*bs + l*bs;
      } else {
        value = v + l*(stepval+bs)*bs + k*bs;
      }
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  bs2*i;
          if (roworiented) { 
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] = *value++; 
                }
               }
            }
          } else {
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++ += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++  = *value++; 
                }
              }
            }
          }
          goto noinsert2;
        }
      } 
      if (nonew == 1) goto noinsert2;
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ierr = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      if (N >= i) {
        ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      rp[i] = col; 
      bap   = ap +  bs2*i;
      if (roworiented) { 
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=ii; jj<bs2; jj+=bs) {
            bap[jj] = *value++; 
          }
        }
      } else {
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=0; jj<bs; jj++) {
            *bap++  = *value++; 
          }
        }
       }
    noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
   PetscFunctionReturn(0);
} 

/*
    This is not yet used 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqSBAIJ_SeqAIJ_Inode"
PetscErrorCode MatAssemblyEnd_SeqSBAIJ_SeqAIJ_Inode(Mat A)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode  ierr;
  const PetscInt  *ai = a->i, *aj = a->j,*cols;
  PetscInt        i = 0,j,blk_size,m = A->rmap->n,node_count = 0,nzx,nzy,*ns,row,nz,cnt,cnt2,*counts;
  PetscBool       flag;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscInt),&ns);CHKERRQ(ierr);
  while (i < m){
    nzx = ai[i+1] - ai[i];       /* Number of nonzeros */
    /* Limits the number of elements in a node to 'a->inode.limit' */
    for (j=i+1,blk_size=1; j<m && blk_size <a->inode.limit; ++j,++blk_size) {
      nzy  = ai[j+1] - ai[j];
      if (nzy != (nzx - j + i)) break;
      ierr = PetscMemcmp(aj + ai[i] + j - i,aj + ai[j],nzy*sizeof(PetscInt),&flag);CHKERRQ(ierr);
      if (!flag) break;
    }
    ns[node_count++] = blk_size;
    i = j;
  }
  if (!a->inode.size && m && node_count > .9*m) {
    ierr = PetscFree(ns);CHKERRQ(ierr);
    ierr = PetscInfo2(A,"Found %D nodes out of %D rows. Not using Inode routines\n",node_count,m);CHKERRQ(ierr);
  } else {
    a->inode.node_count = node_count;
    ierr = PetscMalloc(node_count*sizeof(PetscInt),&a->inode.size);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,node_count*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->inode.size,ns,node_count*sizeof(PetscInt));
    ierr = PetscFree(ns);CHKERRQ(ierr);
    ierr = PetscInfo3(A,"Found %D nodes of %D. Limit used: %D. Using Inode routines\n",node_count,m,a->inode.limit);CHKERRQ(ierr);
    
    /* count collections of adjacent columns in each inode */
    row = 0;
    cnt = 0;
    for (i=0; i<node_count; i++) {
      cols = aj + ai[row] + a->inode.size[i]; 
      nz   = ai[row+1] - ai[row] - a->inode.size[i]; 
      for (j=1; j<nz; j++) {
        if (cols[j] != cols[j-1]+1) {
          cnt++;
        } 
      }
      cnt++;
      row += a->inode.size[i];
    }
    ierr = PetscMalloc(2*cnt*sizeof(PetscInt),&counts);CHKERRQ(ierr);
    cnt = 0;
    row = 0;
    for (i=0; i<node_count; i++) {
      cols          = aj + ai[row] + a->inode.size[i]; 
	  CHKMEMQ;
      counts[2*cnt] = cols[0];
	  CHKMEMQ;
      nz            = ai[row+1] - ai[row] - a->inode.size[i]; 
      cnt2          = 1;
      for (j=1; j<nz; j++) {
        if (cols[j] != cols[j-1]+1) {
	  CHKMEMQ;
          counts[2*(cnt++)+1] = cnt2;
          counts[2*cnt]       = cols[j];
	  CHKMEMQ;
          cnt2                = 1;
        } else cnt2++;
      }
	  CHKMEMQ;
      counts[2*(cnt++)+1] = cnt2;
	  CHKMEMQ;
      row += a->inode.size[i];
    }
    ierr = PetscIntView(2*cnt,counts,0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqSBAIJ"
PetscErrorCode MatAssemblyEnd_SeqSBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m = A->rmap->N,*ip,N,*ailen = a->ilen;
  PetscInt       mbs = a->mbs,bs2 = a->bs2,rmax = 0;
  MatScalar      *aa = a->a,*ap;
  
  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  
  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
     rmax   = PetscMax(rmax,ailen[i]);
     if (fshift) {
       ip = aj + ai[i]; ap = aa + bs2*ai[i];
       N = ailen[i];
       for (j=0; j<N; j++) {
         ip[j-fshift] = ip[j];
         ierr = PetscMemcpy(ap+(j-fshift)*bs2,ap+j*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
       }
     } 
     ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
     ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<mbs; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[mbs]; 
  
  /* diagonals may have moved, reset it */
  if (a->diag) {
    ierr = PetscMemcpy(a->diag,ai,mbs*sizeof(PetscInt));CHKERRQ(ierr);
  } 
  if (fshift && a->nounused == -1) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Unused space detected in matrix: %D X %D block size %D, %D unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);
  }
  ierr = PetscInfo5(A,"Matrix size: %D X %D, block size %D; storage space: %D unneeded, %D used\n",m,A->rmap->N,A->rmap->bs,fshift*bs2,a->nz*bs2);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Most nonzeros blocks in any row is %D\n",rmax);CHKERRQ(ierr);
  A->info.mallocs     += a->reallocs;
  a->reallocs          = 0;
  A->info.nz_unneeded  = (PetscReal)fshift*bs2;
  a->idiagvalid = PETSC_FALSE;

  if (A->cmap->n < 65536 && A->cmap->bs == 1) {
    if (a->jshort && a->free_jshort){
      /* when matrix data structure is changed, previous jshort must be replaced */
      ierr = PetscFree(a->jshort);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(a->i[A->rmap->n]*sizeof(unsigned short),&a->jshort);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,a->i[A->rmap->n]*sizeof(unsigned short));CHKERRQ(ierr);
    for (i=0; i<a->i[A->rmap->n]; i++) a->jshort[i] = a->j[i];
    A->ops->mult  = MatMult_SeqSBAIJ_1_ushort;
    A->ops->sor = MatSOR_SeqSBAIJ_ushort;
    a->free_jshort = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/* 
   This function returns an array of flags which indicate the locations of contiguous
   blocks that should be zeroed. for eg: if bs = 3  and is = [0,1,2,3,5,6,7,8,9]
   then the resulting sizes = [3,1,1,3,1] correspondig to sets [(0,1,2),(3),(5),(6,7,8),(9)]
   Assume: sizes should be long enough to hold all the values.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqSBAIJ_Check_Blocks"
PetscErrorCode MatZeroRows_SeqSBAIJ_Check_Blocks(PetscInt idx[],PetscInt n,PetscInt bs,PetscInt sizes[], PetscInt *bs_max)
{
  PetscInt   i,j,k,row;
  PetscBool  flg;
  
  PetscFunctionBegin;
   for (i=0,j=0; i<n; j++) {
     row = idx[i];
     if (row%bs!=0) { /* Not the begining of a block */
       sizes[j] = 1;
       i++; 
     } else if (i+bs > n) { /* Beginning of a block, but complete block doesn't exist (at idx end) */
       sizes[j] = 1;         /* Also makes sure atleast 'bs' values exist for next else */
       i++; 
     } else { /* Begining of the block, so check if the complete block exists */
       flg = PETSC_TRUE;
       for (k=1; k<bs; k++) {
         if (row+k != idx[i+k]) { /* break in the block */
           flg = PETSC_FALSE;
           break;
         }
       }
       if (flg) { /* No break in the bs */
         sizes[j] = bs;
         i+= bs;
       } else {
         sizes[j] = 1;
         i++;
       }
     }
   }
   *bs_max = j;
   PetscFunctionReturn(0);
}


/* Only add/insert a(i,j) with i<=j (blocks). 
   Any a(i,j) with i>j input by user is ingored. 
*/

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqSBAIJ"
PetscErrorCode MatSetValues_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen,roworiented=a->roworiented;
  PetscInt       *aj=a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscInt       ridx,cidx,bs2=a->bs2;
  MatScalar      *ap,value,*aa=a->a,*bap;
  
  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];       /* row number */ 
    brow = row/bs;      /* block row number */ 
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
#endif
    rp   = aj + ai[brow]; /*ptr to beginning of column value of the row block*/
    ap   = aa + bs2*ai[brow]; /*ptr to beginning of element value of the row block*/
    rmax = imax[brow];  /* maximum space allocated for this row */
    nrow = ailen[brow]; /* actual length of this row */
    low  = 0;
    
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)  
      if (in[l] >= A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->rmap->N-1);
#endif
      col = in[l]; 
      bcol = col/bs;              /* block col number */
      
      if (brow > bcol) {
        if (a->ignore_ltriangular){
          continue; /* ignore lower triangular values */
        } else {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
        }
      }
      
      ridx = row % bs; cidx = col % bs; /*row and col index inside the block */
      if ((brow==bcol && ridx<=cidx) || (brow<bcol)){    
        /* element value a(k,l) */
        if (roworiented) {
          value = v[l + k*n];             
        } else {
          value = v[k + l*m];
        }
        
        /* move pointer bap to a(k,l) quickly and add/insert value */
        if (col <= lastcol) low = 0; high = nrow;
        lastcol = col;
        while (high-low > 7) {
          t = (low+high)/2;
          if (rp[t] > bcol) high = t;
          else              low  = t;
        }
        for (i=low; i<high; i++) {
          if (rp[i] > bcol) break;
          if (rp[i] == bcol) {
            bap  = ap +  bs2*i + bs*cidx + ridx;
            if (is == ADD_VALUES) *bap += value;  
            else                  *bap  = value; 
            /* for diag block, add/insert its symmetric element a(cidx,ridx) */
            if (brow == bcol && ridx < cidx){
              bap  = ap +  bs2*i + bs*ridx + cidx;
              if (is == ADD_VALUES) *bap += value;  
              else                  *bap  = value; 
            }
            goto noinsert1;
          }
        }      
        
        if (nonew == 1) goto noinsert1;
        if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
        
        N = nrow++ - 1; high++;
        /* shift up all the later entries in this row */
        for (ii=N; ii>=i; ii--) {
          rp[ii+1] = rp[ii];
          ierr     = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
        }
        if (N>=i) {
          ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
        }
        rp[i]                      = bcol; 
        ap[bs2*i + bs*cidx + ridx] = value; 
      noinsert1:;
        low = i;      
      }
    }   /* end of loop over added columns */
    ailen[brow] = nrow; 
  }   /* end of loop over added rows */
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactor_SeqSBAIJ"
PetscErrorCode MatICCFactor_SeqSBAIJ(Mat inA,IS row,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)inA->data;
  Mat            outA;
  PetscErrorCode ierr;
  PetscBool      row_identity;
  
  PetscFunctionBegin;
  if (info->levels != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only levels=0 is supported for in-place icc");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  if (!row_identity) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported");
  if (inA->rmap->bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix block size %D is not supported",inA->rmap->bs); /* Need to replace MatCholeskyFactorSymbolic_SeqSBAIJ_MSR()! */

  outA            = inA; 
  inA->factortype = MAT_FACTOR_ICC;
  
  ierr = MatMarkDiagonal_SeqSBAIJ(inA);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetNumericFactorization_inplace(inA,row_identity);CHKERRQ(ierr);

  ierr   = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->row);CHKERRQ(ierr); 
  a->row = row;
  ierr   = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr); 
  a->col = row;
    
  /* Create the invert permutation so that it can be used in MatCholeskyFactorNumeric() */
  if (a->icol) {ierr = ISInvertPermutation(row,PETSC_DECIDE, &a->icol);CHKERRQ(ierr);}
  ierr = PetscLogObjectParent(inA,a->icol);CHKERRQ(ierr);
    
  if (!a->solve_work) {
    ierr = PetscMalloc((inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar),&a->solve_work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(inA,(inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
   
  ierr = MatCholeskyFactorNumeric(outA,inA,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqSBAIJSetColumnIndices_SeqSBAIJ"
PetscErrorCode  MatSeqSBAIJSetColumnIndices_SeqSBAIJ(Mat mat,PetscInt *indices)
{
  Mat_SeqSBAIJ   *baij = (Mat_SeqSBAIJ *)mat->data;
  PetscInt       i,nz,n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nz = baij->maxnz;
  n  = mat->cmap->n;
  for (i=0; i<nz; i++) {
    baij->j[i] = indices[i];
  }
   baij->nz = nz;
   for (i=0; i<n; i++) {
     baij->ilen[i] = baij->imax[i];
   }
  ierr = MatSetOption(mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSeqSBAIJSetColumnIndices"
/*@
  MatSeqSBAIJSetColumnIndices - Set the column indices for all the rows
  in the matrix.
  
  Input Parameters:
  +  mat     - the SeqSBAIJ matrix
  -  indices - the column indices
  
  Level: advanced
  
  Notes:
  This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.
  
  You MUST have set the correct numbers of nonzeros per row in the call to 
  MatCreateSeqSBAIJ(), and the columns indices MUST be sorted.
  
  MUST be called before any calls to MatSetValues()
  
  .seealso: MatCreateSeqSBAIJ
@*/ 
PetscErrorCode  MatSeqSBAIJSetColumnIndices(Mat mat,PetscInt *indices)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = PetscUseMethod(mat,"MatSeqSBAIJSetColumnIndices_C",(Mat,PetscInt *),(mat,indices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_SeqSBAIJ"
PetscErrorCode MatCopy_SeqSBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data; 
    Mat_SeqSBAIJ *b = (Mat_SeqSBAIJ*)B->data; 

    if (a->i[A->rmap->N] != b->i[B->rmap->N]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    ierr = PetscMemcpy(b->a,a->a,(a->i[A->rmap->N])*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr); 
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUp_SeqSBAIJ"
PetscErrorCode MatSetUp_SeqSBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqSBAIJSetPreallocation_SeqSBAIJ(A,A->rmap->bs,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqSBAIJ"
PetscErrorCode MatGetArray_SeqSBAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data; 
  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqSBAIJ"
PetscErrorCode MatRestoreArray_SeqSBAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
 }

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqSBAIJ"
PetscErrorCode MatAXPY_SeqSBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_SeqSBAIJ   *x=(Mat_SeqSBAIJ *)X->data, *y=(Mat_SeqSBAIJ *)Y->data;
  PetscErrorCode ierr;
  PetscInt       i,bs=Y->rmap->bs,bs2=bs*bs,j;
  PetscBLASInt   one = 1;
  
  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar alpha = a;
    PetscBLASInt bnz = PetscBLASIntCast(x->nz*bs2);
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    if (y->xtoy && y->XtoY != X) {
      ierr = PetscFree(y->xtoy);CHKERRQ(ierr);
      ierr = MatDestroy(&y->XtoY);CHKERRQ(ierr);
    }
    if (!y->xtoy) { /* get xtoy */
      ierr = MatAXPYGetxtoy_Private(x->mbs,x->i,x->j,PETSC_NULL, y->i,y->j,PETSC_NULL, &y->xtoy);CHKERRQ(ierr);
      y->XtoY = X;
    }
    for (i=0; i<x->nz; i++) {
      j = 0;
      while (j < bs2){
        y->a[bs2*y->xtoy[i]+j] += a*(x->a[bs2*i+j]); 
        j++; 
      }
    }
    ierr = PetscInfo3(Y,"ratio of nnz_s(X)/nnz_s(Y): %D/%D = %G\n",bs2*x->nz,bs2*y->nz,(PetscReal)(bs2*x->nz)/(bs2*y->nz));CHKERRQ(ierr);
  } else {
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr); 
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsSymmetric_SeqSBAIJ"
PetscErrorCode MatIsSymmetric_SeqSBAIJ(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsStructurallySymmetric_SeqSBAIJ"
PetscErrorCode MatIsStructurallySymmetric_SeqSBAIJ(Mat A,PetscBool  *flg)
{
   PetscFunctionBegin;
   *flg = PETSC_TRUE;
   PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsHermitian_SeqSBAIJ"
PetscErrorCode MatIsHermitian_SeqSBAIJ(Mat A,PetscReal tol,PetscBool  *flg)
 {
   PetscFunctionBegin;
   *flg = PETSC_FALSE;
   PetscFunctionReturn(0);
 }

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_SeqSBAIJ"
PetscErrorCode MatRealPart_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  PetscInt       i,nz = a->bs2*a->i[a->mbs];
  MatScalar      *aa = a->a;

  PetscFunctionBegin;  
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_SeqSBAIJ"
PetscErrorCode MatImaginaryPart_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data; 
  PetscInt       i,nz = a->bs2*a->i[a->mbs];
  MatScalar      *aa = a->a;

  PetscFunctionBegin;  
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRowsColumns_SeqSBAIJ"
PetscErrorCode MatZeroRowsColumns_SeqSBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqSBAIJ      *baij=(Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,count;
  PetscInt          bs=A->rmap->bs,bs2=baij->bs2,row,col;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;
  PetscBool         *zeroed,vecs = PETSC_FALSE;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    vecs = PETSC_TRUE;
  }
  A->same_nonzero = PETSC_TRUE;

  /* zero the columns */
  ierr = PetscMalloc(A->rmap->n*sizeof(PetscBool),&zeroed);CHKERRQ(ierr);
  ierr = PetscMemzero(zeroed,A->rmap->n*sizeof(PetscBool));CHKERRQ(ierr);
  for (i=0; i<is_n; i++) {
    if (is_idx[i] < 0 || is_idx[i] >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range",is_idx[i]);
    zeroed[is_idx[i]] = PETSC_TRUE;
  }
  if (vecs) {
    for (i=0; i<A->rmap->N; i++) {
      row = i/bs;
      for (j=baij->i[row]; j<baij->i[row+1]; j++) {
	for (k=0; k<bs; k++) {
	  col = bs*baij->j[j] + k;
          if (col <= i) continue;
	  aa = ((MatScalar*)(baij->a)) + j*bs2 + (i%bs) + bs*k;
	  if (!zeroed[i] && zeroed[col]) {
	    bb[i] -= aa[0]*xx[col];
	  }
	  if (zeroed[i] && !zeroed[col]) {
	    bb[col] -= aa[0]*xx[i];
	  }
	}
      }
    }
    for (i=0; i<is_n; i++) {
      bb[is_idx[i]] = diag*xx[is_idx[i]];
    }
  }

  for (i=0; i<A->rmap->N; i++) {
    if (!zeroed[i]) {
      row = i/bs;
      for (j=baij->i[row]; j<baij->i[row+1]; j++) {
        for (k=0; k<bs; k++) {
          col = bs*baij->j[j] + k;
	  if (zeroed[col]) {
	    aa = ((MatScalar*)(baij->a)) + j*bs2 + (i%bs) + bs*k;
            aa[0] = 0.0;
          }
        }
      }
    }
  }
  ierr = PetscFree(zeroed);CHKERRQ(ierr);
  if (vecs) {
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* zero the rows */
  for (i=0; i<is_n; i++) {
    row   = is_idx[i];
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    for (k=0; k<count; k++) { 
      aa[0] =  zero; 
      aa    += bs;
    }
    if (diag != 0.0) {
      ierr = (*A->ops->setvalues)(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd_SeqSBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqSBAIJ,
       MatGetRow_SeqSBAIJ,
       MatRestoreRow_SeqSBAIJ,
       MatMult_SeqSBAIJ_N,
/* 4*/ MatMultAdd_SeqSBAIJ_N,
       MatMult_SeqSBAIJ_N,       /* transpose versions are same as non-transpose versions */
       MatMultAdd_SeqSBAIJ_N,
       0,
       0,
       0,
/*10*/ 0,
       0,
       MatCholeskyFactor_SeqSBAIJ,
       MatSOR_SeqSBAIJ,
       MatTranspose_SeqSBAIJ,
/*15*/ MatGetInfo_SeqSBAIJ,
       MatEqual_SeqSBAIJ,
       MatGetDiagonal_SeqSBAIJ,
       MatDiagonalScale_SeqSBAIJ,
       MatNorm_SeqSBAIJ,
/*20*/ 0,
       MatAssemblyEnd_SeqSBAIJ,
       MatSetOption_SeqSBAIJ,
       MatZeroEntries_SeqSBAIJ,
/*24*/ 0,
       0,
       0,
       0,
       0,
/*29*/ MatSetUp_SeqSBAIJ,
       0,
       0,
       MatGetArray_SeqSBAIJ,
       MatRestoreArray_SeqSBAIJ,
/*34*/ MatDuplicate_SeqSBAIJ,
       0,
       0,
       0,
       MatICCFactor_SeqSBAIJ,
/*39*/ MatAXPY_SeqSBAIJ,
       MatGetSubMatrices_SeqSBAIJ,
       MatIncreaseOverlap_SeqSBAIJ,
       MatGetValues_SeqSBAIJ,
       MatCopy_SeqSBAIJ,
/*44*/ 0,
       MatScale_SeqSBAIJ,
       0,
       0,
       MatZeroRowsColumns_SeqSBAIJ,
/*49*/ 0,
       MatGetRowIJ_SeqSBAIJ,
       MatRestoreRowIJ_SeqSBAIJ,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       MatSetValuesBlocked_SeqSBAIJ,
/*59*/ MatGetSubMatrix_SeqSBAIJ,
       0,
       0,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,    
/*69*/ MatGetRowMaxAbs_SeqSBAIJ,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       MatGetInertia_SeqSBAIJ,
       MatLoad_SeqSBAIJ,
/*84*/ MatIsSymmetric_SeqSBAIJ,
       MatIsHermitian_SeqSBAIJ,
       MatIsStructurallySymmetric_SeqSBAIJ,
       0,
       0,
/*89*/ 0,
       0,
       0,
       0,
       0,
/*94*/ 0,
       0,
       0,
       0,
       0,
/*99*/ 0,
       0,
       0,
       0,
       0,
/*104*/0,
       MatRealPart_SeqSBAIJ,
       MatImaginaryPart_SeqSBAIJ,
       MatGetRowUpperTriangular_SeqSBAIJ,
       MatRestoreRowUpperTriangular_SeqSBAIJ,
/*109*/0,
       0,
       0,
       0,
       MatMissingDiagonal_SeqSBAIJ,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0,
       0,
       0
};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_SeqSBAIJ"
PetscErrorCode  MatStoreValues_SeqSBAIJ(Mat mat)
{
  Mat_SeqSBAIJ   *aij = (Mat_SeqSBAIJ *)mat->data;
  PetscInt       nz = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  
  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&aij->saved_values);CHKERRQ(ierr);
  }
  
  /* copy values over */
  ierr = PetscMemcpy(aij->saved_values,aij->a,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_SeqSBAIJ"
PetscErrorCode  MatRetrieveValues_SeqSBAIJ(Mat mat)
{
  Mat_SeqSBAIJ   *aij = (Mat_SeqSBAIJ *)mat->data;
  PetscErrorCode ierr;
  PetscInt       nz = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;
  
  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  if (!aij->saved_values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
  
  /* copy values over */
  ierr = PetscMemcpy(aij->a,aij->saved_values,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatSeqSBAIJSetPreallocation_SeqSBAIJ"
PetscErrorCode  MatSeqSBAIJSetPreallocation_SeqSBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqSBAIJ   *b = (Mat_SeqSBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs,bs2;
  PetscBool      skipallocation = PETSC_FALSE,flg = PETSC_FALSE,realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  mbs  = B->rmap->N/bs;
  bs2  = bs*bs;
  
  if (mbs*bs != B->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows, cols must be divisible by blocksize");
  
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 3;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %D",nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %D value %D",i,nnz[i]);
      if (nnz[i] > mbs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %D value %D rowlength %D",i,nnz[i],mbs);
    }
  }
  
  B->ops->mult             = MatMult_SeqSBAIJ_N; 
  B->ops->multadd          = MatMultAdd_SeqSBAIJ_N;
  B->ops->multtranspose    = MatMult_SeqSBAIJ_N;
  B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_N;
  ierr  = PetscOptionsGetBool(((PetscObject)B)->prefix,"-mat_no_unroll",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) {
    switch (bs) {
    case 1:
      B->ops->mult             = MatMult_SeqSBAIJ_1;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_1;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_1;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_1;
      break;
    case 2:
      B->ops->mult             = MatMult_SeqSBAIJ_2;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_2;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_2;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_2;
      break;
    case 3:
      B->ops->mult             = MatMult_SeqSBAIJ_3;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_3;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_3;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_3;
      break;
    case 4:
      B->ops->mult             = MatMult_SeqSBAIJ_4;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_4;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_4;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_4;
      break;
    case 5:
      B->ops->mult             = MatMult_SeqSBAIJ_5;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_5;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_5;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_5;
      break;
    case 6:
      B->ops->mult             = MatMult_SeqSBAIJ_6;
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_6;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_6;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_6;
      break;
    case 7:
      B->ops->mult             = MatMult_SeqSBAIJ_7; 
      B->ops->multadd          = MatMultAdd_SeqSBAIJ_7;
      B->ops->multtranspose    = MatMult_SeqSBAIJ_7;
      B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_7;
      break;
    }
  }
  
  b->mbs = mbs;
  b->nbs = mbs; 
  if (!skipallocation) {
    if (!b->imax) {
      ierr = PetscMalloc2(mbs,PetscInt,&b->imax,mbs,PetscInt,&b->ilen);CHKERRQ(ierr);
      b->free_imax_ilen = PETSC_TRUE;
      ierr = PetscLogObjectMemory(B,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz <= 0)        nz = 1;
      for (i=0; i<mbs; i++) {
        b->imax[i] = nz; 
      }
      nz = nz*mbs; /* total nz */
    } else {
      nz = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) { b->ilen[i] = 0;}
    /* nz=(nz+mbs)/2; */ /* total diagonal and superdiagonal nonzero blocks */
  
    /* allocate the matrix space */
    ierr = MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(bs2*nz,PetscScalar,&b->a,nz,PetscInt,&b->j,B->rmap->N+1,PetscInt,&b->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
    ierr = PetscMemzero(b->a,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(b->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
    b->singlemalloc = PETSC_TRUE;
  
    /* pointer to beginning of each row */
    b->i[0] = 0;
    for (i=1; i<mbs+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }
    b->free_a     = PETSC_TRUE;
    b->free_ij    = PETSC_TRUE;
  } else {
    b->free_a     = PETSC_FALSE;
    b->free_ij    = PETSC_FALSE;
  }
  
  B->rmap->bs     = bs;
  b->bs2          = bs2;
  b->nz           = 0;
  b->maxnz        = nz;
  
  b->inew             = 0;
  b->jnew             = 0;
  b->anew             = 0;
  b->a2anew           = 0;
  b->permute          = PETSC_FALSE;
  if (realalloc) {ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
   This is used to set the numeric factorization for both Cholesky and ICC symbolic factorization
*/
#undef __FUNCT__  
#define __FUNCT__ "MatSeqSBAIJSetNumericFactorization_inplace"
PetscErrorCode MatSeqSBAIJSetNumericFactorization_inplace(Mat B,PetscBool  natural)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       bs = B->rmap->bs;

  PetscFunctionBegin;
  ierr    = PetscOptionsGetBool(((PetscObject)B)->prefix,"-mat_no_unroll",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) bs = 8;

  if (!natural) {
    switch (bs) {
    case 1:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_inplace;
      break;
    case 2:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_2;  
      break;
    case 3:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_3;  
      break;
    case 4:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_4;  
      break;
    case 5:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_5;  
      break;
    case 6:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_6;  
      break;
    case 7:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_7;
      break;
    default:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_N;
      break;
    }
  } else {
    switch (bs) {
    case 1:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace;
      break;
    case 2:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering;  
      break;
    case 3:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering;  
      break;
    case 4:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering;  
      break;
    case 5:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering;  
      break;
    case 6:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering;  
      break;
    case 7:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering;
      break;
    default:
      B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering;
      break;
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode  MatConvert_SeqSBAIJ_SeqAIJ(Mat, MatType,MatReuse,Mat*); 
extern PetscErrorCode  MatConvert_SeqSBAIJ_SeqBAIJ(Mat, MatType,MatReuse,Mat*); 
EXTERN_C_END

  
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_petsc"
PetscErrorCode MatGetFactor_seqsbaij_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt           n = A->rmap->n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (A->hermitian)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian Factor is not supported");
#endif
  ierr = MatCreate(((PetscObject)A)->comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    ierr = MatSetType(*B,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*B,A->rmap->bs,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ;
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqSBAIJ;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  (*B)->factortype = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactorAvailable_seqsbaij_petsc"
PetscErrorCode MatGetFactorAvailable_seqsbaij_petsc(Mat A,MatFactorType ftype,PetscBool  *flg)
{
  PetscFunctionBegin;
  if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    *flg = PETSC_TRUE;
  } else {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_MUMPS)
extern PetscErrorCode MatGetFactor_sbaij_mumps(Mat,MatFactorType,Mat*);
#endif
#if defined(PETSC_HAVE_SPOOLES)
extern PetscErrorCode MatGetFactor_seqsbaij_spooles(Mat,MatFactorType,Mat*);
#endif
#if defined(PETSC_HAVE_PASTIX)
extern PetscErrorCode MatGetFactor_seqsbaij_pastix(Mat,MatFactorType,Mat*);
#endif
#if defined(PETSC_HAVE_CHOLMOD)
extern PetscErrorCode MatGetFactor_seqsbaij_cholmod(Mat,MatFactorType,Mat*);
#endif
extern PetscErrorCode MatGetFactor_seqsbaij_sbstrm(Mat,MatFactorType,Mat*);
EXTERN_C_END 

/*MC
  MATSEQSBAIJ - MATSEQSBAIJ = "seqsbaij" - A matrix type to be used for sequential symmetric block sparse matrices, 
  based on block compressed sparse row format.  Only the upper triangular portion of the matrix is stored.

  For complex numbers by default this matrix is symmetric, NOT Hermitian symmetric. To make it Hermitian symmetric you
  can call MatSetOption(Mat, MAT_HERMITIAN); after MatAssemblyEnd()

  Options Database Keys:
  . -mat_type seqsbaij - sets the matrix type to "seqsbaij" during a call to MatSetFromOptions()
  
  Notes: By default if you insert values into the lower triangular part of the matrix they are simply ignored (since they are not 
     stored and it is assumed they symmetric to the upper triangular). If you call MatSetOption(Mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_FALSE) or use
     the options database -mat_ignore_lower_triangular false it will generate an error if you try to set a value in the lower triangular portion.


  Level: beginner
  
  .seealso: MatCreateSeqSBAIJ
M*/

EXTERN_C_BEGIN
extern PetscErrorCode  MatConvert_SeqSBAIJ_SeqSBSTRM(Mat, MatType,MatReuse,Mat*);
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqSBAIJ"
PetscErrorCode  MatCreate_SeqSBAIJ(Mat B)
{
  Mat_SeqSBAIJ   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      no_unroll = PETSC_FALSE,no_inode = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");
  
  ierr    = PetscNewLog(B,Mat_SeqSBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy     = MatDestroy_SeqSBAIJ;
  B->ops->view        = MatView_SeqSBAIJ;
  b->row              = 0;
  b->icol             = 0;
  b->reallocs         = 0;
  b->saved_values     = 0;
  b->inode.limit      = 5;  
  b->inode.max_limit  = 5;  
  
  b->roworiented      = PETSC_TRUE;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->mult_work        = 0;
  B->spptr            = 0;
  B->info.nz_unneeded = (PetscReal)b->maxnz*b->bs2;
  b->keepnonzeropattern   = PETSC_FALSE;
  b->xtoy             = 0;
  b->XtoY             = 0;
  
  b->inew             = 0;
  b->jnew             = 0;
  b->anew             = 0;
  b->a2anew           = 0;
  b->permute          = PETSC_FALSE;

  b->ignore_ltriangular = PETSC_TRUE;
  ierr = PetscOptionsGetBool(((PetscObject)B)->prefix,"-mat_ignore_lower_triangular",&b->ignore_ltriangular,PETSC_NULL);CHKERRQ(ierr);

  b->getrow_utriangular = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)B)->prefix,"-mat_getrow_uppertriangular",&b->getrow_utriangular,PETSC_NULL);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PASTIX)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_pastix_C",
					   "MatGetFactor_seqsbaij_pastix",
					   MatGetFactor_seqsbaij_pastix);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_spooles_C",
                                     "MatGetFactor_seqsbaij_spooles",
                                     MatGetFactor_seqsbaij_spooles);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_mumps_C",
                                     "MatGetFactor_sbaij_mumps",
                                     MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CHOLMOD)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_cholmod_C",
                                     "MatGetFactor_seqsbaij_cholmod",
                                     MatGetFactor_seqsbaij_cholmod);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactorAvailable_petsc_C",
                                     "MatGetFactorAvailable_seqsbaij_petsc",
                                     MatGetFactorAvailable_seqsbaij_petsc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_petsc_C",
                                     "MatGetFactor_seqsbaij_petsc",
                                     MatGetFactor_seqsbaij_petsc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_sbstrm_C",
                                     "MatGetFactor_seqsbaij_sbstrm",
                                     MatGetFactor_seqsbaij_sbstrm);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqSBAIJ",
                                     MatStoreValues_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqSBAIJ",
                                     (void*)MatRetrieveValues_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqSBAIJSetColumnIndices_C",
                                     "MatSeqSBAIJSetColumnIndices_SeqSBAIJ",
                                     MatSeqSBAIJSetColumnIndices_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaij_seqaij_C",
                                     "MatConvert_SeqSBAIJ_SeqAIJ",
                                      MatConvert_SeqSBAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaij_seqbaij_C",
                                     "MatConvert_SeqSBAIJ_SeqBAIJ",
                                      MatConvert_SeqSBAIJ_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqSBAIJSetPreallocation_C",
                                     "MatSeqSBAIJSetPreallocation_SeqSBAIJ",
                                     MatSeqSBAIJSetPreallocation_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqsbaij_seqsbstrm_C",
                                     "MatConvert_SeqSBAIJ_SeqSBSTRM",
                                      MatConvert_SeqSBAIJ_SeqSBSTRM);CHKERRQ(ierr);

  B->symmetric                  = PETSC_TRUE;
  B->structurally_symmetric     = PETSC_TRUE;
  B->symmetric_set              = PETSC_TRUE;
  B->structurally_symmetric_set = PETSC_TRUE;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSBAIJ);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(((PetscObject)B)->comm,((PetscObject)B)->prefix,"Options for SEQSBAIJ matrix","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_no_unroll","Do not optimize for inodes (slower)",PETSC_NULL,no_unroll,&no_unroll,PETSC_NULL);CHKERRQ(ierr);
    if (no_unroll) {ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_unroll\n");CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-mat_no_inode","Do not optimize for inodes (slower)",PETSC_NULL,no_inode,&no_inode,PETSC_NULL);CHKERRQ(ierr);
    if (no_inode) {ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_inode\n");CHKERRQ(ierr);}
    ierr = PetscOptionsInt("-mat_inode_limit","Do not use inodes larger then this value",PETSC_NULL,b->inode.limit,&b->inode.limit,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  b->inode.use = (PetscBool)(!(no_unroll || no_inode));
  if (b->inode.limit > b->inode.max_limit) b->inode.limit = b->inode.max_limit;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSeqSBAIJSetPreallocation"
/*@C
   MatSeqSBAIJSetPreallocation - Creates a sparse symmetric matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on Mat

   Input Parameters:
+  A - the symmetric matrix 
.  bs - size of block
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the upper triangular plus
         diagonal portion of each block (possibly different for each block row) or PETSC_NULL

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use (only works if a negative bs is passed in

   Level: intermediate

   Notes:
   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  See the <a href="../../docs/manual.pdf#nameddest=ch_mat">Mat chapter of the users manual</a> for details.

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string 
   malloc in them to see if additional memory allocation was needed.

   If the nnz parameter is given then the nz parameter is ignored


.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateSBAIJ()
@*/
PetscErrorCode  MatSeqSBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[]) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatSeqSBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqSBAIJ"
/*@C
   MatCreateSeqSBAIJ - Creates a sparse symmetric matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block
.  m - number of rows, or number of columns
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the upper triangular plus
         diagonal portion of each block (possibly different for each block row) or PETSC_NULL

   Output Parameter:
.  A - the symmetric matrix 

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Level: intermediate

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly. 
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   The number of rows and columns must be divisible by blocksize.
   This matrix type does not support complex Hermitian operation.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  See the <a href="../../docs/manual.pdf#nameddest=ch_mat">Mat chapter of the users manual</a> for details.

   If the nnz parameter is given then the nz parameter is ignored

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateSBAIJ()
@*/
PetscErrorCode  MatCreateSeqSBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqSBAIJ"
PetscErrorCode MatDuplicate_SeqSBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat            C;
  Mat_SeqSBAIJ   *c,*a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 =a->bs2;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt matrix");

  *B = 0;
  ierr = MatCreate(((PetscObject)A)->comm,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,A->rmap->N,A->cmap->n,A->rmap->N,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  c    = (Mat_SeqSBAIJ*)C->data;

  C->preallocated       = PETSC_TRUE;
  C->factortype         = A->factortype;
  c->row                = 0;
  c->icol               = 0;
  c->saved_values       = 0;
  c->keepnonzeropattern = a->keepnonzeropattern;
  C->assembled          = PETSC_TRUE;

  ierr = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);  
  ierr = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);  
  c->bs2  = a->bs2;
  c->mbs  = a->mbs;
  c->nbs  = a->nbs;

  if  (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax           = a->imax;
    c->ilen           = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    ierr = PetscMalloc2((mbs+1),PetscInt,&c->imax,(mbs+1),PetscInt,&c->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,2*(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i]; 
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    ierr = PetscMalloc(bs2*nz*sizeof(MatScalar),&c->a);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
    c->i            = a->i;
    c->j            = a->j;
    c->singlemalloc = PETSC_FALSE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_FALSE;
    c->parent       = A;
    ierr            = PetscObjectReference((PetscObject)A);CHKERRQ(ierr); 
    ierr            = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr            = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc3(bs2*nz,MatScalar,&c->a,nz,PetscInt,&c->j,mbs+1,PetscInt,&c->i);CHKERRQ(ierr);
    ierr = PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,(mbs+1)*sizeof(PetscInt) + nz*(bs2*sizeof(MatScalar) + sizeof(PetscInt)));CHKERRQ(ierr);
    c->singlemalloc = PETSC_TRUE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
  }
  if (mbs > 0) {
    if (cpvalues != MAT_SHARE_NONZERO_PATTERN) {
      ierr = PetscMemcpy(c->j,a->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscMemcpy(c->a,a->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(c->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
    }
    if (a->jshort) {
      /* cannot share jshort, it is reallocated in MatAssemblyEnd_SeqSBAIJ() */
      /* if the parent matrix is reassembled, this child matrix will never notice */
      ierr = PetscMalloc(nz*sizeof(unsigned short),&c->jshort);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(C,nz*sizeof(unsigned short));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->jshort,a->jshort,nz*sizeof(unsigned short));CHKERRQ(ierr);
      c->free_jshort = PETSC_TRUE;
    }
  }

  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  if (a->diag) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      c->diag      = a->diag;
      c->free_diag = PETSC_FALSE;
    } else {
      ierr = PetscMalloc(mbs*sizeof(PetscInt),&c->diag);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(C,mbs*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0; i<mbs; i++) {
	c->diag[i] = a->diag[i];
      }
      c->free_diag = PETSC_TRUE;
    }
  }
  c->nz           = a->nz;
  c->maxnz        = a->nz; /* Since we allocate exactly the right amount */
  c->solve_work   = 0;
  c->mult_work    = 0;
  *B = C;
  ierr = PetscFListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqSBAIJ"
PetscErrorCode MatLoad_SeqSBAIJ(Mat newmat,PetscViewer viewer)
{
  Mat_SeqSBAIJ   *a;
  PetscErrorCode ierr;
  int            fd;
  PetscMPIInt    size;
  PetscInt       i,nz,header[4],*rowlengths=0,M,N,bs=1;
  PetscInt       *mask,mbs,*jj,j,rowcount,nzcount,k,*s_browlengths,maskcount;
  PetscInt       kmax,jcount,block,idx,point,nzcountb,extra_rows,rows,cols;
  PetscInt       *masked,nmask,tmp,bs2,ishift;
  PetscScalar    *aa;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(((PetscObject)newmat)->prefix,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  bs2  = bs*bs;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (header[3] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as SeqSBAIJ");

  if (M != N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  mbs        = M/bs;
  extra_rows = bs - M + bs*(mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  mbs++;
  if (extra_rows) {
    ierr = PetscInfo(viewer,"Padding loaded matrix to match blocksize\n");CHKERRQ(ierr);
  }

  /* Set global sizes if not already set */
  if (newmat->rmap->n < 0 && newmat->rmap->N < 0 && newmat->cmap->n < 0 && newmat->cmap->N < 0) {
    ierr = MatSetSizes(newmat,PETSC_DECIDE,PETSC_DECIDE,M+extra_rows,N+extra_rows);CHKERRQ(ierr);
  } else { /* Check if the matrix global sizes are correct */
    ierr = MatGetSize(newmat,&rows,&cols);CHKERRQ(ierr);
    if (M != rows ||  N != cols) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix in file of different length (%d, %d) than the input matrix (%d, %d)",M,N,rows,cols);
  }

  /* read in row lengths */
  ierr = PetscMalloc((M+extra_rows)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;

  /* read in column indices */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(PetscInt),&jj);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,jj,nz,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) jj[nz+i] = M+i;

  /* loop over row lengths determining block row lengths */
  ierr     = PetscMalloc(mbs*sizeof(PetscInt),&s_browlengths);CHKERRQ(ierr);
  ierr     = PetscMemzero(s_browlengths,mbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr     = PetscMalloc2(mbs,PetscInt,&mask,mbs,PetscInt,&masked);CHKERRQ(ierr);
  ierr     = PetscMemzero(mask,mbs*sizeof(PetscInt));CHKERRQ(ierr);
  rowcount = 0;
  nzcount  = 0;
  for (i=0; i<mbs; i++) {
    nmask = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[rowcount];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs;   /* block col. index */
        if (!mask[tmp] && tmp >= i) {masked[nmask++] = tmp; mask[tmp] = 1;} 
      }
      rowcount++;
    }
    s_browlengths[i] += nmask;
    
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }

  /* Do preallocation */
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(newmat,bs,0,s_browlengths);CHKERRQ(ierr);
  a = (Mat_SeqSBAIJ*)newmat->data;

  /* set matrix "i" values */
  a->i[0] = 0;
  for (i=1; i<= mbs; i++) {
    a->i[i]      = a->i[i-1] + s_browlengths[i-1];
    a->ilen[i-1] = s_browlengths[i-1];
  }
  a->nz = a->i[mbs];

  /* read in nonzero values */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(PetscScalar),&aa);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,aa,nz,PETSC_SCALAR);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) aa[nz+i] = 1.0;

  /* set "a" and "j" values into matrix */
  nzcount = 0; jcount = 0;
  for (i=0; i<mbs; i++) {
    nzcountb = nzcount;
    nmask    = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs; /* block col. index */
        if (!mask[tmp] && tmp >= i) { masked[nmask++] = tmp; mask[tmp] = 1;} 
      }
    }
    /* sort the masked values */
    ierr = PetscSortInt(nmask,masked);CHKERRQ(ierr);

    /* set "j" values into matrix */
    maskcount = 1;
    for (j=0; j<nmask; j++) {
      a->j[jcount++]  = masked[j];
      mask[masked[j]] = maskcount++; 
    }

    /* set "a" values into matrix */
    ishift = bs2*a->i[i]; 
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp       = jj[nzcountb]/bs ; /* block col. index */
        if (tmp >= i){ 
          block     = mask[tmp] - 1;
          point     = jj[nzcountb] - bs*tmp;
          idx       = ishift + bs2*block + j + bs*point;
          a->a[idx] = aa[nzcountb];
        } 
        nzcountb++;
      }
    }
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }
  if (jcount != a->nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Bad binary matrix");

  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = PetscFree(s_browlengths);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);
  ierr = PetscFree2(mask,masked);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView_Private(newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqSBAIJWithArrays"
/*@
     MatCreateSeqSBAIJWithArrays - Creates an sequential SBAIJ matrix using matrix elements 
              (upper triangular entries in CSR format) provided by the user.

     Collective on MPI_Comm

   Input Parameters:
+  comm - must be an MPI communicator of size 1
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  i - row indices
.  j - column indices
-  a - matrix values

   Output Parameter:
.  mat - the matrix

   Level: advanced

   Notes:
       The i, j, and a arrays are not copied by this routine, the user must free these arrays
    once the matrix is destroyed

       You cannot set new nonzero locations into this matrix, that will generate an error.

       The i and j indices are 0 based

       When block size is greater than 1 the matrix values must be stored using the SBAIJ storage format (see the SBAIJ code to determine this). For block size of 1
       it is the regular CSR format excluding the lower triangular elements.

.seealso: MatCreate(), MatCreateSBAIJ(), MatCreateSeqSBAIJ()

@*/
PetscErrorCode  MatCreateSeqSBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt* i,PetscInt*j,PetscScalar *a,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqSBAIJ   *sbaij;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"block size %D > 1 is not supported yet",bs);
  if (i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(*mat,bs,MAT_SKIP_ALLOCATION,0);CHKERRQ(ierr);
  sbaij = (Mat_SeqSBAIJ*)(*mat)->data;
  ierr = PetscMalloc2(m,PetscInt,&sbaij->imax,m,PetscInt,&sbaij->ilen);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(*mat,2*m*sizeof(PetscInt));CHKERRQ(ierr);

  sbaij->i = i;
  sbaij->j = j;
  sbaij->a = a;
  sbaij->singlemalloc = PETSC_FALSE;
  sbaij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  sbaij->free_a       = PETSC_FALSE; 
  sbaij->free_ij      = PETSC_FALSE; 

  for (ii=0; ii<m; ii++) {
    sbaij->ilen[ii] = sbaij->imax[ii] = i[ii+1] - i[ii];
#if defined(PETSC_USE_DEBUG)
    if (i[ii+1] - i[ii] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
#endif    
  }
#if defined(PETSC_USE_DEBUG)
  for (ii=0; ii<sbaij->i[m]; ii++) {
    if (j[ii] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %d index = %d",ii,j[ii]);
    if (j[ii] > n - 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %d index = %d",ii,j[ii]);
  }
#endif    

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





