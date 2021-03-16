
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

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_Elemental(Mat,MatType,MatReuse,Mat*);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_SBAIJ_ScaLAPACK(Mat,MatType,MatReuse,Mat*);
#endif
PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Basic(Mat,MatType,MatReuse,Mat*);

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqSBAIJ(Mat A,PetscBool  *missing,PetscInt *dd)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *diag,*ii = a->i,i;

  PetscFunctionBegin;
  ierr     = MatMarkDiagonal_SeqSBAIJ(A);CHKERRQ(ierr);
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !ii) {
    *missing = PETSC_TRUE;
    if (dd) *dd = 0;
    ierr = PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n");CHKERRQ(ierr);
  } else {
    diag = a->diag;
    for (i=0; i<a->mbs; i++) {
      if (diag[i] >= ii[i+1]) {
        *missing = PETSC_TRUE;
        if (dd) *dd = i;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr         = PetscMalloc1(a->mbs,&a->diag);CHKERRQ(ierr);
    ierr         = PetscLogObjectMemory((PetscObject)A,a->mbs*sizeof(PetscInt));CHKERRQ(ierr);
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<a->mbs; i++) {
    a->diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowIJ_SeqSBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *nn,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,nz = a->i[n],*tia,*tja,bs = A->rmap->bs,k,l,cnt;
  PetscInt       **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,PETSC_FALSE,0,0,&tia,&tja);CHKERRQ(ierr);
    nz   = tia[n];
  } else {
    tia = a->i; tja = a->j;
  }

  if (!blockcompressed && bs > 1) {
    (*nn) *= bs;
    /* malloc & create the natural set of indices */
    ierr = PetscMalloc1((n+1)*bs,ia);CHKERRQ(ierr);
    if (n) {
      (*ia)[0] = oshift;
      for (j=1; j<bs; j++) {
        (*ia)[j] = (tia[1]-tia[0])*bs+(*ia)[j-1];
      }
    }

    for (i=1; i<n; i++) {
      (*ia)[i*bs] = (tia[i]-tia[i-1])*bs + (*ia)[i*bs-1];
      for (j=1; j<bs; j++) {
        (*ia)[i*bs+j] = (tia[i+1]-tia[i])*bs + (*ia)[i*bs+j-1];
      }
    }
    if (n) {
      (*ia)[n*bs] = (tia[n]-tia[n-1])*bs + (*ia)[n*bs-1];
    }

    if (inja) {
      ierr = PetscMalloc1(nz*bs*bs,ja);CHKERRQ(ierr);
      cnt = 0;
      for (i=0; i<n; i++) {
        for (j=0; j<bs; j++) {
          for (k=tia[i]; k<tia[i+1]; k++) {
            for (l=0; l<bs; l++) {
              (*ja)[cnt++] = bs*tja[k] + l;
            }
          }
        }
      }
    }

    if (symmetric) { /* deallocate memory allocated in MatToSymmetricIJ_SeqAIJ() */
      ierr = PetscFree(tia);CHKERRQ(ierr);
      ierr = PetscFree(tja);CHKERRQ(ierr);
    }
  } else if (oshift == 1) {
    if (symmetric) {
      nz = tia[A->rmap->n/bs];
      /*  add 1 to i and j indices */
      for (i=0; i<A->rmap->n/bs+1; i++) tia[i] = tia[i] + 1;
      *ia = tia;
      if (ja) {
        for (i=0; i<nz; i++) tja[i] = tja[i] + 1;
        *ja = tja;
      }
    } else {
      nz = a->i[A->rmap->n/bs];
      /* malloc space and  add 1 to i and j indices */
      ierr = PetscMalloc1(A->rmap->n/bs+1,ia);CHKERRQ(ierr);
      for (i=0; i<A->rmap->n/bs+1; i++) (*ia)[i] = a->i[i] + 1;
      if (ja) {
        ierr = PetscMalloc1(nz,ja);CHKERRQ(ierr);
        for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
      }
    }
  } else {
    *ia = tia;
    if (ja) *ja = tja;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRowIJ_SeqSBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if ((!blockcompressed && A->rmap->bs > 1) || (symmetric || oshift == 1)) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    if (ja) {ierr = PetscFree(*ja);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, NZ=%D",A->rmap->N,a->nz);
#endif
  ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
  if (a->free_diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
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
  if (a->free_jshort) {ierr = PetscFree(a->jshort);CHKERRQ(ierr);}
  ierr = PetscFree(a->inew);CHKERRQ(ierr);
  ierr = MatDestroy(&a->parent);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqSBAIJSetColumnIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_seqaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_seqbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqSBAIJSetPreallocationCSR_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_elemental_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqsbaij_scalapack_C",NULL);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqSBAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       bs;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
#endif
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
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
  case MAT_SORTED_FULL:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_HERMITIAN:
#if defined(PETSC_USE_COMPLEX)
    if (flg) { /* disable transpose ops */
      if (bs > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for Hermitian with block size greater than 1");
      A->ops->multtranspose    = NULL;
      A->ops->multtransposeadd = NULL;
      A->symmetric             = PETSC_FALSE;
    }
#endif
    break;
  case MAT_SYMMETRIC:
  case MAT_SPD:
#if defined(PETSC_USE_COMPLEX)
    if (flg) { /* An hermitian and symmetric matrix has zero imaginary part (restore back transpose ops) */
      A->ops->multtranspose    = A->ops->mult;
      A->ops->multtransposeadd = A->ops->multadd;
    }
#endif
    break;
    /* These options are handled directly by MatSetOption() */
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_STRUCTURE_ONLY:
    /* These options are handled directly by MatSetOption() */
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
  case MAT_SUBMAT_SINGLEIS:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_SeqSBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A && !a->getrow_utriangular) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatGetRow is not supported for SBAIJ matrix format. Getting the upper triangular part of row, run with -mat_getrow_uppertriangular, call MatSetOption(mat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE) or MatGetRowUpperTriangular()");

  /* Get the upper triangular part of the row */
  ierr = MatGetRow_SeqBAIJ_private(A,row,nz,idx,v,a->i,a->j,a->a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqSBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowUpperTriangular_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  a->getrow_utriangular = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRowUpperTriangular_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  a->getrow_utriangular = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SeqSBAIJ(Mat A,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,B);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqSBAIJ_ASCII(Mat A,PetscViewer viewer)
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
    Mat        aij;
    const char *matname;

    if (A->factortype && bs>1) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Warning: matrix is factored with bs>1. MatView() with PETSC_VIEWER_ASCII_MATLAB is not supported and ignored!\n");CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)A,&matname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)aij,matname);CHKERRQ(ierr);
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
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
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
    if (A->factortype) { /* for factored matrix */
      if (bs>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"matrix is factored with bs>1. Not implemented yet");

      diag=a->diag;
      for (i=0; i<a->mbs; i++) { /* for row block i */
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        /* diagonal entry */
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[diag[i]]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",a->j[diag[i]],(double)PetscRealPart(1.0/a->a[diag[i]]),(double)PetscImaginaryPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[diag[i]]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i) ",a->j[diag[i]],(double)PetscRealPart(1.0/a->a[diag[i]]),-(double)PetscImaginaryPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[diag[i]],(double)PetscRealPart(1.0/a->a[diag[i]]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[diag[i]],(double)(1.0/a->a[diag[i]]));CHKERRQ(ierr);
#endif
        /* off-diagonal entries */
        for (k=a->i[i]; k<a->i[i+1]-1; k++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[k]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",bs*a->j[k],(double)PetscRealPart(a->a[k]),(double)PetscImaginaryPart(a->a[k]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->a[k]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i) ",bs*a->j[k],(double)PetscRealPart(a->a[k]),-(double)PetscImaginaryPart(a->a[k]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",bs*a->j[k],(double)PetscRealPart(a->a[k]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[k],(double)a->a[k]);CHKERRQ(ierr);
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
                ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",bs*a->j[k]+l,
                                              (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i) ",bs*a->j[k]+l,
                                              (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
              }
#else
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
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

#include <petscdraw.h>
static PetscErrorCode MatView_SeqSBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat            A = (Mat) Aa;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,mbs=a->mbs,color,bs=A->rmap->bs,bs2=a->bs2;
  PetscReal      xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar      *aa;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  ierr = PetscDrawString(draw, .3*(xl+xr), .3*(yl+yr), PETSC_DRAW_BLACK, "symmetric");CHKERRQ(ierr);
  /* Blue for negative, Cyan for zero and  Red for positive */
  color = PETSC_DRAW_BLUE;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa  = a->a + j*bs2;
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
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqSBAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      xl,yl,xr,yr,w,h;
  PetscDraw      draw;
  PetscBool      isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->rmap->N; yr = A->rmap->N; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqSBAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
#define MatView_SeqSBAIJ_Binary MatView_SeqBAIJ_Binary

PetscErrorCode MatView_SeqSBAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_SeqSBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqSBAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqSBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    Mat        B;
    const char *matname;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)A,&matname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)B,matname);CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode MatGetValues_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  PetscInt     *ai = a->i,*ailen = a->ilen;
  PetscInt     brow,bcol,ridx,cidx,bs=A->rmap->bs,bs2=a->bs2;
  MatScalar    *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over rows */
    row = im[k]; brow = row/bs;
    if (row < 0) {v += n; continue;} /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",row); */
    if (row >= A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
    rp   = aj + ai[brow]; ap = aa + bs2*ai[brow];
    nrow = ailen[brow];
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) {v++; continue;} /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",in[l]); */
      if (in[l] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap->n-1);
      col  = in[l];
      bcol = col/bs;
      cidx = col%bs;
      ridx = row%bs;
      high = nrow;
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
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

PetscErrorCode MatPermute_SeqSBAIJ(Mat A,IS rowp,IS colp,Mat *B)
{
  Mat            C;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSEQBAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatPermute(C,rowp,colp,B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  if (rowp == colp) {
    ierr = MatConvert(*B,MATSEQSBAIJ,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValuesBlocked_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt          *imax      =a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs,stepval;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (roworiented) stepval = (n-1)*bs;
  else stepval = (m-1)*bs;

  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    if (PetscUnlikelyDebug(row >= a->mbs)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block index row too large %D max %D",row,a->mbs-1);
    rp   = aj + ai[row];
    ap   = aa + bs2*ai[row];
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      col = in[l];
      if (PetscUnlikelyDebug(col >= a->nbs)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block index column too large %D max %D",col,a->nbs-1);
      if (col < row) {
        if (a->ignore_ltriangular) continue; /* ignore lower triangular block */
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
      }
      if (roworiented) value = v + k*(stepval+bs)*bs + l*bs;
      else value = v + l*(stepval+bs)*bs + k*bs;

      if (col <= lastcol) low = 0;
      else high = nrow;

      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap = ap +  bs2*i;
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
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block index nonzero block (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      ierr  = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
      ierr  = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
      ierr  = PetscArrayzero(ap+bs2*i,bs2);CHKERRQ(ierr);
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
            *bap++ = *value++;
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
PetscErrorCode MatAssemblyEnd_SeqSBAIJ_SeqAIJ_Inode(Mat A)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  const PetscInt *ai = a->i, *aj = a->j,*cols;
  PetscInt       i   = 0,j,blk_size,m = A->rmap->n,node_count = 0,nzx,nzy,*ns,row,nz,cnt,cnt2,*counts;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscMalloc1(m,&ns);CHKERRQ(ierr);
  while (i < m) {
    nzx = ai[i+1] - ai[i];       /* Number of nonzeros */
    /* Limits the number of elements in a node to 'a->inode.limit' */
    for (j=i+1,blk_size=1; j<m && blk_size <a->inode.limit; ++j,++blk_size) {
      nzy = ai[j+1] - ai[j];
      if (nzy != (nzx - j + i)) break;
      ierr = PetscArraycmp(aj + ai[i] + j - i,aj + ai[j],nzy,&flag);CHKERRQ(ierr);
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

    ierr = PetscMalloc1(node_count,&a->inode.size);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,node_count*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(a->inode.size,ns,node_count);CHKERRQ(ierr);
    ierr = PetscFree(ns);CHKERRQ(ierr);
    ierr = PetscInfo3(A,"Found %D nodes of %D. Limit used: %D. Using Inode routines\n",node_count,m,a->inode.limit);CHKERRQ(ierr);

    /* count collections of adjacent columns in each inode */
    row = 0;
    cnt = 0;
    for (i=0; i<node_count; i++) {
      cols = aj + ai[row] + a->inode.size[i];
      nz   = ai[row+1] - ai[row] - a->inode.size[i];
      for (j=1; j<nz; j++) {
        if (cols[j] != cols[j-1]+1) cnt++;
      }
      cnt++;
      row += a->inode.size[i];
    }
    ierr = PetscMalloc1(2*cnt,&counts);CHKERRQ(ierr);
    cnt  = 0;
    row  = 0;
    for (i=0; i<node_count; i++) {
      cols = aj + ai[row] + a->inode.size[i];
      counts[2*cnt] = cols[0];
      nz   = ai[row+1] - ai[row] - a->inode.size[i];
      cnt2 = 1;
      for (j=1; j<nz; j++) {
        if (cols[j] != cols[j-1]+1) {
          counts[2*(cnt++)+1] = cnt2;
          counts[2*cnt]       = cols[j];
          cnt2 = 1;
        } else cnt2++;
      }
      counts[2*(cnt++)+1] = cnt2;
      row += a->inode.size[i];
    }
    ierr = PetscIntView(2*cnt,counts,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqSBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       fshift = 0,i,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m      = A->rmap->N,*ip,N,*ailen = a->ilen;
  PetscInt       mbs    = a->mbs,bs2 = a->bs2,rmax = 0;
  MatScalar      *aa    = a->a,*ap;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + bs2*ai[i];
      N  = ailen[i];
      ierr  = PetscArraymove(ip-fshift,ip,N);CHKERRQ(ierr);
      ierr  = PetscArraymove(ap-bs2*fshift,ap,bs2*N);CHKERRQ(ierr);
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
    ierr = PetscArraycpy(a->diag,ai,mbs);CHKERRQ(ierr);
  }
  if (fshift && a->nounused == -1) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Unused space detected in matrix: %D X %D block size %D, %D unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);

  ierr = PetscInfo5(A,"Matrix size: %D X %D, block size %D; storage space: %D unneeded, %D used\n",m,A->rmap->N,A->rmap->bs,fshift*bs2,a->nz*bs2);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Most nonzeros blocks in any row is %D\n",rmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (PetscReal)fshift*bs2;
  a->idiagvalid       = PETSC_FALSE;
  a->rmax             = rmax;

  if (A->cmap->n < 65536 && A->cmap->bs == 1) {
    if (a->jshort && a->free_jshort) {
      /* when matrix data structure is changed, previous jshort must be replaced */
      ierr = PetscFree(a->jshort);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(a->i[A->rmap->n],&a->jshort);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,a->i[A->rmap->n]*sizeof(unsigned short));CHKERRQ(ierr);
    for (i=0; i<a->i[A->rmap->n]; i++) a->jshort[i] = a->j[i];
    A->ops->mult   = MatMult_SeqSBAIJ_1_ushort;
    A->ops->sor    = MatSOR_SeqSBAIJ_ushort;
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
PetscErrorCode MatZeroRows_SeqSBAIJ_Check_Blocks(PetscInt idx[],PetscInt n,PetscInt bs,PetscInt sizes[], PetscInt *bs_max)
{
  PetscInt  i,j,k,row;
  PetscBool flg;

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
        i       += bs;
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

PetscErrorCode MatSetValues_SeqSBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen,roworiented=a->roworiented;
  PetscInt       *aj  =a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscInt       ridx,cidx,bs2=a->bs2;
  MatScalar      *ap,value,*aa=a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];       /* row number */
    brow = row/bs;      /* block row number */
    if (row < 0) continue;
    if (PetscUnlikelyDebug(row >= A->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
    rp   = aj + ai[brow]; /*ptr to beginning of column value of the row block*/
    ap   = aa + bs2*ai[brow]; /*ptr to beginning of element value of the row block*/
    rmax = imax[brow];  /* maximum space allocated for this row */
    nrow = ailen[brow]; /* actual length of this row */
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      if (PetscUnlikelyDebug(in[l] >= A->cmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap->N-1);
      col  = in[l];
      bcol = col/bs;              /* block col number */

      if (brow > bcol) {
        if (a->ignore_ltriangular) continue; /* ignore lower triangular values */
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
      }

      ridx = row % bs; cidx = col % bs; /*row and col index inside the block */
      if ((brow==bcol && ridx<=cidx) || (brow<bcol)) {
        /* element value a(k,l) */
        if (roworiented) value = v[l + k*n];
        else value = v[k + l*m];

        /* move pointer bap to a(k,l) quickly and add/insert value */
        if (col <= lastcol) low = 0;
        else high = nrow;

        lastcol = col;
        while (high-low > 7) {
          t = (low+high)/2;
          if (rp[t] > bcol) high = t;
          else              low  = t;
        }
        for (i=low; i<high; i++) {
          if (rp[i] > bcol) break;
          if (rp[i] == bcol) {
            bap = ap +  bs2*i + bs*cidx + ridx;
            if (is == ADD_VALUES) *bap += value;
            else                  *bap  = value;
            /* for diag block, add/insert its symmetric element a(cidx,ridx) */
            if (brow == bcol && ridx < cidx) {
              bap = ap +  bs2*i + bs*ridx + cidx;
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
        ierr = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
        ierr = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
        ierr = PetscArrayzero(ap+bs2*i,bs2);CHKERRQ(ierr);
        rp[i]                      = bcol;
        ap[bs2*i + bs*cidx + ridx] = value;
        /* for diag block, add/insert its symmetric element a(cidx,ridx) */
        if (brow == bcol && ridx < cidx) {
          ap[bs2*i + bs*ridx + cidx] = value;
        }
        A->nonzerostate++;
noinsert1:;
        low = i;
      }
    }   /* end of loop over added columns */
    ailen[brow] = nrow;
  }   /* end of loop over added rows */
  PetscFunctionReturn(0);
}

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
  ierr = PetscFree(inA->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&inA->solvertype);CHKERRQ(ierr);

  ierr = MatMarkDiagonal_SeqSBAIJ(inA);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetNumericFactorization_inplace(inA,row_identity);CHKERRQ(ierr);

  ierr   = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr   = ISDestroy(&a->row);CHKERRQ(ierr);
  a->row = row;
  ierr   = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr   = ISDestroy(&a->col);CHKERRQ(ierr);
  a->col = row;

  /* Create the invert permutation so that it can be used in MatCholeskyFactorNumeric() */
  if (a->icol) {ierr = ISInvertPermutation(row,PETSC_DECIDE, &a->icol);CHKERRQ(ierr);}
  ierr = PetscLogObjectParent((PetscObject)inA,(PetscObject)a->icol);CHKERRQ(ierr);

  if (!a->solve_work) {
    ierr = PetscMalloc1(inA->rmap->N+inA->rmap->bs,&a->solve_work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)inA,(inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = MatCholeskyFactorNumeric(outA,inA,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqSBAIJSetColumnIndices_SeqSBAIJ(Mat mat,PetscInt *indices)
{
  Mat_SeqSBAIJ   *baij = (Mat_SeqSBAIJ*)mat->data;
  PetscInt       i,nz,n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nz = baij->maxnz;
  n  = mat->cmap->n;
  for (i=0; i<nz; i++) baij->j[i] = indices[i];

  baij->nz = nz;
  for (i=0; i<n; i++) baij->ilen[i] = baij->imax[i];

  ierr = MatSetOption(mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = PetscUseMethod(mat,"MatSeqSBAIJSetColumnIndices_C",(Mat,PetscInt*),(mat,indices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqSBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  PetscBool      isbaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&isbaij,MATSEQSBAIJ,MATMPISBAIJ,"");CHKERRQ(ierr);
  if (!isbaij) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)B)->type_name);
  /* If the two matrices have the same copy implementation and nonzero pattern, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && A->ops->copy == B->ops->copy) {
    Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
    Mat_SeqSBAIJ *b = (Mat_SeqSBAIJ*)B->data;

    if (a->i[a->mbs] != b->i[b->mbs]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    if (a->mbs != b->mbs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of rows in two matrices are different");
    if (a->bs2 != b->bs2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Different block size");
    ierr = PetscArraycpy(b->a,a->a,a->bs2*a->i[a->mbs]);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  } else {
    ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqSBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqSBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqSBAIJGetArray_SeqSBAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqSBAIJRestoreArray_SeqSBAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPYGetPreallocation_SeqSBAIJ(Mat Y,Mat X,PetscInt *nnz)
{
  PetscInt       bs = Y->rmap->bs,mbs = Y->rmap->N/bs;
  Mat_SeqSBAIJ   *x = (Mat_SeqSBAIJ*)X->data;
  Mat_SeqSBAIJ   *y = (Mat_SeqSBAIJ*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  ierr = MatAXPYGetPreallocation_SeqX_private(mbs,x->i,x->j,y->i,y->j,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqSBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_SeqSBAIJ   *x=(Mat_SeqSBAIJ*)X->data, *y=(Mat_SeqSBAIJ*)Y->data;
  PetscErrorCode ierr;
  PetscInt       bs=Y->rmap->bs,bs2=bs*bs;
  PetscBLASInt   one = 1;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar  alpha = a;
    PetscBLASInt bnz;
    ierr = PetscBLASIntCast(x->nz*bs2,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one));
    ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    ierr = MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
    ierr = MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    Mat      B;
    PetscInt *nnz;
    if (bs != X->rmap->bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices must have same block size");
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatGetRowUpperTriangular(Y);CHKERRQ(ierr);
    ierr = PetscMalloc1(Y->rmap->N,&nnz);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)Y),&B);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name);CHKERRQ(ierr);
    ierr = MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(B,Y,Y);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)Y)->type_name);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_SeqSBAIJ(Y,X,nnz);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);

    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);

    ierr = MatHeaderReplace(Y,&B);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsSymmetric_SeqSBAIJ(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsStructurallySymmetric_SeqSBAIJ(Mat A,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsHermitian_SeqSBAIJ(Mat A,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_SeqSBAIJ(Mat A)
{
#if defined(PETSC_USE_COMPLEX)
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     i,nz = a->bs2*a->i[a->mbs];
  MatScalar    *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscConj(aa[i]);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     i,nz = a->bs2*a->i[a->mbs];
  MatScalar    *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt     i,nz = a->bs2*a->i[a->mbs];
  MatScalar    *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqSBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqSBAIJ      *baij=(Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,count;
  PetscInt          bs   =A->rmap->bs,bs2=baij->bs2,row,col;
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

  /* zero the columns */
  ierr = PetscCalloc1(A->rmap->n,&zeroed);CHKERRQ(ierr);
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
          if (!zeroed[i] && zeroed[col]) bb[i]   -= aa[0]*xx[col];
          if (zeroed[i] && !zeroed[col]) bb[col] -= aa[0]*xx[i];
        }
      }
    }
    for (i=0; i<is_n; i++) bb[is_idx[i]] = diag*xx[is_idx[i]];
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
      aa   += bs;
    }
    if (diag != 0.0) {
      ierr = (*A->ops->setvalues)(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd_SeqSBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqSBAIJ(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqSBAIJ    *aij = (Mat_SeqSBAIJ*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !aij->nz) {
    ierr = MatSeqSBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqSBAIJ,
                                       MatGetRow_SeqSBAIJ,
                                       MatRestoreRow_SeqSBAIJ,
                                       MatMult_SeqSBAIJ_N,
                               /*  4*/ MatMultAdd_SeqSBAIJ_N,
                                       MatMult_SeqSBAIJ_N,       /* transpose versions are same as non-transpose versions */
                                       MatMultAdd_SeqSBAIJ_N,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       MatCholeskyFactor_SeqSBAIJ,
                                       MatSOR_SeqSBAIJ,
                                       MatTranspose_SeqSBAIJ,
                               /* 15*/ MatGetInfo_SeqSBAIJ,
                                       MatEqual_SeqSBAIJ,
                                       MatGetDiagonal_SeqSBAIJ,
                                       MatDiagonalScale_SeqSBAIJ,
                                       MatNorm_SeqSBAIJ,
                               /* 20*/ NULL,
                                       MatAssemblyEnd_SeqSBAIJ,
                                       MatSetOption_SeqSBAIJ,
                                       MatZeroEntries_SeqSBAIJ,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ MatSetUp_SeqSBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ MatDuplicate_SeqSBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatICCFactor_SeqSBAIJ,
                               /* 39*/ MatAXPY_SeqSBAIJ,
                                       MatCreateSubMatrices_SeqSBAIJ,
                                       MatIncreaseOverlap_SeqSBAIJ,
                                       MatGetValues_SeqSBAIJ,
                                       MatCopy_SeqSBAIJ,
                               /* 44*/ NULL,
                                       MatScale_SeqSBAIJ,
                                       MatShift_SeqSBAIJ,
                                       NULL,
                                       MatZeroRowsColumns_SeqSBAIJ,
                               /* 49*/ NULL,
                                       MatGetRowIJ_SeqSBAIJ,
                                       MatRestoreRowIJ_SeqSBAIJ,
                                       NULL,
                                       NULL,
                               /* 54*/ NULL,
                                       NULL,
                                       NULL,
                                       MatPermute_SeqSBAIJ,
                                       MatSetValuesBlocked_SeqSBAIJ,
                               /* 59*/ MatCreateSubMatrix_SeqSBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ MatGetRowMaxAbs_SeqSBAIJ,
                                       NULL,
                                       MatConvert_MPISBAIJ_Basic,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       MatGetInertia_SeqSBAIJ,
                                       MatLoad_SeqSBAIJ,
                               /* 84*/ MatIsSymmetric_SeqSBAIJ,
                                       MatIsHermitian_SeqSBAIJ,
                                       MatIsStructurallySymmetric_SeqSBAIJ,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_SeqSBAIJ,
                                       NULL,
                               /*104*/ NULL,
                                       MatRealPart_SeqSBAIJ,
                                       MatImaginaryPart_SeqSBAIJ,
                                       MatGetRowUpperTriangular_SeqSBAIJ,
                                       MatRestoreRowUpperTriangular_SeqSBAIJ,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_SeqSBAIJ,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ MatSetBlockSizes_Default,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*144*/MatCreateMPIMatConcatenateSeqMat_SeqSBAIJ
};

PetscErrorCode  MatStoreValues_SeqSBAIJ(Mat mat)
{
  Mat_SeqSBAIJ   *aij = (Mat_SeqSBAIJ*)mat->data;
  PetscInt       nz   = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    ierr = PetscMalloc1(nz+1,&aij->saved_values);CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscArraycpy(aij->saved_values,aij->a,nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_SeqSBAIJ(Mat mat)
{
  Mat_SeqSBAIJ   *aij = (Mat_SeqSBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nz = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;

  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  if (!aij->saved_values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");

  /* copy values over */
  ierr = PetscArraycpy(aij->a,aij->saved_values,nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSeqSBAIJSetPreallocation_SeqSBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqSBAIJ   *b = (Mat_SeqSBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs,nbs,bs2;
  PetscBool      skipallocation = PETSC_FALSE,flg = PETSC_FALSE,realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;

  ierr = MatSetBlockSize(B,PetscAbs(bs));CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (B->rmap->N > B->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"SEQSBAIJ matrix cannot have more rows %D than columns %D",B->rmap->N,B->cmap->N);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;

  mbs = B->rmap->N/bs;
  nbs = B->cmap->n/bs;
  bs2 = bs*bs;

  if (mbs*bs != B->rmap->N || nbs*bs!=B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows, cols must be divisible by blocksize");

  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 3;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %D",nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %D value %D",i,nnz[i]);
      if (nnz[i] > nbs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %D value %D block rowlength %D",i,nnz[i],nbs);
    }
  }

  B->ops->mult             = MatMult_SeqSBAIJ_N;
  B->ops->multadd          = MatMultAdd_SeqSBAIJ_N;
  B->ops->multtranspose    = MatMult_SeqSBAIJ_N;
  B->ops->multtransposeadd = MatMultAdd_SeqSBAIJ_N;

  ierr  = PetscOptionsGetBool(((PetscObject)B)->options,((PetscObject)B)->prefix,"-mat_no_unroll",&flg,NULL);CHKERRQ(ierr);
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
  b->nbs = nbs;
  if (!skipallocation) {
    if (!b->imax) {
      ierr = PetscMalloc2(mbs,&b->imax,mbs,&b->ilen);CHKERRQ(ierr);

      b->free_imax_ilen = PETSC_TRUE;

      ierr = PetscLogObjectMemory((PetscObject)B,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz <= 0) nz = 1;
      nz = PetscMin(nbs,nz);
      for (i=0; i<mbs; i++) b->imax[i] = nz;
      ierr = PetscIntMultError(nz,mbs,&nz);CHKERRQ(ierr);
    } else {
      PetscInt64 nz64 = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz64 += nnz[i];}
      ierr = PetscIntCast(nz64,&nz);CHKERRQ(ierr);
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) b->ilen[i] = 0;
    /* nz=(nz+mbs)/2; */ /* total diagonal and superdiagonal nonzero blocks */

    /* allocate the matrix space */
    ierr = MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(bs2*nz,&b->a,nz,&b->j,B->rmap->N+1,&b->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
    ierr = PetscArrayzero(b->a,nz*bs2);CHKERRQ(ierr);
    ierr = PetscArrayzero(b->j,nz);CHKERRQ(ierr);

    b->singlemalloc = PETSC_TRUE;

    /* pointer to beginning of each row */
    b->i[0] = 0;
    for (i=1; i<mbs+1; i++) b->i[i] = b->i[i-1] + b->imax[i-1];

    b->free_a  = PETSC_TRUE;
    b->free_ij = PETSC_TRUE;
  } else {
    b->free_a  = PETSC_FALSE;
    b->free_ij = PETSC_FALSE;
  }

  b->bs2     = bs2;
  b->nz      = 0;
  b->maxnz   = nz;
  b->inew    = NULL;
  b->jnew    = NULL;
  b->anew    = NULL;
  b->a2anew  = NULL;
  b->permute = PETSC_FALSE;

  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  if (realalloc) {ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqSBAIJSetPreallocationCSR_SeqSBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[], const PetscScalar V[])
{
  PetscInt       i,j,m,nz,anz, nz_max=0,*nnz;
  PetscScalar    *values=NULL;
  PetscBool      roworiented = ((Mat_SeqSBAIJ*)B->data)->roworiented;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  ierr   = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr   = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m      = B->rmap->n/bs;

  if (ii[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %D",ii[0]);
  ierr = PetscMalloc1(m+1,&nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    if (nz < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D has a negative number of columns %D",i,nz);
    anz = 0;
    for (j=0; j<nz; j++) {
      /* count only values on the diagonal or above */
      if (jj[ii[i] + j] >= i) {
        anz = nz - j;
        break;
      }
    }
    nz_max = PetscMax(nz_max,anz);
    nnz[i] = anz;
  }
  ierr = MatSeqSBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscCalloc1(bs*bs*nz_max,&values);CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (!roworiented || bs == 1) {
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      ierr = MatSetValuesBlocked_SeqSBAIJ(B,1,&i,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_SeqSBAIJ(B,1,&i,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   This is used to set the numeric factorization for both Cholesky and ICC symbolic factorization
*/
PetscErrorCode MatSeqSBAIJSetNumericFactorization_inplace(Mat B,PetscBool natural)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       bs  = B->rmap->bs;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject)B)->options,((PetscObject)B)->prefix,"-mat_no_unroll",&flg,NULL);CHKERRQ(ierr);
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

PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_SeqAIJ(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_SeqBAIJ(Mat,MatType,MatReuse,Mat*);

PETSC_INTERN PetscErrorCode MatGetFactor_seqsbaij_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (A->hermitian && !A->symmetric && (ftype == MAT_FACTOR_CHOLESKY||ftype == MAT_FACTOR_ICC)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian CHOLESKY or ICC Factor is not supported");
#endif

  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    ierr = MatSetType(*B,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*B,A->rmap->bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ;
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqSBAIJ;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  (*B)->factortype = ftype;
  (*B)->canuseordering = PETSC_TRUE;
  ierr = PetscFree((*B)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&(*B)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqSBAIJGetArray - gives access to the array where the data for a MATSEQSBAIJ matrix is stored

   Not Collective

   Input Parameter:
.  mat - a MATSEQSBAIJ matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: MatSeqSBAIJRestoreArray(), MatSeqAIJGetArray(), MatSeqAIJRestoreArray()
@*/
PetscErrorCode  MatSeqSBAIJGetArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqSBAIJGetArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqSBAIJRestoreArray - returns access to the array where the data for a MATSEQSBAIJ matrix is stored obtained by MatSeqSBAIJGetArray()

   Not Collective

   Input Parameters:
+  mat - a MATSEQSBAIJ matrix
-  array - pointer to the data

   Level: intermediate

.seealso: MatSeqSBAIJGetArray(), MatSeqAIJGetArray(), MatSeqAIJRestoreArray()
@*/
PetscErrorCode  MatSeqSBAIJRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqSBAIJRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSEQSBAIJ - MATSEQSBAIJ = "seqsbaij" - A matrix type to be used for sequential symmetric block sparse matrices,
  based on block compressed sparse row format.  Only the upper triangular portion of the matrix is stored.

  For complex numbers by default this matrix is symmetric, NOT Hermitian symmetric. To make it Hermitian symmetric you
  can call MatSetOption(Mat, MAT_HERMITIAN).

  Options Database Keys:
  . -mat_type seqsbaij - sets the matrix type to "seqsbaij" during a call to MatSetFromOptions()

  Notes:
    By default if you insert values into the lower triangular part of the matrix they are simply ignored (since they are not
     stored and it is assumed they symmetric to the upper triangular). If you call MatSetOption(Mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_FALSE) or use
     the options database -mat_ignore_lower_triangular false it will generate an error if you try to set a value in the lower triangular portion.

    The number of rows in the matrix must be less than or equal to the number of columns

  Level: beginner

  .seealso: MatCreateSeqSBAIJ(), MatType, MATMPISBAIJ
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqSBAIJ(Mat B)
{
  Mat_SeqSBAIJ   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      no_unroll = PETSC_FALSE,no_inode = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  B->ops->destroy    = MatDestroy_SeqSBAIJ;
  B->ops->view       = MatView_SeqSBAIJ;
  b->row             = NULL;
  b->icol            = NULL;
  b->reallocs        = 0;
  b->saved_values    = NULL;
  b->inode.limit     = 5;
  b->inode.max_limit = 5;

  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = NULL;
  b->solve_work         = NULL;
  b->mult_work          = NULL;
  B->spptr              = NULL;
  B->info.nz_unneeded   = (PetscReal)b->maxnz*b->bs2;
  b->keepnonzeropattern = PETSC_FALSE;

  b->inew    = NULL;
  b->jnew    = NULL;
  b->anew    = NULL;
  b->a2anew  = NULL;
  b->permute = PETSC_FALSE;

  b->ignore_ltriangular = PETSC_TRUE;

  ierr = PetscOptionsGetBool(((PetscObject)B)->options,((PetscObject)B)->prefix,"-mat_ignore_lower_triangular",&b->ignore_ltriangular,NULL);CHKERRQ(ierr);

  b->getrow_utriangular = PETSC_FALSE;

  ierr = PetscOptionsGetBool(((PetscObject)B)->options,((PetscObject)B)->prefix,"-mat_getrow_uppertriangular",&b->getrow_utriangular,NULL);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqSBAIJGetArray_C",MatSeqSBAIJGetArray_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqSBAIJRestoreArray_C",MatSeqSBAIJRestoreArray_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqSBAIJSetColumnIndices_C",MatSeqSBAIJSetColumnIndices_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_seqaij_C",MatConvert_SeqSBAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_seqbaij_C",MatConvert_SeqSBAIJ_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqSBAIJSetPreallocation_C",MatSeqSBAIJSetPreallocation_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqSBAIJSetPreallocationCSR_C",MatSeqSBAIJSetPreallocationCSR_SeqSBAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_elemental_C",MatConvert_SeqSBAIJ_Elemental);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_scalapack_C",MatConvert_SBAIJ_ScaLAPACK);CHKERRQ(ierr);
#endif

  B->symmetric                  = PETSC_TRUE;
  B->structurally_symmetric     = PETSC_TRUE;
  B->symmetric_set              = PETSC_TRUE;
  B->structurally_symmetric_set = PETSC_TRUE;
  B->symmetric_eternal          = PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
  B->hermitian                  = PETSC_FALSE;
  B->hermitian_set              = PETSC_FALSE;
#else
  B->hermitian                  = PETSC_TRUE;
  B->hermitian_set              = PETSC_TRUE;
#endif

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSBAIJ);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),((PetscObject)B)->prefix,"Options for SEQSBAIJ matrix","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_no_unroll","Do not optimize for inodes (slower)",NULL,no_unroll,&no_unroll,NULL);CHKERRQ(ierr);
  if (no_unroll) {
    ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_unroll\n");CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-mat_no_inode","Do not optimize for inodes (slower)",NULL,no_inode,&no_inode,NULL);CHKERRQ(ierr);
  if (no_inode) {
    ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_inode\n");CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-mat_inode_limit","Do not use inodes larger then this value",NULL,b->inode.limit,&b->inode.limit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  b->inode.use = (PetscBool)(!(no_unroll || no_inode));
  if (b->inode.limit > b->inode.max_limit) b->inode.limit = b->inode.max_limit;
  PetscFunctionReturn(0);
}

/*@C
   MatSeqSBAIJSetPreallocation - Creates a sparse symmetric matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on Mat

   Input Parameters:
+  B - the symmetric matrix
.  bs - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the upper triangular plus
         diagonal portion of each block (possibly different for each block row) or NULL

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-   -mat_block_size - size of the blocks to use (only works if a negative bs is passed in

   Level: intermediate

   Notes:
   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.

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

/*@C
   MatSeqSBAIJSetPreallocationCSR - Creates a sparse parallel matrix in SBAIJ format using the given nonzero structure and (optional) numerical values

   Input Parameters:
+  B - the matrix
.  bs - size of block, the blocks are ALWAYS square.
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: advanced

   Notes:
   The order of the entries in values is specified by the MatOption MAT_ROW_ORIENTED.  For example, C programs
   may want to use the default MAT_ROW_ORIENTED=PETSC_TRUE and use an array v[nnz][bs][bs] where the second index is
   over rows within a block and the last index is over columns within a block row.  Fortran programs will likely set
   MAT_ROW_ORIENTED=PETSC_FALSE and use a Fortran array v(bs,bs,nnz) in which the first index is over rows within a
   block column and the second index is over columns within a block.

   Any entries below the diagonal are ignored

   Though this routine has Preallocation() in the name it also sets the exact nonzero locations of the matrix entries
   and usually the numerical values as well

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValuesBlocked(), MatSeqSBAIJSetPreallocation(), MATSEQSBAIJ
@*/
PetscErrorCode MatSeqSBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatSeqSBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqSBAIJ - Creates a sparse symmetric matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  m - number of rows, or number of columns
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the upper triangular plus
         diagonal portion of each block (possibly different for each block row) or NULL

   Output Parameter:
.  A - the symmetric matrix

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-   -mat_block_size - size of the blocks to use

   Level: intermediate

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   The number of rows and columns must be divisible by blocksize.
   This matrix type does not support complex Hermitian operation.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.

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
  ierr = MatSeqSBAIJSetPreallocation(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqSBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat            C;
  Mat_SeqSBAIJ   *c,*a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 =a->bs2;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt matrix");

  *B   = NULL;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,A->rmap->N,A->cmap->n,A->rmap->N,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,A);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQSBAIJ);CHKERRQ(ierr);
  c    = (Mat_SeqSBAIJ*)C->data;

  C->preallocated       = PETSC_TRUE;
  C->factortype         = A->factortype;
  c->row                = NULL;
  c->icol               = NULL;
  c->saved_values       = NULL;
  c->keepnonzeropattern = a->keepnonzeropattern;
  C->assembled          = PETSC_TRUE;

  ierr   = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);
  c->bs2 = a->bs2;
  c->mbs = a->mbs;
  c->nbs = a->nbs;

  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax           = a->imax;
    c->ilen           = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    ierr = PetscMalloc2((mbs+1),&c->imax,(mbs+1),&c->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,2*(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i];
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    ierr            = PetscMalloc1(bs2*nz,&c->a);CHKERRQ(ierr);
    ierr            = PetscLogObjectMemory((PetscObject)C,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
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
    ierr            = PetscMalloc3(bs2*nz,&c->a,nz,&c->j,mbs+1,&c->i);CHKERRQ(ierr);
    ierr            = PetscArraycpy(c->i,a->i,mbs+1);CHKERRQ(ierr);
    ierr            = PetscLogObjectMemory((PetscObject)C,(mbs+1)*sizeof(PetscInt) + nz*(bs2*sizeof(MatScalar) + sizeof(PetscInt)));CHKERRQ(ierr);
    c->singlemalloc = PETSC_TRUE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
  }
  if (mbs > 0) {
    if (cpvalues != MAT_SHARE_NONZERO_PATTERN) {
      ierr = PetscArraycpy(c->j,a->j,nz);CHKERRQ(ierr);
    }
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscArraycpy(c->a,a->a,bs2*nz);CHKERRQ(ierr);
    } else {
      ierr = PetscArrayzero(c->a,bs2*nz);CHKERRQ(ierr);
    }
    if (a->jshort) {
      /* cannot share jshort, it is reallocated in MatAssemblyEnd_SeqSBAIJ() */
      /* if the parent matrix is reassembled, this child matrix will never notice */
      ierr = PetscMalloc1(nz,&c->jshort);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,nz*sizeof(unsigned short));CHKERRQ(ierr);
      ierr = PetscArraycpy(c->jshort,a->jshort,nz);CHKERRQ(ierr);

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
      ierr = PetscMalloc1(mbs,&c->diag);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,mbs*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0; i<mbs; i++) c->diag[i] = a->diag[i];
      c->free_diag = PETSC_TRUE;
    }
  }
  c->nz         = a->nz;
  c->maxnz      = a->nz; /* Since we allocate exactly the right amount */
  c->solve_work = NULL;
  c->mult_work  = NULL;

  *B   = C;
  ierr = PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Used for both SeqBAIJ and SeqSBAIJ matrices */
#define MatLoad_SeqSBAIJ_Binary MatLoad_SeqBAIJ_Binary

PetscErrorCode MatLoad_SeqSBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ2(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  ierr = MatLoad_SeqSBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     MatCreateSeqSBAIJWithArrays - Creates an sequential SBAIJ matrix using matrix elements
              (upper triangular entries in CSR format) provided by the user.

     Collective

   Input Parameters:
+  comm - must be an MPI communicator of size 1
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  i - row indices; that is i[0] = 0, i[row] = i[row-1] + number of block elements in that row block row of the matrix
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
PetscErrorCode  MatCreateSeqSBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt i[],PetscInt j[],PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqSBAIJ   *sbaij;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"block size %D > 1 is not supported yet",bs);
  if (m > 0 && i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");

  ierr  = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr  = MatSetSizes(*mat,m,n,m,n);CHKERRQ(ierr);
  ierr  = MatSetType(*mat,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr  = MatSeqSBAIJSetPreallocation(*mat,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  sbaij = (Mat_SeqSBAIJ*)(*mat)->data;
  ierr  = PetscMalloc2(m,&sbaij->imax,m,&sbaij->ilen);CHKERRQ(ierr);
  ierr  = PetscLogObjectMemory((PetscObject)*mat,2*m*sizeof(PetscInt));CHKERRQ(ierr);

  sbaij->i = i;
  sbaij->j = j;
  sbaij->a = a;

  sbaij->singlemalloc   = PETSC_FALSE;
  sbaij->nonew          = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  sbaij->free_a         = PETSC_FALSE;
  sbaij->free_ij        = PETSC_FALSE;
  sbaij->free_imax_ilen = PETSC_TRUE;

  for (ii=0; ii<m; ii++) {
    sbaij->ilen[ii] = sbaij->imax[ii] = i[ii+1] - i[ii];
    if (PetscUnlikelyDebug(i[ii+1] - i[ii] < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
  }
  if (PetscDefined(USE_DEBUG)) {
    for (ii=0; ii<sbaij->i[m]; ii++) {
      if (j[ii] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %d index = %d",ii,j[ii]);
      if (j[ii] > n - 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %d index = %d",ii,j[ii]);
    }
  }

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqSBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1 && scall == MAT_REUSE_MATRIX) {
    ierr = MatCopy(inmat,*outmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIMatConcatenateSeqMat_MPISBAIJ(comm,inmat,n,scall,outmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
