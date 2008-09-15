#define PETSCMAT_DLL

/*
    Factorization code for BAIJ format. 
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/ilu.h"

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetNumericFactorization"
/*
   This is used to set the numeric factorization for both LU and ILU symbolic factorization
*/
PetscErrorCode MatSeqBAIJSetNumericFactorization(Mat inA,PetscTruth natural)
{
  PetscFunctionBegin;
  if (natural) {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering;
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering;
      break; 
    case 4:
#if defined(PETSC_USE_MAT_SINGLE)
      {
        PetscTruth  sse_enabled_local;
        PetscErrorCode ierr;
        ierr = PetscSSEIsEnabled(inA->comm,&sse_enabled_local,PETSC_NULL);CHKERRQ(ierr);
        if (sse_enabled_local) {
#  if defined(PETSC_HAVE_SSE)
          int i,*AJ=a->j,nz=a->nz,n=a->mbs;
          if (n==(unsigned short)n) {
            unsigned short *aj=(unsigned short *)AJ;
            for (i=0;i<nz;i++) {
              aj[i] = (unsigned short)AJ[i];
            }
            inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE_usj;
            inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj;
            ierr = PetscInfo(inA,"Using special SSE, in-place natural ordering, ushort j index factor BS=4\n");CHKERRQ(ierr);
          } else {
        /* Scale the column indices for easier indexing in MatSolve. */
/*            for (i=0;i<nz;i++) { */
/*              AJ[i] = AJ[i]*4; */
/*            } */
            inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE;
            inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE;
            ierr = PetscInfo(inA,"Using special SSE, in-place natural ordering, int j index factor BS=4\n");CHKERRQ(ierr);
          }
#  else
        /* This should never be reached.  If so, problem in PetscSSEIsEnabled. */
          SETERRQ(PETSC_ERR_SUP,"SSE Hardware unavailable");
#  endif
        } else {
          inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
        }
      }
#else
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
#endif
      break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering;
      break;
    case 6: 
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering;
      break; 
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering;
      break; 
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;  
      break;
    }
  } else {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;  
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;  
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;  
      break;
    case 4:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;  
      break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;  
      break;
    case 6:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6;  
      break;
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7;  
      break;
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;  
      break;
    }
  }
  PetscFunctionReturn(0); 
}

/*
    The symbolic factorization code is identical to that for AIJ format,
  except for very small changes since this is now a SeqBAIJ datastructure.
  NOT good code reuse.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqBAIJ"
PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *B)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b;
  IS             isicol;
  PetscErrorCode ierr;
  PetscInt       *r,*ic,i,n = a->mbs,*ai = a->i,*aj = a->j;
  PetscInt       *ainew,*ajnew,jmax,*fill,*ajtmp,nz,bs = A->rmap->bs,bs2=a->bs2;
  PetscInt       *idnew,idx,row,m,fm,nnz,nzi,reallocs = 0,nzbd,*im;
  PetscReal      f = 1.0;
  PetscTruth     row_identity,col_identity;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_ERR_ARG_WRONG,"matrix must be square");
  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  f = info->fill;
  /* get new row pointers */
  ierr     = PetscMalloc((n+1)*sizeof(PetscInt),&ainew);CHKERRQ(ierr);
  ainew[0] = 0;
  /* don't know how many column pointers are needed so estimate */
  jmax     = (PetscInt)(f*ai[n] + 1);
  ierr     = PetscMalloc((jmax)*sizeof(PetscInt),&ajnew);CHKERRQ(ierr);
  /* fill is a linked list of nonzeros in active row */
  ierr     = PetscMalloc((2*n+1)*sizeof(PetscInt),&fill);CHKERRQ(ierr);
  im       = fill + n + 1;
  /* idnew is location of diagonal in factor */
  ierr     = PetscMalloc((n+1)*sizeof(PetscInt),&idnew);CHKERRQ(ierr);
  idnew[0] = 0;

  for (i=0; i<n; i++) {
    /* first copy previous fill into linked list */
    nnz     = nz    = ai[r[i]+1] - ai[r[i]];
    if (!nz) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix");
    ajtmp   = aj + ai[r[i]];
    fill[n] = n;
    while (nz--) {
      fm  = n;
      idx = ic[*ajtmp++];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
    }
    row = fill[n];
    while (row < i) {
      ajtmp = ajnew + idnew[row] + 1;
      nzbd  = 1 + idnew[row] - ainew[row];
      nz    = im[row] - nzbd;
      fm    = row;
      while (nz-- > 0) {
        idx = *ajtmp++;
        nzbd++;
        if (idx == i) im[row] = nzbd;
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nnz++;
        }
      }
      row = fill[row];
    }
    /* copy new filled row into permanent storage */
    ainew[i+1] = ainew[i] + nnz;
    if (ainew[i+1] > jmax) {

      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      PetscInt maxadd = jmax;
      /* maxadd = (int)((f*(ai[n]+1)*(n-i+5))/n); */
      if (maxadd < nnz) maxadd = (n-i)*(nnz+1);
      jmax += maxadd;

      /* allocate a longer ajnew */
      ierr  = PetscMalloc(jmax*sizeof(PetscInt),&ajtmp);CHKERRQ(ierr);
      ierr  = PetscMemcpy(ajtmp,ajnew,ainew[i]*sizeof(PetscInt));CHKERRQ(ierr);
      ierr  = PetscFree(ajnew);CHKERRQ(ierr);
      ajnew = ajtmp;
      reallocs++; /* count how many times we realloc */
    }
    ajtmp = ajnew + ainew[i];
    fm    = fill[n];
    nzi   = 0;
    im[i] = nnz;
    while (nnz--) {
      if (fm < i) nzi++;
      *ajtmp++ = fm;
      fm       = fill[fm];
    }
    idnew[i] = ainew[i] + nzi;
  }

#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = ((PetscReal)ainew[n])/((PetscReal)ai[n]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,f,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G);\n",af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  ierr = PetscFree(fill);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(*B,bs,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*B,isicol);CHKERRQ(ierr);
  b = (Mat_SeqBAIJ*)(*B)->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a     = PETSC_TRUE;
  b->free_ij    = PETSC_TRUE;
  ierr          = PetscMalloc((ainew[n]+1)*sizeof(MatScalar)*bs2,&b->a);CHKERRQ(ierr);
  b->j          = ajnew;
  b->i          = ainew;
  b->diag       = idnew;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((bs*n+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  ierr = PetscLogObjectMemory(*B,(ainew[n]-n)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = ainew[n];
  
  (*B)->factor                 = MAT_FACTOR_LU;
  (*B)->info.factor_mallocs    = reallocs;
  (*B)->info.fill_ratio_given  = f;
  if (ai[n] != 0) {
    (*B)->info.fill_ratio_needed = ((PetscReal)ainew[n])/((PetscReal)ai[n]);
  } else {
    (*B)->info.fill_ratio_needed = 0.0;
  }
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetNumericFactorization(*B,row_identity && col_identity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 }

