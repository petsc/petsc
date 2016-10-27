
/*
  Defines the basic matrix operations for the ELL matrix storage format.
*/
#include <../src/mat/impls/ell/seq/ell.h>  /*I   "petscmat.h"  I*/
#include <petscblaslapack.h>
#include <petsc/private/kernels/blocktranspose.h>

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLSetPreallocation"
PetscErrorCode  MatSeqELLSetPreallocation(Mat B,PetscInt rlenmax,const PetscInt rlen[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatSeqELLSetPreallocation_C",(Mat,PetscInt,const PetscInt[]),(B,rlenmax,rlen));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLSetPreallocation_SeqELL"
PetscErrorCode  MatSeqELLSetPreallocation_SeqELL(Mat B,PetscInt maxallocrow,const PetscInt rlen[])
{
  Mat_SeqELL     *b;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      skipallocation=PETSC_FALSE,realalloc=PETSC_FALSE;

  PetscFunctionBegin;
  if (maxallocrow >= 0 || rlen) realalloc = PETSC_TRUE;
  if (maxallocrow == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    maxallocrow    = 0;
  }

  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;

  if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 5;
  if (maxallocrow < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"maxallocrow cannot be less than 0: value %D",maxallocrow);
  if (rlen) {
    for (i=0; i<B->rmap->n; i++) {
      if (rlen[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be less than 0: local row %D value %D",i,rlen[i]);
      if (rlen[i] > B->cmap->n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be greater than row length: local row %D value %D rowlength %D",i,rlen[i],B->cmap->n);
    }
  }

  B->preallocated = PETSC_TRUE;

  b = (Mat_SeqELL*)B->data;

  if (!skipallocation) {
    if (!b->rlen) {
      ierr = PetscMalloc1(B->rmap->n,&b->rlen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,B->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!rlen) {
      if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 10;
      else if (maxallocrow < 0) maxallocrow = 1;
    } else {
      maxallocrow = 0;
      for (i=0; i<B->rmap->n; i++) maxallocrow = PetscMax(maxallocrow,rlen[i]);
    }
    /* b->rlen will count nonzeros in each row so far. */
    for (i=0; i<B->rmap->n; i++) b->rlen[i] = 0;

    /* allocate the matrix space */
    /* FIXME: should B's old memory be unlogged? */
    ierr = MatSeqXELLFreeELL(B,&b->val,&b->colidx);CHKERRQ(ierr);
    ierr = PetscMalloc2(maxallocrow*B->rmap->n,&b->val,maxallocrow*B->rmap->n,&b->colidx);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)B,maxallocrow*B->rmap->n*(sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
    b->singlemalloc = PETSC_TRUE;
    b->free_val     = PETSC_TRUE;
    b->free_colidx  = PETSC_TRUE;
  } else {
    b->free_val    = PETSC_FALSE;
    b->free_colidx = PETSC_FALSE;
  }

  b->nz               = 0;
  b->maxallocrow      = maxallocrow;
  b->rlenmax          = maxallocrow;
  b->maxallocmat      = maxallocrow*B->rmap->n;
  B->info.nz_unneeded = (double)b->maxallocmat;
  if (realalloc) {
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqELL"
PetscErrorCode MatMult_SeqELL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqELL        *a=(Mat_SeqELL*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n; /* number of rows */
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
  PetscScalar       sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<(m>>3); i++) { /* loop over slices */
    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;
    sum4 = 0.0;
    sum5 = 0.0;
    sum6 = 0.0;
    sum7 = 0.0;
    sum8 = 0.0;
    for (j=0; j<8*a->rlenmax; j+=8) {
       sum1 += *aval++*x[*acolidx++]; 
       sum2 += *aval++*x[*acolidx++];
       sum3 += *aval++*x[*acolidx++];
       sum4 += *aval++*x[*acolidx++];
       sum5 += *aval++*x[*acolidx++];
       sum6 += *aval++*x[*acolidx++];
       sum7 += *aval++*x[*acolidx++];
       sum8 += *aval++*x[*acolidx++];
    }
    y[8*i]   = sum1;
    y[8*i+1] = sum2;
    y[8*i+2] = sum3;
    y[8*i+3] = sum4;
    y[8*i+4] = sum5;
    y[8*i+5] = sum6;
    y[8*i+6] = sum7;
    y[8*i+7] = sum8;
  }

  ierr = PetscLogFlops(2.0*a->nz-a->nonzerorowcnt);CHKERRQ(ierr); /* theoretical minimal FLOPs */
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmultadd.h>
#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqELL"
PetscErrorCode MatMultAdd_SeqELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqELL        *a=(Mat_SeqELL*)A->data;
  PetscScalar       *y,*z;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n; /* number of rows */
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
  PetscScalar       sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  for (i=0; i<(m>>3); i++) { /* loop over slices */
    sum1 = y[8*i];
    sum2 = y[8*i+1];
    sum3 = y[8*i+2];
    sum4 = y[8*i+3];
    sum5 = y[8*i+4];
    sum6 = y[8*i+5];
    sum7 = y[8*i+6];
    sum8 = y[8*i+7];
    for (j=0; j<8*a->rlenmax; j+=8) {
      sum1 += *aval++*x[*acolidx++];
      sum2 += *aval++*x[*acolidx++];
      sum3 += *aval++*x[*acolidx++];
      sum4 += *aval++*x[*acolidx++];
      sum5 += *aval++*x[*acolidx++];
      sum6 += *aval++*x[*acolidx++];
      sum7 += *aval++*x[*acolidx++];
      sum8 += *aval++*x[*acolidx++];
    }
    z[8*i]   = sum1;
    z[8*i+1] = sum2;
    z[8*i+2] = sum3;
    z[8*i+3] = sum4;
    z[8*i+4] = sum5;
    z[8*i+5] = sum6;
    z[8*i+6] = sum7;
    z[8*i+7] = sum8;
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Checks for missing diagonals
*/
#undef __FUNCT__
#define __FUNCT__ "MatMissingDiagonal_SeqELL"
PetscErrorCode MatMissingDiagonal_SeqELL(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_SeqELL *a = (Mat_SeqELL*)A->data;
  PetscInt   *diag,i;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !(a->colidx)) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n");
  } else {
    diag = a->diag;
    for (i=0; i<A->rmap->n; i++) {
      if (diag[i] == -1) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        PetscInfo1(A,"Matrix is missing diagonal number %D\n",i);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMarkDiagonal_SeqELL"
PetscErrorCode MatMarkDiagonal_SeqELL(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m=A->rmap->n,shift;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr         = PetscMalloc1(m,&a->diag);CHKERRQ(ierr);
    ierr         = PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt));CHKERRQ(ierr);
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<m; i++) { /* loop over rows */
    shift = (i & ~0x07)*a->rlenmax+(i & 0x07); /* starting index of the row i */
    a->diag[i] = -1;
    for (j=0; j<a->rlen[i]; j++) {
      if (a->colidx[shift+j*8] == i) {
        a->diag[i] = shift+j*8;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_SeqELL"
PetscErrorCode MatZeroEntries_SeqELL(Mat A)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(a->val,(a->rlenmax*A->rmap->n)*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqELL"
PetscErrorCode MatDestroy_SeqELL(Mat A)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D",A->rmap->n,A->cmap->n,a->nz);
#endif
  ierr = MatSeqXELLFreeELL(A,&a->val,&a->colidx);CHKERRQ(ierr);
  ierr = ISDestroy(&a->row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr);
  ierr = PetscFree(a->diag);CHKERRQ(ierr);
  ierr = PetscFree(a->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(a->rlen);CHKERRQ(ierr);
  ierr = PetscFree3(a->idiag,a->mdiag,a->ssor_work);CHKERRQ(ierr);
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);

  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
#endif
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqELLSetPreallocation_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_SeqELL"
PetscErrorCode MatSetOption_SeqELL(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
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
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_SeqELL"
PetscErrorCode MatGetDiagonal_SeqELL(Mat A,Vec v)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       i,j,n,shift;
  PetscScalar    *x,zero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");

  if (A->factortype == MAT_FACTOR_ILU || A->factortype == MAT_FACTOR_LU) {
    PetscInt *diag=a->diag;
    ierr = VecGetArray(v,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) x[i] = 1.0/a->val[diag[i]];
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* loop over rows */
    shift = (i & ~0x07)*a->rlenmax+(i & 0x07); /* starting index of the row i */
    x[i] = 0;
    for (j=0; j<a->rlen[i]; j++) {
      if (a->colidx[shift+j*8] == i) {
        x[i] = a->val[shift+j*8];
        break;
      }
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_SeqELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

#undef __FUNCT__
#define __FUNCT__ "MatGetValues_SeqELL"
PetscErrorCode MatGetValues_SeqELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqELL  *a = (Mat_SeqELL*)A->data;
  PetscInt    *cp,i,k,low,high,t,row,col,l;
  PetscInt    shift;
  MatScalar   *vp;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row<0) continue;
#if defined(PETSC_USE_DEBUG)
    if (row >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->n-1);
#endif
    shift = (row & ~0x07)*a->rlenmax+(row & 0x07); /* starting index of the row */
    cp = a->colidx+shift; /* pointer to the row */
    vp = a->val+shift; /* pointer to the row */
    for (l=0; l<n; l++) { /* loop over added rows */
      col = in[l];
      if (col<0) continue;
#if defined(PETSC_USE_DEBUG)
      if (col >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: row %D max %D",col,A->cmap->n-1);
#endif
      high = a->rlen[row]; low = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (*(cp+t*8) > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (*(cp+8*i) > col) break;
        if (*(cp+8*i) == col) {
          *v++ = *(vp+8*i);
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
#define __FUNCT__ "MatView_SeqELL_ASCII"
PetscErrorCode MatView_SeqELL_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqELL        *a = (Mat_SeqELL*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,shift;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscInt nofinalvalue = 0;
    /*
    if (m && ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != A->cmap->n-1))) {
      nofinalvalue = 1;
    }
    */
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",m,A->cmap->n);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %D \n",a->nz);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,4);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
      }
    }
    /*
    if (nofinalvalue) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e %18.16e\n",m,A->cmap->n,0.,0.);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",m,A->cmap->n,0.0);CHKERRQ(ierr);
#endif
    }
    */
    ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
        } else if (PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]));CHKERRQ(ierr);
        }
#else
        if (a->val[shift*8*j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]);CHKERRQ(ierr);}
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_DENSE) {
    PetscInt    cnt = 0,jcnt;
    PetscScalar value;
#if defined(PETSC_USE_COMPLEX)
    PetscBool   realonly = PETSC_TRUE;

    for (i=0; i<a->rlenmax*A->rmap->n; i++) {
      if (PetscImaginaryPart(a->val[i]) != 0.0) {
        realonly = PETSC_FALSE;
        break;
      }
    }
#endif

    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      jcnt = 0;
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      for (j=0; j<A->cmap->n; j++) {
        if (jcnt < a->rlen[i] && j == a->colidx[shift+8*j]) {
          value = a->val[cnt++];
          jcnt++;
        } else {
          value = 0.0;
        }
#if defined(PETSC_USE_COMPLEX)
        if (realonly) {
          ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",(double)PetscRealPart(value));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," %7.5e+%7.5e i ",(double)PetscRealPart(value),(double)PetscImaginaryPart(value));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",(double)value);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATRIXMARKET) {
    PetscInt fshift=1;
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate complex general\n");CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate real general\n");CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer,"%D %D %D\n", m, A->cmap->n, a->nz);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D %g %g\n", i+fshift,a->colidx[shift+8*j]+fshift,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D %g\n", i+fshift, a->colidx[shift+8*j]+fshift, (double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (A->factortype) {
      for (i=0; i<m; i++) {
        shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        /* L part */
        for (j=shift; j<a->diag[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j])));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(a->val[j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)a->val[j]);CHKERRQ(ierr);
#endif
        }
        /* diagonal */
        j = a->diag[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[j]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)PetscImaginaryPart(1.0/a->val[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)(-PetscImaginaryPart(1.0/a->val[j])));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)(1.0/a->val[j]));CHKERRQ(ierr);
#endif

        /* U part */
        for (j=a->diag[i]+1; j<shift+8*a->rlen[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j])));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(a->val[j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)a->val[j]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } else {
      for (i=0; i<m; i++) {
        shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->[j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#undef __FUNCT__
#define __FUNCT__ "MatView_SeqELL_Draw_Zoom"
PetscErrorCode MatView_SeqELL_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A  = (Mat) Aa;
  Mat_SeqELL        *a = (Mat_SeqELL*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,shift;
  int               color;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07); /* starting index of the row i */
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) >=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    color = PETSC_DRAW_CYAN;
    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (a->val[shift+8*j] !=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    color = PETSC_DRAW_RED;
    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) <=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscInt  count = 0;
    PetscDraw popup;

    for (i=0; i<a->rlenmax*A->rmap->n; i++) {
      if (PetscAbsScalar(a->val[i]) > maxv) maxv = PetscAbsScalar(a->val[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup,minv,maxv);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      shift = (i & ~0x07)*a->rlenmax+(i & 0x07);
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j=0; j<a->rlen[j]; j++) {
        x_l = a->colidx[shift+j*8];
        x_r = x_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(a->val[count]),minv,maxv);
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
        count++;
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
#undef __FUNCT__
#define __FUNCT__ "MatView_SeqELL_Draw"
PetscErrorCode MatView_SeqELL_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscBool      isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr  = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;         xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqELL_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqELL"
PetscErrorCode MatView_SeqELL(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_SeqELL_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    /* ierr = MatView_SeqELL_Binary(A,viewer);CHKERRQ(ierr); */
  } else if (isdraw) {
    ierr = MatView_SeqELL_Draw(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqELL"
PetscErrorCode MatAssemblyEnd_SeqELL(Mat A,MatAssemblyType mode)
{
  Mat_SeqELL  *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  ierr = MatMarkDiagonal_SeqELL(A);CHKERRQ(ierr);
  ierr = PetscInfo4(A,"Matrix size: %D X %D; storage space: %D nonzeros %D used\n",A->rmap->n,A->cmap->n,a->nz,a->rlenmax*A->rmap->n);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues() is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",a->rlenmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;

  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_SeqELL"
PetscErrorCode MatGetInfo_SeqELL(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqELL *a = (Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = (double)a->maxallocmat;
  info->nz_used      = (double)a->rlenmax*A->rmap->n; /* include padding zeros */
  info->nz_unneeded  = (double)(a->maxallocmat-a->rlenmax*A->rmap->n);
  info->assemblies   = (double)A->num_ass;
  info->mallocs      = (double)A->info.mallocs;
  info->memory       = ((PetscObject)A)->mem;
  if (A->factortype) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_SeqELL"
PetscErrorCode MatSetValues_SeqELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       shift,i,k,l,low,high,t,ii,row,col,nrow;
  PetscInt       *cp,nonew=a->nonew,lastcol=-1;
  MatScalar      *vp,value;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (row >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->n-1);
#endif
    shift = (row & ~0x07)*a->rlenmax+(row & 0x07); /* starting index of the row */
    cp    = a->colidx+shift; /* pointer to the row */
    vp    = a->val+shift; /* pointer to the row */
    nrow  = a->rlen[row];
    low   = 0;
    high  = nrow;

    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l];
      if (col<0) continue;
#if defined(PETSC_USE_DEBUG)
      if (col >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %D max %D",col,A->cmap->n-1);
#endif
      if (a->roworiented) {
        value = v[l+k*n];
      } else {
        value = v[k+l*m];
      }
      if ((value == 0.0 && a->ignorezeroentries) && (is == ADD_VALUES)) continue;

      /* search in this row for the specified colmun */
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (*(cp+t*8) > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (*(cp+i*8) > col) break;
        if (*(cp+i*8) == col) {
          if (is == ADD_VALUES) *(vp+i*8) += value;
          else *(vp+i*8) = value;
          low = i + 1;
          goto noinsert;
        }
      }

      if (value == 0.0 && a->ignorezeroentries) goto noinsert;
      if (nonew == 1) goto noinsert;
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      /* Allocate a new space only if there is a->rlen[row]==a->maxallocrow */
      MatSeqXELLReallocateELL(A,A->rmap->n,1,nrow,a->maxallocrow,row,col,cp,vp,nonew,MatScalar);
      /* add the new nonzero to the high position */
      for (ii=nrow-1; ii>=i; ii--) {
        *(cp+(ii+1)*8) = *(cp+ii*8);
        *(vp+(ii+1)*8) = *(vp+ii*8);
      }
      a->rlen[row]++;
      *(cp+i*8) = col;
      *(vp+i*8) = value;
      a->nz++;
      A->nonzerostate++;
      low = i+1; high++; nrow++;
noinsert:;
    }
    a->rlen[row] = nrow;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCopy_SeqELL"
PetscErrorCode MatCopy_SeqELL(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqELL *a = (Mat_SeqELL*)A->data;
    Mat_SeqELL *b = (Mat_SeqELL*)B->data;

    if (a->rlenmax != b->rlenmax) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    ierr = PetscMemcpy(b->val,a->val,(a->rlenmax*A->rmap->n)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_SeqELL"
PetscErrorCode MatSetUp_SeqELL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqELLSetPreallocation(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLGetArray_SeqELL"
PetscErrorCode MatSeqELLGetArray_SeqELL(Mat A,PetscScalar *array[])
{
  Mat_SeqELL *a=(Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  *array = a->val;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLRestoreArray_SeqELL"
PetscErrorCode MatSeqELLRestoreArray_SeqELL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRealPart_SeqELL"
PetscErrorCode MatRealPart_SeqELL(Mat A)
{
  Mat_SeqELL  *a=(Mat_SeqELL*)A->data;
  PetscInt    i,matsize=a->rlenmax*A->rmap->n;
  MatScalar   *aval=a->val;

  PetscFunctionBegin;
  for (i=0; i<matsize; i++) aval[i]=PetscRealPart(aval[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatImaginaryPart_SeqELL"
PetscErrorCode MatImaginaryPart_SeqELL(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       i,matsize=a->rlenmax*A->rmap->n;
  MatScalar      *aval=a->val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<matsize; i++) aval[i] = PetscImaginaryPart(aval[i]);
  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_SeqELL"
PetscErrorCode MatShift_SeqELL(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqELL     *y=(Mat_SeqELL*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !y->nz) {
    ierr = MatSeqELLSetPreallocation(Y,1,NULL);CHKERRQ(ierr);
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqELL,
                                       0,
                                       0,
                                       MatMult_SeqELL,
                               /* 4*/  MatMultAdd_SeqELL,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 10*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 15*/ MatGetInfo_SeqELL,
                                       MatEqual_SeqELL,
                                       MatGetDiagonal_SeqELL,
                                       0,
                                       0,
                               /* 20*/ 0,
                                       MatAssemblyEnd_SeqELL,
                                       MatSetOption_SeqELL,
                                       MatZeroEntries_SeqELL,
                               /* 24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 29*/ MatSetUp_SeqELL,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 34*/ MatDuplicate_SeqELL,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 39*/ 0,
                                       0,
                                       0,
                                       MatGetValues_SeqELL,
                                       MatCopy_SeqELL,
                               /* 44*/ 0,
                                       0,
                                       MatShift_SeqELL,
                                       0,
                                       0,
                               /* 49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 54*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 59*/ 0,
                                       MatDestroy_SeqELL,
                                       MatView_SeqELL,
                                       0,
                                       0,
                               /* 64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 69*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 74*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 79*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 84*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 89*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 94*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 99*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*104*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*109*/ 0,
                                       0,
                                       0,
                                       0,
                                       MatMissingDiagonal_SeqELL,
                               /*114*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*119*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*124*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*144*/0
};

#undef __FUNCT__
#define __FUNCT__ "MatStoreValues_SeqELL"
PetscErrorCode  MatStoreValues_SeqELL(Mat mat)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)mat->data;
  PetscErrorCode ierr;
  size_t         matsize=a->rlenmax*mat->rmap->n;

  PetscFunctionBegin;
  if (!a->nonew) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!a->saved_values) {
    ierr = PetscMalloc1(matsize+1,&a->saved_values);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(matsize+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscMemcpy(a->saved_values,a->val,matsize*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRetrieveValues_SeqELL"
PetscErrorCode  MatRetrieveValues_SeqELL(Mat mat)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       matsize=a->rlenmax*mat->rmap->n;

  PetscFunctionBegin;
  if (!a->nonew) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  if (!a->saved_values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
  /* copy values over */
  ierr = PetscMemcpy(a->val,a->saved_values,matsize*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLRestoreArray"
/*@C
 MatSeqELLRestoreArray - returns access to the array where the data for a MATSEQELL matrix is stored obtained by MatSeqELLGetArray()

 Not Collective

 Input Parameters:
 .  mat - a MATSEQELL matrix
 .  array - pointer to the data

 Level: intermediate

 .seealso: MatSeqELLGetArray(), MatSeqELLRestoreArrayF90()
 @*/
PetscErrorCode  MatSeqELLRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqELLRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqELL"
PETSC_EXTERN PetscErrorCode MatCreate_SeqELL(Mat B)
{
  Mat_SeqELL     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Comm must be of size 1");

  ierr = PetscNewLog(B,&b);CHKERRQ(ierr);

  B->data = (void*)b;

  ierr = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  b->row                = 0;
  b->col                = 0;
  b->icol               = 0;
  b->reallocs           = 0;
  b->ignorezeroentries  = PETSC_FALSE;
  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = 0;
  b->solve_work         = 0;
  B->spptr              = 0;
  b->saved_values       = 0;
  b->idiag              = 0;
  b->mdiag              = 0;
  b->ssor_work          = 0;
  b->omega              = 1.0;
  b->fshift             = 0.0;
  b->idiagvalid         = PETSC_FALSE;
  b->ibdiagvalid        = PETSC_FALSE;
  b->keepnonzeropattern = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLGetArray_C",MatSeqELLGetArray_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLRestoreArray_C",MatSeqELLRestoreArray_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLSetPreallocation_C",MatSeqELLSetPreallocation_SeqELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicateNoCreate_SeqELL"
/*
 Given a matrix generated with MatGetFactor() duplicates all the information in A into B
 */
PetscErrorCode MatDuplicateNoCreate_SeqELL(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqELL     *c,*a=(Mat_SeqELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m=A->rmap->n;

  PetscFunctionBegin;
  c = (Mat_SeqELL*)C->data;

  C->factortype = A->factortype;
  c->row        = 0;
  c->col        = 0;
  c->icol       = 0;
  c->reallocs   = 0;

  C->assembled = PETSC_TRUE;

  ierr = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);

  ierr = PetscMalloc1(m,&c->rlen);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)C, m*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<m; i++) c->rlen[i] = a->rlen[i];

  /* allocate the matrix space */
  if (mallocmatspace) {
    ierr = PetscMalloc2(a->maxallocmat,&c->val,a->maxallocmat,&c->colidx);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C, a->maxallocmat*(sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);

    c->singlemalloc = PETSC_TRUE;

    if (m > 0) {
      ierr = PetscMemcpy(c->colidx,a->colidx,(a->maxallocmat)*sizeof(PetscInt));CHKERRQ(ierr);
      if (cpvalues == MAT_COPY_VALUES) {
        ierr = PetscMemcpy(c->val,a->val,a->maxallocmat*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(c->val,a->maxallocmat*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
  }

  c->ignorezeroentries = a->ignorezeroentries;
  c->roworiented       = a->roworiented;
  c->nonew             = a->nonew;
  if (a->diag) {
    ierr = PetscMalloc1(m+1,&c->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag = 0;

  c->solve_work         = 0;
  c->saved_values       = 0;
  c->idiag              = 0;
  c->ssor_work          = 0;
  c->keepnonzeropattern = a->keepnonzeropattern;
  c->free_val           = PETSC_TRUE;
  c->free_colidx        = PETSC_TRUE;

  c->maxallocmat  = a->maxallocmat;
  c->maxallocrow  = a->maxallocrow;
  c->rlenmax      = a->rlenmax;
  c->nz           = a->nz;
  C->preallocated = PETSC_TRUE;

  c->nonzerorowcnt = a->nonzerorowcnt;
  C->nonzerostate  = A->nonzerostate;

  ierr = PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SeqELL"
PetscErrorCode MatDuplicate_SeqELL(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (!(A->rmap->n % A->rmap->bs) && !(A->cmap->n % A->cmap->bs)) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
  }
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqELL(*B,A,cpvalues,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqELL"
/*@C
 MatCreateSeqELL - Creates a sparse matrix in ELL format.

 Collective on MPI_Comm

 Input Parameters:
 +  comm - MPI communicator, set to PETSC_COMM_SELF
 .  m - number of rows
 .  n - number of columns
 .  rlenmax - maximum number of nonzeros in a row
 -  rlen - array containing the number of nonzeros in the various rows
 (possibly different for each row) or NULL

 Output Parameter:
 .  A - the matrix

 It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
 MatXXXXSetPreallocation() paradgm instead of this routine directly.
 [MatXXXXSetPreallocation() is, for example, MatSeqELLSetPreallocation]

 Notes:
 If nnz is given then nz is ignored

 Specify the preallocated storage with either rlenmax or rlen (not both).
 Set rlenmax=PETSC_DEFAULT and rlen=NULL for PETSc to control dynamic memory
 allocation.  For large problems you MUST preallocate memory or you
 will get TERRIBLE performance, see the users' manual chapter on matrices.

 Level: intermediate

 .seealso: MatCreate(), MatCreateELL(), MatSetValues(), MatCreateSeqELLWithArrays()

 @*/
PetscErrorCode  MatCreateSeqELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt maxallocrow,const PetscInt rlen[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQELL);CHKERRQ(ierr);
  ierr = MatSeqELLSetPreallocation_SeqELL(*A,maxallocrow,rlen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatEqual_SeqELL"
PetscErrorCode MatEqual_SeqELL(Mat A,Mat B,PetscBool * flg)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data,*b=(Mat_SeqELL*)B->data;
  PetscInt       matsize;
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt k;
#endif

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) || (A->cmap->n != B->cmap->n) ||(a->nz != b->nz) || (a->rlenmax != b->rlenmax)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  matsize = a->rlenmax*A->rmap->n;
  /* if the a->colidx are the same */
  ierr = PetscMemcmp(a->colidx,b->colidx,(matsize)*sizeof(PetscInt),flg);CHKERRQ(ierr);
  if (!*flg) PetscFunctionReturn(0);

  /* if a->val are the same */
#if defined(PETSC_USE_COMPLEX)
  for (k=0; k<matsize; k++) {
    if (PetscRealPart(a->val[k]) != PetscRealPart(b->val[k]) || PetscImaginaryPart(a->val[k]) != PetscImaginaryPart(b->val[k])) {
      *flg = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
#else
  ierr = PetscMemcmp(a->val,b->val,(matsize)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqELLInvalidateDiagonal"
PetscErrorCode MatSeqELLInvalidateDiagonal(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  a->idiagvalid  = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}
