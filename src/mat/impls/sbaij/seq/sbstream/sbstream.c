#define PETSCMAT_DLL

#include "../src/mat/impls/sbaij/seq/sbaij.h"
#include "../src/mat/impls/sbaij/seq/sbstream/sbstream.h"

#if 0
extern   PetscErrorCode MatFactorGetSolverPackage_seqsbaij_sbstrm(Mat, const MatSolverPackage *);
extern   PetscErrorCode MatICCFactorSymbolic_sbstrm(Mat,Mat, IS, const MatFactorInfo *);
extern   PetscErrorCode MatCholeskyFactorSymbolic_sbstrm(Mat, Mat, IS, const MatFactorInfo *);
extern   PetscErrorCode MatCholeskyFactorNumeric_sbstrm (Mat, Mat, const MatFactorInfo *);
#endif

extern   PetscErrorCode  MatAssemblyEnd_SeqSBAIJ(Mat,MatAssemblyType);


#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqSBSTRM"
PetscErrorCode MatDestroy_SeqSBSTRM(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqSBSTRM       *sbstrm = (Mat_SeqSBSTRM *) A->spptr;

  if (sbstrm) {
    ierr = PetscFree3(sbstrm->as, sbstrm->asi, sbstrm->asj);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatDestroy_SeqSBAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/
PetscErrorCode MatDuplicate_SeqSBSTRM(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot duplicate STRM matrices yet");
  PetscFunctionReturn(0);
}
/*=========================================================*/

#undef __FUNCT__
#define __FUNCT__ "SeqSBSTRM_convert_sbstrm"
PetscErrorCode SeqSBSTRM_convert_sbstrm(Mat A)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ *) A->data;
  Mat_SeqSBSTRM   *sbstrm = (Mat_SeqSBSTRM*) A->spptr;
  PetscInt       m = a->mbs, bs = A->rmap->bs;
  PetscInt       *ai = a->i;
  PetscInt       i,j,ib,jb;
  MatScalar      *aa = a->a;
  PetscErrorCode ierr;
  PetscInt  bs2, rbs,  cbs, blen, slen;
  PetscScalar **asp;

  PetscFunctionBegin;
  sbstrm->rbs = bs;
  sbstrm->cbs = bs;


  rbs = cbs = bs;
  bs2 = rbs*cbs;
  blen = ai[m]-ai[0];
  slen = blen*cbs;

  ierr = PetscFree(sbstrm->as);CHKERRQ(ierr);
  ierr = PetscMalloc(bs2*blen*sizeof(MatScalar), &sbstrm->as);CHKERRQ(ierr);

  ierr  = PetscMalloc(rbs*sizeof(MatScalar *), &asp);CHKERRQ(ierr);

  for (i=0;i<rbs;i++) asp[i] = sbstrm->as + i*slen;

  for (j=0;j<blen;j++) {
     for (jb=0; jb<cbs; jb++){
     for (ib=0; ib<rbs; ib++){
         asp[ib][j*cbs+jb] = aa[j*bs2+jb*rbs+ib];
     }}
  }

  ierr = PetscFree(asp);CHKERRQ(ierr);
/*
  switch (bs){
    case 4:
       A->ops->solve   = MatSolve_SeqSBSTRM_4;
       break;
    case 5:
       A->ops->solve   = MatSolve_SeqSBSTRM_5;
       break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
*/
  PetscFunctionReturn(0);
}
/*=========================================================*/
extern PetscErrorCode SeqSBSTRM_create_sbstrm(Mat);
/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqSBSTRM"
PetscErrorCode MatAssemblyEnd_SeqSBSTRM(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatAssemblyEnd_SeqSBAIJ(A, mode);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  ierr = SeqSBSTRM_create_sbstrm(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqSBAIJ_SeqSBSTRM"
PetscErrorCode MatConvert_SeqSBAIJ_SeqSBSTRM(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;

  Mat            B = *newmat;
  Mat_SeqSBSTRM   *sbstrm;
  /* PetscInt       bs = A->rmap->bs; */

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }


  ierr = PetscNewLog(B,Mat_SeqSBSTRM,&sbstrm);CHKERRQ(ierr);
  B->spptr = (void *) sbstrm;

  /* Set function pointers for methods that we inherit from BAIJ but override. */
  B->ops->duplicate        = MatDuplicate_SeqSBSTRM;
  B->ops->assemblyend      = MatAssemblyEnd_SeqSBSTRM;
  B->ops->destroy          = MatDestroy_SeqSBSTRM;
  /*B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_sbstrm;
    B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_sbstrm; */

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
      ierr = SeqSBSTRM_create_sbstrm(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSBSTRM);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqSBSTRM"
PetscErrorCode MatCreateSeqSBSTRM(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
    ierr = MatCreate(comm,A);CHKERRQ(ierr);
    ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
    ierr = MatSetType(*A,MATSEQSBSTRM);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
    (*A)->rmap->bs = bs;
  PetscFunctionReturn(0);
}
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqSBSTRM"
PetscErrorCode  MatCreate_SeqSBSTRM(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqSBAIJ_SeqSBSTRM(A,MATSEQSBSTRM,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatConvert_seqisbaij_seqsbstrm_C",
                                     "MatConvert_SeqSBAIJ_SeqSBSTRM",
                                      MatConvert_SeqSBAIJ_SeqSBSTRM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
/*=========================================================*/
/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqSBSTRM_4"
PetscErrorCode MatMult_SeqSBSTRM_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  PetscScalar       zero = 0.0;
  PetscScalar       sum1,sum2,sum3,sum4,x1,x2,x3,x4, xr1,xr2,xr3,xr4;
  PetscScalar       *x, *xb, *z;
  MatScalar         *v1, *v2, *v3, *v4;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;
  PetscInt slen;

  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  slen = 4*(ai[mbs]-ai[0]);
  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;

  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i];
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
    sum1 = z[4*i]; sum2 = z[4*i+1]; sum3 = z[4*i+2]; sum4 = z[4*i+3];
    nonzerorow += (n>0);
    jmin = 0;
    ib = aj + ai[i];
    if (*ib == i){     /* (diag of A)*x */
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4;
      sum2 += v1[1]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4;
      sum3 += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v3[3]*x4;
      sum4 += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
      jmin++;
    }

    for (j=jmin; j<n; j++) {
      cval       = ib[j]*4;
      z[cval]     += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4;
      z[cval+1]   += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4;
      z[cval+2]   += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4;
      z[cval+3]   += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;

      xr1 = x[cval]; xr2 = x[cval+1]; xr3 = x[cval+2]; xr4 = x[cval+3];
      sum1 += v1[0]*xr1 + v1[1]*xr2 + v1[2]*xr3  + v1[3]*xr4;
      sum2 += v2[0]*xr1 + v2[1]*xr2 + v2[2]*xr3  + v2[3]*xr4;
      sum3 += v3[0]*xr1 + v3[1]*xr2 + v3[2]*xr3  + v3[3]*xr4;
      sum4 += v4[0]*xr1 + v4[1]*xr2 + v4[2]*xr3  + v4[3]*xr4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    z[4*i]   = sum1;
    z[4*i+1] = sum2;
    z[4*i+2] = sum3;
    z[4*i+3] = sum4;
    xb +=4;

  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(32.0*(a->nz*2.0 - nonzerorow) - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqSBSTRM_5"
PetscErrorCode MatMult_SeqSBSTRM_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  PetscScalar       zero = 0.0;
  PetscScalar       *z = 0;
  PetscScalar       *x,*xb;
  const MatScalar   *v1, *v2, *v3, *v4, *v5;
  PetscScalar       x1, x2, x3, x4, x5;
  PetscScalar       xr1, xr2, xr3, xr4, xr5;
  PetscScalar       sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;
  PetscInt slen;


  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  slen = 5*(ai[mbs]-ai[0]);

  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  v5 = v4 + slen;

  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i];
    nonzerorow += (n>0);

    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
    sum1 = z[5*i];
    sum2 = z[5*i+1];
    sum3 = z[5*i+2];
    sum4 = z[5*i+3];
    sum5 = z[5*i+4];
    jmin = 0;
    ib = aj + ai[i];
    if (*ib == i){     /* (diag of A)*x */
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      sum2 += v1[1]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      sum3 += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      sum4 += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4 + v4[4]*x5;
      sum5 += v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
      jmin++;
    }

    PetscPrefetchBlock(ib+jmin+n,n,0,PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v1+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v2+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v3+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v4+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v5+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */

    for (j=jmin; j<n; j++) {
      cval       = ib[j]*5;
      z[cval]   += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4 + v5[0]*x5;
      z[cval+1] += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4 + v5[1]*x5;
      z[cval+2] += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4 + v5[2]*x5;
      z[cval+3] += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4 + v5[3]*x5;
      z[cval+4] += v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4 + v5[4]*x5;

      xr1 = x[cval]; xr2 = x[cval+1]; xr3 = x[cval+2]; xr4 = x[cval+3]; xr5 = x[cval+4];
      sum1 += v1[0]*xr1 + v1[1]*xr2 + v1[2]*xr3 + v1[3]*xr4 + v1[4]*xr5;
      sum2 += v2[0]*xr1 + v2[1]*xr2 + v2[2]*xr3 + v2[3]*xr4 + v2[4]*xr5;
      sum3 += v3[0]*xr1 + v3[1]*xr2 + v3[2]*xr3 + v3[3]*xr4 + v3[4]*xr5;
      sum4 += v4[0]*xr1 + v4[1]*xr2 + v4[2]*xr3 + v4[3]*xr4 + v4[4]*xr5;
      sum5 += v5[0]*xr1 + v5[1]*xr2 + v5[2]*xr3 + v5[3]*xr4 + v5[4]*xr5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    z[5*i]   = sum1; z[5*i+1] = sum2; z[5*i+2] = sum3; z[5*i+3] = sum4; z[5*i+4] = sum5;
    xb += 5;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(50.0*(a->nz*2.0 - nonzerorow) - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqSBSTRM_4"
PetscErrorCode MatMultAdd_SeqSBSTRM_4(Mat A,Vec xx,Vec yy,Vec zz)
{

  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  PetscScalar       sum1,sum2,sum3,sum4,x1,x2,x3,x4, xr1,xr2,xr3,xr4;
  PetscScalar       *x,*z, *xb;
  MatScalar         *v1, *v2, *v3, *v4;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;
  PetscInt slen;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  slen = 4*(ai[mbs]-ai[0]);
  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;

  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i];
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];  xb += 4;
    sum1 = z[4*i];
    sum2 = z[4*i + 1];
    sum3 = z[4*i + 2];
    sum4 = z[4*i + 3];
    nonzerorow += (n>0);
    jmin = 0;
    ib = aj + ai[i];
    if (*ib == i){     /* (diag of A)*x */
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4;
      sum2 += v1[1]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4;
      sum3 += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v3[3]*x4;
      sum4 += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
      jmin++;
    }

    for (j=jmin; j<n; j++) {
      cval       = ib[j]*4;
      z[cval]     += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4;
      z[cval+1]   += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4;
      z[cval+2]   += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4;
      z[cval+3]   += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;

      xr1 = x[cval]; xr2 = x[cval+1]; xr3 = x[cval+2]; xr4 = x[cval+3];
      sum1 += v1[0]*xr1 + v1[1]*xr2 + v1[2]*xr3  + v1[3]*xr4;
      sum2 += v2[0]*xr1 + v2[1]*xr2 + v2[2]*xr3  + v2[3]*xr4;
      sum3 += v3[0]*xr1 + v3[1]*xr2 + v3[2]*xr3  + v3[3]*xr4;
      sum4 += v4[0]*xr1 + v4[1]*xr2 + v4[2]*xr3  + v4[3]*xr4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    z[4*i]   = sum1;
    z[4*i+1] = sum2;
    z[4*i+2] = sum3;
    z[4*i+3] = sum4;

  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
   ierr = PetscLogFlops(32.0*(a->nz*2.0 - nonzerorow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqSBSTRM_5"
PetscErrorCode MatMultAdd_SeqSBSTRM_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  PetscScalar       *x,*xb, *z;
  MatScalar         *v1, *v2, *v3, *v4, *v5;
  PetscScalar       x1, x2, x3, x4, x5;
  PetscScalar       xr1, xr2, xr3, xr4, xr5;
  PetscScalar       sum1, sum2, sum3, sum4, sum5;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;
  PetscInt slen;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);


  slen = 5*(ai[mbs]-ai[0]);
  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  v5 = v4 + slen;

  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i];
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; xb += 5;
    sum1 = z[5*i];
    sum2 = z[5*i+1];
    sum3 = z[5*i+2];
    sum4 = z[5*i+3];
    sum5 = z[5*i+4];
    nonzerorow += (n>0);
    jmin = 0;
    ib = aj + ai[i];
    if (*ib == i){     /* (diag of A)*x, only upper triangular part is used */
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      sum2 += v1[1]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      sum3 += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      sum4 += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4 + v4[4]*x5;
      sum5 += v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
      jmin++;
    }

    for (j=jmin; j<n; j++) {
      cval       = ib[j]*5;
      z[cval]   += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4 + v5[0]*x5;
      z[cval+1] += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4 + v5[1]*x5;
      z[cval+2] += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4 + v5[2]*x5;
      z[cval+3] += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4 + v5[3]*x5;
      z[cval+4] += v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4 + v5[4]*x5;

      xr1 = x[cval]; xr2 = x[cval+1]; xr3 = x[cval+2]; xr4 = x[cval+3]; xr5 = x[cval+4];
      sum1 += v1[0]*xr1 + v1[1]*xr2 + v1[2]*xr3 + v1[3]*xr4 + v1[4]*xr5;
      sum2 += v2[0]*xr1 + v2[1]*xr2 + v2[2]*xr3 + v2[3]*xr4 + v2[4]*xr5;
      sum3 += v3[0]*xr1 + v3[1]*xr2 + v3[2]*xr3 + v3[3]*xr4 + v3[4]*xr5;
      sum4 += v4[0]*xr1 + v4[1]*xr2 + v4[2]*xr3 + v4[3]*xr4 + v4[4]*xr5;
      sum5 += v5[0]*xr1 + v5[1]*xr2 + v5[2]*xr3 + v5[3]*xr4 + v5[4]*xr5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    z[5*i]   = sum1;
    z[5*i+1] = sum2;
    z[5*i+2] = sum3;
    z[5*i+3] = sum4;
    z[5*i+4] = sum5;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(50.0*(a->nz*2.0 - nonzerorow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "SeqSBSTRM_create_sbstrm"
PetscErrorCode SeqSBSTRM_create_sbstrm(Mat A)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ *) A->data;
  Mat_SeqSBSTRM   *sbstrm = (Mat_SeqSBSTRM*) A->spptr;
  PetscInt       MROW = a->mbs, bs = A->rmap->bs;
  PetscInt       *ai = a->i;
  PetscInt       i,j,k;
  MatScalar      *aa = a->a;
  PetscErrorCode ierr;
  PetscInt  bs2, rbs,  cbs, blen, slen;
  PetscScalar **asp;

  PetscFunctionBegin;
  sbstrm->rbs = sbstrm->cbs = bs;

  rbs = cbs = bs;
  bs2 = rbs*cbs;
  blen = ai[MROW]-ai[0];
  slen = blen*cbs;

  ierr  = PetscMalloc(bs2*blen*sizeof(PetscScalar), &sbstrm->as);CHKERRQ(ierr);

  ierr  = PetscMalloc(rbs*sizeof(PetscScalar *), &asp);CHKERRQ(ierr);

  for (i=0;i<rbs;i++) asp[i] = sbstrm->as + i*slen;

  for (k=0; k<blen; k++) {
    for (j=0; j<cbs; j++)
    for (i=0; i<rbs; i++)
        asp[i][k*cbs+j] = aa[k*bs2+j*rbs+i];
  }

  /* ierr = PetscFree(a->a);CHKERRQ(ierr);  */

  switch (bs){
    case 4:
       A->ops->mult          = MatMult_SeqSBSTRM_4;
       A->ops->multadd       = MatMultAdd_SeqSBSTRM_4;
       /** A->ops->sor     = MatSOR_SeqSBSTRM_4;  **/
       break;
    case 5:
       A->ops->mult          = MatMult_SeqSBSTRM_5;
       A->ops->multadd       = MatMultAdd_SeqSBSTRM_5;
       /** A->ops->sor     = MatSOR_SeqSBSTRM_5;  **/
       break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
  PetscFunctionReturn(0);
}
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatSeqSBSTRMTransform"
PetscErrorCode MatSeqSBSTRMTransform(Mat A)
{
  PetscFunctionBegin;
    SeqSBSTRM_convert_sbstrm(A);
  PetscFunctionReturn(0);
}
EXTERN_C_END

