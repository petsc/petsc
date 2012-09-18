#define PETSCMAT_DLL

#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/impls/baij/seq/bstream/bstream.h"

extern PetscErrorCode MatDestroy_SeqBSTRM(Mat A);
extern PetscErrorCode MatSeqBSTRM_convert_bstrm(Mat A);
/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqBSTRM_4"
PetscErrorCode MatSolve_SeqBSTRM_4(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscErrorCode    ierr;
  PetscInt          i,j,n=a->mbs,*ai=a->i,*aj=a->j, *diag=a->diag,idx,jdx;
  PetscScalar       *x,s1,s2,s3,s4,x1,x2,x3,x4;
  PetscScalar       *v1, *v2, *v3, *v4;
  const PetscScalar *b;
  PetscInt slen;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  slen = 4*(ai[n]-ai[0]+diag[0]-diag[n]);

  v1  = bstrm->as;
  v2  = v1 + slen;
  v3  = v2 + slen;
  v4  = v3 + slen;

  /* forward solve the lower triangular */
  x[0] = b[0];
  x[1] = b[1];
  x[2] = b[2];
  x[3] = b[3];

  for (i=1; i<n; i++) {
    idx  = 4*i;
    s1 = b[idx  ];
    s2 = b[idx+1];
    s3 = b[idx+2];
    s4 = b[idx+3];
    for (j=ai[i]; j<ai[i+1]; j++) {
      jdx   = 4*aj[j];
      x1    = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];
      s1 -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3  + v1[3]*x4;
      s2 -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3  + v2[3]*x4;
      s3 -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3  + v3[3]*x4;
      s4 -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3  + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    x[idx  ] = s1;
    x[idx+1] = s2;
    x[idx+2] = s3;
    x[idx+3] = s4;
  }

  /* backward solve the upper triangular */
  for (i=n-1;i>=0;i--){
    idx  = 4*i;
    s1 = x[idx  ];
    s2 = x[idx+1];
    s3 = x[idx+2];
    s4 = x[idx+3];
    for (j=diag[i+1]+1; j<diag[i]; j++) {
      jdx = 4*aj[j];
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];
      s1 -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3  + v1[3]*x4;
      s2 -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3  + v2[3]*x4;
      s3 -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3  + v3[3]*x4;
      s4 -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3  + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    x[idx  ]  =  v1[0]*s1 + v1[1]*s2 + v1[2]*s3  + v1[3]*s4;
    x[idx+1]  =  v2[0]*s1 + v2[1]*s2 + v2[2]*s3  + v2[3]*s4;
    x[idx+2]  =  v3[0]*s1 + v3[1]*s2 + v3[2]*s3  + v3[3]*s4;
    x[idx+3]  =  v4[0]*s1 + v4[1]*s2 + v4[2]*s3  + v4[3]*s4;
    v1 += 4; v2 += 4; v3 += 4; v4 += 4;
  }

  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqBSTRM_5"
PetscErrorCode MatSolve_SeqBSTRM_5(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscErrorCode    ierr;
  PetscInt          i,j,n=a->mbs,*ai=a->i,*aj=a->j,*diag = a->diag,idx,jdx;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5;
  PetscScalar       *v1, *v2, *v3, *v4, *v5;
  const PetscScalar *b;
  PetscInt slen;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  slen = 5*(ai[n]-ai[0]+diag[0]-diag[n]);
  v1  = bstrm->as;
  v2  = v1 + slen;
  v3  = v2 + slen;
  v4  = v3 + slen;
  v5  = v4 + slen;


  /* forward solve the lower triangular */
  x[0] = b[0];
  x[1] = b[1];
  x[2] = b[2];
  x[3] = b[3];
  x[4] = b[4];

  for (i=1; i<n; i++) {
    idx  = 5*i;
    s1 = b[idx  ];
    s2 = b[idx+1];
    s3 = b[idx+2];
    s4 = b[idx+3];
    s5 = b[idx+4];
    for (j=ai[i]; j<ai[i+1]; j++) {
      jdx = 5*aj[j];
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx]; x5 = x[4+jdx];
      s1 -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      s2 -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      s3 -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      s4 -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4 + v4[4]*x5;
      s5 -= v5[0]*x1 + v5[1]*x2 + v5[2]*x3 + v5[3]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    x[idx  ] = s1;
    x[idx+1] = s2;
    x[idx+2] = s3;
    x[idx+3] = s4;
    x[idx+4] = s5;
  }

  /* backward solve the upper triangular */
  for (i=n-1;i>=0;i--){
    idx  = 5*i;
    s1 = x[idx  ];
    s2 = x[idx+1];
    s3 = x[idx+2];
    s4 = x[idx+3];
    s5 = x[idx+4];
    for (j=diag[i+1]+1; j<diag[i]; j++) {
      jdx = 5*aj[j];
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx]; x5 = x[4+jdx];
      s1 -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      s2 -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      s3 -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      s4 -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4 + v4[4]*x5;
      s5 -= v5[0]*x1 + v5[1]*x2 + v5[2]*x3 + v5[3]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    x[idx  ] = v1[0]*s1 + v1[1]*s2 + v1[2]*s3 + v1[3]*s4 + v1[4]*s5;
    x[idx+1] = v2[0]*s1 + v2[1]*s2 + v2[2]*s3 + v2[3]*s4 + v2[4]*s5;
    x[idx+2] = v3[0]*s1 + v3[1]*s2 + v3[2]*s3 + v3[3]*s4 + v3[4]*s5;
    x[idx+3] = v4[0]*s1 + v4[1]*s2 + v4[2]*s3 + v4[3]*s4 + v4[4]*s5;
    x[idx+4] = v5[0]*s1 + v5[1]*s2 + v5[2]*s3 + v5[3]*s4 + v5[4]*s5;
    v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
  }

  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_bstrm"
PetscErrorCode MatFactorGetSolverPackage_bstrm(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERBSTRM;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_bstrm"
PetscErrorCode MatLUFactorNumeric_bstrm(Mat F,Mat A,const MatFactorInfo *info)
{
  /* Mat_SeqBSTRM     *bstrm = (Mat_SeqBSTRM *) F->spptr; */
  PetscInt          bs = A->rmap->bs;
  PetscErrorCode ierr;
  Mat_SeqBSTRM  *bstrm;

  PetscFunctionBegin;
  /*ierr = (*bstrm ->MatLUFactorNumeric)(F,A,info);CHKERRQ(ierr); */
  switch (bs){
    case 4:
       ierr = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(F,A,info);CHKERRQ(ierr);
       break;
    case 5:
       ierr = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(F,A,info);CHKERRQ(ierr);
       /* ierr = MatLUFactorNumeric_SeqBAIJ_5(F,A,info);CHKERRQ(ierr); */
       break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }

  ierr = PetscNewLog(F,Mat_SeqBSTRM,&bstrm);CHKERRQ(ierr);
  F->spptr = (void *) bstrm;
  ierr = MatSeqBSTRM_convert_bstrm(F);CHKERRQ(ierr);
/*.........................................................
  F->ops->solve          = MatSolve_SeqBSTRM_5;
  .........................................................*/


  PetscFunctionReturn(0);
}
EXTERN_C_END
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatILUFactorSymbolic_bstrm"
PetscErrorCode MatILUFactorSymbolic_bstrm(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscInt ierr;
  PetscFunctionBegin;
  ierr = (MatILUFactorSymbolic_SeqBAIJ)(B,A,r,c,info);CHKERRQ(ierr);
  B->ops->lufactornumeric  = MatLUFactorNumeric_bstrm;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_bstrm"
PetscErrorCode MatLUFactorSymbolic_bstrm(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscInt ierr;
  PetscFunctionBegin;
  /* ierr = (*bstrm ->MatLUFactorSymbolic)(B,A,r,c,info);CHKERRQ(ierr); */
  ierr = (MatLUFactorSymbolic_SeqBAIJ)(B,A,r,c,info);CHKERRQ(ierr);
  B->ops->lufactornumeric  = MatLUFactorNumeric_bstrm;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqbaij_bstrm"
PetscErrorCode MatGetFactor_seqbaij_bstrm(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;
  Mat_SeqBSTRM   *bstrm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"matrix must be square");
  ierr = MatCreate(((PetscObject)A)->comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  /* ierr = MatSeqBAIJSetPreallocation(*B,bs,0,PETSC_NULL);CHKERRQ(ierr); */

  (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_bstrm;
  (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_bstrm;
  (*B)->ops->lufactornumeric   = MatLUFactorNumeric_bstrm;
  (*B)->ops->destroy           = MatDestroy_SeqBSTRM;
  (*B)->factortype             = ftype;
  (*B)->assembled              = PETSC_TRUE;  /* required by -ksp_view */
  (*B)->preallocated           = PETSC_TRUE;
  ierr = PetscNewLog(*B,Mat_SeqBSTRM,&bstrm);CHKERRQ(ierr);
  (*B)->spptr = (void *) bstrm;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)*B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_bstrm",MatFactorGetSolverPackage_bstrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
