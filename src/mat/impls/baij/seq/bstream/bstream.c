#define PETSCMAT_DLL

#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/impls/baij/seq/bstream/bstream.h"

/*=========================================================*/ 
PetscErrorCode MatLUFactorSymbolic_bstrm(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info);
extern PetscErrorCode MatLUFactorNumeric_bstrm(Mat F,Mat A,const MatFactorInfo *info);
/*=========================================================*/ 
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqBSTRM"
PetscErrorCode MatDestroy_SeqBSTRM(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqBSTRM       *bstrm = (Mat_SeqBSTRM *) A->spptr;

  A->ops->lufactorsymbolic = bstrm->MatLUFactorSymbolic;
  A->ops->lufactornumeric  = bstrm->MatLUFactorNumeric;
  A->ops->assemblyend      = bstrm->AssemblyEnd;
  A->ops->destroy          = bstrm->MatDestroy;
  A->ops->duplicate        = bstrm->MatDuplicate;

  ierr = PetscFree(bstrm->as);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQBAIJ);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
PetscErrorCode MatDuplicate_SeqBSTRM(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode ierr;
  Mat_SeqBSTRM        *bstrm = (Mat_SeqBSTRM *) A->spptr;

  PetscFunctionBegin;
  ierr = (*bstrm->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot duplicate STRM matrices yet");    
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBSTRM_4"
PetscErrorCode MatSolve_SeqBSTRM_4(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  PetscInt          i,j,n=a->mbs,*vi,*ai=a->i,*aj=a->j, *diag=a->diag, nz,idx,jdx;     
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,x1,x2,x3,x4,*t;
  PetscScalar       *v1, *v2, *v3, *v4;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  PetscInt *asi = bstrm->asi, *asj = bstrm->asj; 
  PetscInt slen = 4*(ai[n]-ai[0]+diag[0]-diag[n]);

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
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  PetscInt          i,j,n=a->mbs,*vi,*ai=a->i,*aj=a->j,*diag = a->diag, nz,idx,jdx;     
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5,*t;
  PetscScalar       *v1, *v2, *v3, *v4, *v5;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

 

  PetscInt slen = 5*(ai[n]-ai[0]+diag[0]-diag[n]);
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

#undef __FUNCT__
#define __FUNCT__ "SeqBSTRM_convert_bstrm"
PetscErrorCode SeqBSTRM_convert_bstrm(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *) A->data;
  Mat_SeqBSTRM   *bstrm = (Mat_SeqBSTRM*) A->spptr;
  PetscInt       m = a->mbs, bs = A->rmap->bs;
  PetscInt       *aj = a->j, *ai = a->i, *diag = a->diag;
  PetscInt       i,j,k,ib,jb, rmax = a->rmax, *idiag, nnz;
  PetscInt       *asi, *asj;
  MatScalar      *aa = a->a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInt  bs2, rbs,  cbs, blen, slen;
  bstrm->rbs = bs;
  bstrm->cbs = bs;


  rbs = cbs = bs;
  bs2 = rbs*cbs; 
  blen = ai[m]-ai[0]+diag[0] - diag[m];
  slen = blen*cbs;

  ierr = PetscFree(bstrm->as);CHKERRQ(ierr);
  ierr = PetscMalloc(bs2*blen*sizeof(MatScalar), &bstrm->as);CHKERRQ(ierr);

  PetscScalar **asp;
  ierr  = PetscMalloc(rbs*sizeof(MatScalar *), &asp);CHKERRQ(ierr);
   
  for(i=0;i<rbs;i++) asp[i] = bstrm->as + i*slen; 

  for(j=0;j<blen;j++) {
     for (jb=0; jb<cbs; jb++){ 
     for (ib=0; ib<rbs; ib++){ 
         asp[ib][j*cbs+jb] = aa[j*bs2+jb*rbs+ib];
     }}
  }

  ierr = PetscFree(asp);CHKERRQ(ierr);
  switch (bs){
    case 4:
       A->ops->solve   = MatSolve_SeqBSTRM_4;
       break; 
    case 5:
       A->ops->solve   = MatSolve_SeqBSTRM_5;
       break;  
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqBSTRM"
PetscErrorCode MatAssemblyEnd_SeqBSTRM(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqBSTRM    *bstrm = (Mat_SeqBSTRM *) A->spptr;
  Mat_SeqBAIJ     *a    = (Mat_SeqBAIJ *)  A->data;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  (*bstrm->AssemblyEnd)(A, mode);

  ierr = SeqBSTRM_create_bstrm(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqBAIJ_SeqBSTRM"
PetscErrorCode  MatConvert_SeqBAIJ_SeqBSTRM(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqBSTRM   *bstrm;
  PetscInt       bs = A->rmap->bs;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  

  ierr = PetscNewLog(B,Mat_SeqBSTRM,&bstrm);CHKERRQ(ierr);
  B->spptr = (void *) bstrm;

/*.........................................................
  bstrm->MatLUFactorNumeric  = A->ops->lufactornumeric; 
  bstrm->MatLUFactorSymbolic = A->ops->lufactorsymbolic; 
  bstrm->MatLUFactorSymbolic = MatLUFactorSymbolic_SeqBAIJ;
  bstrm->MatLUFactorNumeric  = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
  .........................................................*/ 
  bstrm->AssemblyEnd         = A->ops->assemblyend;
  bstrm->MatDestroy          = A->ops->destroy;
  bstrm->MatDuplicate        = A->ops->duplicate;

  /* Set function pointers for methods that we inherit from BAIJ but override. */
  B->ops->duplicate        = MatDuplicate_SeqBSTRM;
  B->ops->assemblyend      = MatAssemblyEnd_SeqBSTRM;
  B->ops->destroy          = MatDestroy_SeqBSTRM;
  /*B->ops->lufactorsymbolic = MatLUFactorSymbolic_bstrm;
    B->ops->lufactornumeric  = MatLUFactorNumeric_bstrm; */

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
      ierr = SeqBSTRM_create_bstrm(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQBSTRM);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*=========================================================*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqBSTRM"
PetscErrorCode MatCreateSeqBSTRM(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
    ierr = MatCreate(comm,A);CHKERRQ(ierr);
    ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
    ierr = MatSetType(*A,MATSEQBSTRM);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
    (*A)->rmap->bs = bs;
  PetscFunctionReturn(0);
}
/*=========================================================*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqBSTRM"
PetscErrorCode MatCreate_SeqBSTRM(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqBSTRM(A,MATSEQBSTRM,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatConvert_seqbaij_seqbstrm_C",
                                     "MatConvert_SeqBAIJ_SeqBSTRM",
                                      MatConvert_SeqBAIJ_SeqBSTRM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
/*=========================================================*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_bstrm"
PetscErrorCode MatFactorGetSolverPackage_bstrm(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_BSTRM;    
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
  
  Mat_SeqBSTRM  *bstrm;
  ierr = PetscNewLog(F,Mat_SeqBSTRM,&bstrm);CHKERRQ(ierr);
  F->spptr = (void *) bstrm;  
  ierr = SeqBSTRM_convert_bstrm(F);CHKERRQ(ierr);
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
  Mat_SeqBSTRM     *bstrm = (Mat_SeqBSTRM *) B->spptr;
  PetscFunctionBegin;
  PetscInt ierr;
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
  Mat_SeqBSTRM     *bstrm = (Mat_SeqBSTRM *) B->spptr;
  PetscFunctionBegin;
  PetscInt ierr;
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
  PetscInt       n = A->rmap->n, bs = A->rmap->bs;
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
/*=========================================================*/ 

/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBSTRM_4"
PetscErrorCode MatSOR_SeqBSTRM_4(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,s1,s2,s3,s4;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  MatScalar         *v1,*v2,*v3,*v4, *v10,*v20,*v30,*v40, vvs1,vvs2,vvs3,vvs4;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  PetscInt slen = 4*(ai[m]-ai[0]);
  v10 = bstrm->as;
  v20 = v10 + slen;
  v30 = v20 + slen;
  v40 = v30 + slen;

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[4] + b[2]*idiag[8]  + b[3]*idiag[12];
      x[1] = b[0]*idiag[1] + b[1]*idiag[5] + b[2]*idiag[9]  + b[3]*idiag[13];
      x[2] = b[0]*idiag[2] + b[1]*idiag[6] + b[2]*idiag[10] + b[3]*idiag[14];
      x[3] = b[0]*idiag[3] + b[1]*idiag[7] + b[2]*idiag[11] + b[3]*idiag[15];
      i2     = 4;
      idiag += 16;
      for (i=1; i<m; i++) {
	v1     = v10 + 4*ai[i];
	v2     = v20 + 4*ai[i];
	v3     = v30 + 4*ai[i];
	v4     = v40 + 4*ai[i];
	vi    = aj + ai[i];
	nz    = diag[i] - ai[i];
	s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3];
	while (nz--) {
	  idx  = 4*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; 
	  s1  -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4;
	  s2  -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4;
	  s3  -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4;
	  s4  -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4;
	  v1 += 4; v2 += 4; v3 += 4; v4 += 4; 
	}
	nz    = ai[i+1] - diag[i];
	while (nz--) {
	  vvs1 = v1[0]; vvs1 = v1[1]; vvs1 = v1[2]*x3; vvs1 = v1[3]; 
	  vvs2 = v2[0]; vvs2 = v2[1]; vvs2 = v2[2]*x3; vvs2 = v2[3]; 
	  vvs3 = v3[0]; vvs3 = v3[1]; vvs3 = v3[2]*x3; vvs3 = v3[3]; 
	  vvs4 = v4[0]; vvs4 = v4[1]; vvs4 = v4[2]*x3; vvs4 = v4[3]; 
	  v1 += 4; v2 += 4; v3 += 4; v4 += 4; 
	}
        x[i2]   = idiag[0]*s1 + idiag[4]*s2 + idiag[8]*s3  + idiag[12]*s4;
        x[i2+1] = idiag[1]*s1 + idiag[5]*s2 + idiag[9]*s3  + idiag[13]*s4;
        x[i2+2] = idiag[2]*s1 + idiag[6]*s2 + idiag[10]*s3 + idiag[14]*s4;
        x[i2+3] = idiag[3]*s1 + idiag[7]*s2 + idiag[11]*s3 + idiag[15]*s4;
        idiag   += 16;
        i2      += 4;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(16.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+16*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3];
        x[i2]   = mdiag[0]*x1 + mdiag[4]*x2 + mdiag[8]*x3  + mdiag[12]*x4;
        x[i2+1] = mdiag[1]*x1 + mdiag[5]*x2 + mdiag[9]*x3  + mdiag[13]*x4;
        x[i2+2] = mdiag[2]*x1 + mdiag[6]*x2 + mdiag[10]*x3 + mdiag[14]*x4;
        x[i2+3] = mdiag[3]*x1 + mdiag[7]*x2 + mdiag[11]*x3 + mdiag[15]*x4;
        mdiag  += 16;
        i2     += 4;
      }
      ierr = PetscLogFlops(28.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+16*a->mbs - 16;
      i2      = 4*m - 4;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3];
      x[i2]   = idiag[0]*x1 + idiag[4]*x2 + idiag[8]*x3  + idiag[12]*x4;
      x[i2+1] = idiag[1]*x1 + idiag[5]*x2 + idiag[9]*x3  + idiag[13]*x4;
      x[i2+2] = idiag[2]*x1 + idiag[6]*x2 + idiag[10]*x3 + idiag[14]*x4;
      x[i2+3] = idiag[3]*x1 + idiag[7]*x2 + idiag[11]*x3 + idiag[15]*x4;
      idiag -= 16;
      i2    -= 4;
      for (i=m-2; i>=0; i--) {
	v1    = v10 + 4*(ai[i+1]-1);
	v2    = v20 + 4*(ai[i+1]-1);
	v3    = v30 + 4*(ai[i+1]-1);
	v4    = v40 + 4*(ai[i+1]-1);
	vi    = aj + ai[i+1] - 1;
	nz    = ai[i+1] - diag[i] - 1;
	s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3];
	while (nz--) {
	  idx  = 4*(*vi--);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; 
	  s1  -= v1[3]*x4 + v1[2]*x3 + v1[1]*x2 + v1[0]*x1;
	  s2  -= v2[3]*x4 + v2[2]*x3 + v2[1]*x2 + v2[0]*x1;
	  s3  -= v3[3]*x4 + v3[2]*x3 + v3[1]*x2 + v3[0]*x1;
	  s4  -= v4[3]*x4 + v4[2]*x3 + v4[1]*x2 + v4[0]*x1;
	  v1 -= 4; v2 -= 4; v3 -= 4; v4 -= 4; 
	}
	nz    =  diag[i] - ai[i];
	while (nz--) {
	  vvs1 = v1[3]; vvs1 = v1[2]*x3; vvs1 = v1[1]; vvs1 = v1[0];
	  vvs2 = v2[3]; vvs2 = v2[2]*x3; vvs2 = v2[1]; vvs2 = v2[0];
	  vvs3 = v3[3]; vvs3 = v3[2]*x3; vvs3 = v3[1]; vvs3 = v3[0];
	  vvs4 = v4[3]; vvs4 = v4[2]*x3; vvs4 = v4[1]; vvs4 = v4[0];
	  v1 -= 4; v2 -= 4; v3 -= 4; v4 -= 4;
	}
        x[i2]   = idiag[0]*s1 + idiag[4]*s2 + idiag[8]*s3  + idiag[12]*s4;
        x[i2+1] = idiag[1]*s1 + idiag[5]*s2 + idiag[9]*s3  + idiag[13]*s4;
        x[i2+2] = idiag[2]*s1 + idiag[6]*s2 + idiag[10]*s3 + idiag[14]*s4;
        x[i2+3] = idiag[3]*s1 + idiag[7]*s2 + idiag[11]*s3 + idiag[15]*s4;
        idiag   -= 16;
        i2      -= 4;
      }
      ierr = PetscLogFlops(16.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
/*=========================================================*/ 
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBSTRM_5"
PetscErrorCode MatSOR_SeqBSTRM_5(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,x5,s1,s2,s3,s4,s5;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  MatScalar         *v1, *v2, *v3, *v4, *v5, *v10, *v20, *v30, *v40, *v50, vvs1, vvs2,vvs3,vvs4,vvs5;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  PetscInt slen = 5*(ai[m]-ai[0]);
  v10 = bstrm->as;
  v20 = v10 + slen;
  v30 = v20 + slen;
  v40 = v30 + slen;
  v50 = v40 + slen;

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[5] + b[2]*idiag[10] + b[3]*idiag[15] + b[4]*idiag[20];
      x[1] = b[0]*idiag[1] + b[1]*idiag[6] + b[2]*idiag[11] + b[3]*idiag[16] + b[4]*idiag[21];
      x[2] = b[0]*idiag[2] + b[1]*idiag[7] + b[2]*idiag[12] + b[3]*idiag[17] + b[4]*idiag[22];
      x[3] = b[0]*idiag[3] + b[1]*idiag[8] + b[2]*idiag[13] + b[3]*idiag[18] + b[4]*idiag[23];
      x[4] = b[0]*idiag[4] + b[1]*idiag[9] + b[2]*idiag[14] + b[3]*idiag[19] + b[4]*idiag[24];
      i2     = 5;
      idiag += 25;
      for (i=1; i<m; i++) {
	v1     = v10 + 5*ai[i];
	v2     = v20 + 5*ai[i];
	v3     = v30 + 5*ai[i];
	v4     = v40 + 5*ai[i];
	v5     = v50 + 5*ai[i];
	vi    = aj + ai[i];
	nz    = diag[i] - ai[i];
	s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3]; s5 = b[i2+4];
	while (nz--) {
	  idx  = 5*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
	  s1  -= v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
	  s2  -= v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
	  s3  -= v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
	  s4  -= v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4 + v4[4]*x5;
	  s5  -= v5[0]*x1 + v5[1]*x2 + v5[2]*x3 + v5[3]*x4 + v5[4]*x5;
	  v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
	}
	nz    = ai[i+1] - diag[i];
	while (nz--) {
	  vvs1 = v1[0]; vvs1 = v1[1]; vvs1 = v1[2]*x3; vvs1 = v1[3]; vvs1 = v1[4];
	  vvs2 = v2[0]; vvs2 = v2[1]; vvs2 = v2[2]*x3; vvs2 = v2[3]; vvs2 = v2[4];
	  vvs3 = v3[0]; vvs3 = v3[1]; vvs3 = v3[2]*x3; vvs3 = v3[3]; vvs3 = v3[4];
	  vvs4 = v4[0]; vvs4 = v4[1]; vvs4 = v4[2]*x3; vvs4 = v4[3]; vvs4 = v4[4];
	  vvs5 = v5[0]; vvs5 = v5[1]; vvs5 = v5[2]*x3; vvs5 = v5[3]; vvs5 = v5[4];
	  v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
	}
	x[i2]   = idiag[0]*s1 + idiag[5]*s2 + idiag[10]*s3 + idiag[15]*s4 + idiag[20]*s5;
	x[i2+1] = idiag[1]*s1 + idiag[6]*s2 + idiag[11]*s3 + idiag[16]*s4 + idiag[21]*s5;
	x[i2+2] = idiag[2]*s1 + idiag[7]*s2 + idiag[12]*s3 + idiag[17]*s4 + idiag[22]*s5;
	x[i2+3] = idiag[3]*s1 + idiag[8]*s2 + idiag[13]*s3 + idiag[18]*s4 + idiag[23]*s5;
	x[i2+4] = idiag[4]*s1 + idiag[9]*s2 + idiag[14]*s3 + idiag[19]*s4 + idiag[24]*s5;
        idiag   += 25;
        i2      += 5;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(25.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+25*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4];
        x[i2]   = mdiag[0]*x1 + mdiag[5]*x2 + mdiag[10]*x3 + mdiag[15]*x4 + mdiag[20]*x5;
        x[i2+1] = mdiag[1]*x1 + mdiag[6]*x2 + mdiag[11]*x3 + mdiag[16]*x4 + mdiag[21]*x5;
        x[i2+2] = mdiag[2]*x1 + mdiag[7]*x2 + mdiag[12]*x3 + mdiag[17]*x4 + mdiag[22]*x5;
        x[i2+3] = mdiag[3]*x1 + mdiag[8]*x2 + mdiag[13]*x3 + mdiag[18]*x4 + mdiag[23]*x5;
        x[i2+4] = mdiag[4]*x1 + mdiag[9]*x2 + mdiag[14]*x3 + mdiag[19]*x4 + mdiag[24]*x5;
        mdiag  += 25;
        i2     += 5;
      }
      ierr = PetscLogFlops(45.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+25*a->mbs - 25;
      i2      = 5*m - 5;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4];
      x[i2]   = idiag[0]*x1 + idiag[5]*x2 + idiag[10]*x3 + idiag[15]*x4 + idiag[20]*x5;
      x[i2+1] = idiag[1]*x1 + idiag[6]*x2 + idiag[11]*x3 + idiag[16]*x4 + idiag[21]*x5;
      x[i2+2] = idiag[2]*x1 + idiag[7]*x2 + idiag[12]*x3 + idiag[17]*x4 + idiag[22]*x5;
      x[i2+3] = idiag[3]*x1 + idiag[8]*x2 + idiag[13]*x3 + idiag[18]*x4 + idiag[23]*x5;
      x[i2+4] = idiag[4]*x1 + idiag[9]*x2 + idiag[14]*x3 + idiag[19]*x4 + idiag[24]*x5;
      idiag -= 25;
      i2    -= 5;
      for (i=m-2; i>=0; i--) {
	v1    = v10 + 5*(ai[i+1]-1);
	v2    = v20 + 5*(ai[i+1]-1);
	v3    = v30 + 5*(ai[i+1]-1);
	v4    = v40 + 5*(ai[i+1]-1);
	v5    = v50 + 5*(ai[i+1]-1);
	vi    = aj + ai[i+1] - 1;
	nz    = ai[i+1] - diag[i] - 1;
	s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3]; s5 = x[i2+4];
	while (nz--) {
	  idx  = 5*(*vi--);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
	  s1  -= v1[4]*x5 + v1[3]*x4 + v1[2]*x3 + v1[1]*x2 + v1[0]*x1;
	  s2  -= v2[4]*x5 + v2[3]*x4 + v2[2]*x3 + v2[1]*x2 + v2[0]*x1;
	  s3  -= v3[4]*x5 + v3[3]*x4 + v3[2]*x3 + v3[1]*x2 + v3[0]*x1;
	  s4  -= v4[4]*x5 + v4[3]*x4 + v4[2]*x3 + v4[1]*x2 + v4[0]*x1;
	  s5  -= v5[4]*x5 + v5[3]*x4 + v5[2]*x3 + v5[1]*x2 + v5[0]*x1;
	  v1 -= 5; v2 -= 5; v3 -= 5; v4 -= 5; v5 -= 5;
	}
	nz    =  diag[i] - ai[i];
	while (nz--) {
	  vvs1 = v1[4]; vvs1 = v1[3]; vvs1 = v1[2]*x3; vvs1 = v1[1]; vvs1 = v1[0];
	  vvs2 = v2[4]; vvs2 = v2[3]; vvs2 = v2[2]*x3; vvs2 = v2[1]; vvs2 = v2[0];
	  vvs3 = v3[4]; vvs3 = v3[3]; vvs3 = v3[2]*x3; vvs3 = v3[1]; vvs3 = v3[0];
	  vvs4 = v4[4]; vvs4 = v4[3]; vvs4 = v4[2]*x3; vvs4 = v4[1]; vvs4 = v4[0];
	  vvs5 = v5[4]; vvs5 = v5[3]; vvs5 = v5[2]*x3; vvs5 = v5[1]; vvs5 = v5[0];
	  v1 -= 5; v2 -= 5; v3 -= 5; v4 -= 5; v5 -= 5;
	}
	x[i2]   = idiag[0]*s1 + idiag[5]*s2 + idiag[10]*s3 + idiag[15]*s4 + idiag[20]*s5;
	x[i2+1] = idiag[1]*s1 + idiag[6]*s2 + idiag[11]*s3 + idiag[16]*s4 + idiag[21]*s5;
	x[i2+2] = idiag[2]*s1 + idiag[7]*s2 + idiag[12]*s3 + idiag[17]*s4 + idiag[22]*s5;
	x[i2+3] = idiag[3]*s1 + idiag[8]*s2 + idiag[13]*s3 + idiag[18]*s4 + idiag[23]*s5;
	x[i2+4] = idiag[4]*s1 + idiag[9]*s2 + idiag[14]*s3 + idiag[19]*s4 + idiag[24]*s5;
        idiag   -= 25;
        i2      -= 5;
      }
      ierr = PetscLogFlops(25.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
/*=========================================================*/ 
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBSTRM_4"
PetscErrorCode MatMult_SeqBSTRM_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  MatScalar         *aa = a->a;
  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*zarray;
  const PetscScalar *x,*xb;
  const MatScalar   *v1, *v2, *v3, *v4;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscBool         usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;

  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }
  
  PetscInt slen = 4*(ii[mbs]-ii[0]);
  v1 = bstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++;
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0;
    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3  + v1[3]*x4;
      sum2 += v2[0]*x1 + v2[1]*x2 + v2[2]*x3  + v2[3]*x4;
      sum3 += v3[0]*x1 + v3[1]*x2 + v3[2]*x3  + v3[3]*x4;
      sum4 += v4[0]*x1 + v4[1]*x2 + v4[2]*x3  + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    if (usecprow) z = zarray + 4*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    if (!usecprow) z += 4;
  }
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(32*a->nz - 4*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBSTRM_5"
PetscErrorCode MatMult_SeqBSTRM_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  MatScalar         *aa = a->a;
  Mat_SeqBSTRM      *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5,*zarray;
  const PetscScalar *x,*xb;
  const MatScalar   *v1, *v2, *v3, *v4, *v5;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscBool         usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;

  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }
  
  PetscInt slen = 5*(ii[mbs]-ii[0]);
  v1 = bstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  v5 = v4 + slen;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++;
    sum1 = sum2 = sum3 = sum4 = sum5 = 0.0;
    nonzerorow += (n>0);
    PetscPrefetchBlock(idx+n,n,0,PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v1+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v2+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v3+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v4+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v5+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */


    for (j=0; j<n; j++) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      sum2 += v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      sum3 += v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      sum4 += v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4 + v4[4]*x5;
      sum5 += v5[0]*x1 + v5[1]*x2 + v5[2]*x3 + v5[3]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    if (usecprow) z = zarray + 5*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    if (!usecprow) z += 5;
  }
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(50.0*a->nz - 5.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 

/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBSTRM_4"
PetscErrorCode MatMultTranspose_SeqBSTRM_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ      *a = (Mat_SeqBAIJ*)A->data;
  MatScalar         *aa = a->a;
  Mat_SeqBSTRM     *sbstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar       zero = 0.0;  
  PetscScalar       x1,x2,x3,x4;
  PetscScalar       *x, *xb, *z;
  MatScalar         *v1, *v2, *v3, *v4;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;

  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  PetscInt slen = 4*(ai[mbs]-ai[0]);
  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i]; 
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];  xb += 4;
    nonzerorow += (n>0);
    ib = aj + ai[i];
    
    for (j=0; j<n; j++) {
      cval       = ib[j]*4;
      z[cval]     += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4;
      z[cval+1]   += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4;
      z[cval+2]   += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4;
      z[cval+3]   += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(32*a->nz - 4*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBSTRM_5"
PetscErrorCode MatMultTranspose_SeqBSTRM_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  MatScalar         *aa = a->a;
  Mat_SeqBSTRM      *sbstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar       zero = 0.0;  
  PetscScalar       *z = 0;
  PetscScalar       *x,*xb;
  const MatScalar   *v1, *v2, *v3, *v4, *v5;
  PetscScalar       x1, x2, x3, x4, x5;
  PetscErrorCode    ierr;
  PetscInt       mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,*ib,cval,j,jmin;
  PetscInt       nonzerorow=0;


  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  PetscInt slen = 5*(ai[mbs]-ai[0]);

  v1 = sbstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  v5 = v4 + slen;

  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[i+1] - ai[i];
    nonzerorow += (n>0);

    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; xb += 5;
    ib = aj + ai[i];

    PetscPrefetchBlock(ib+n,n,0,PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v1+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v2+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v3+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v4+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v5+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */

    
    for (j=0; j<n; j++) {
      cval       = ib[j]*5;
      z[cval]   += v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4 + v5[0]*x5;
      z[cval+1] += v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4 + v5[1]*x5;
      z[cval+2] += v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4 + v5[2]*x5;
      z[cval+3] += v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4 + v5[3]*x5;
      z[cval+4] += v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(50.0*a->nz - 5.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBSTRM_4"
PetscErrorCode MatMultAdd_SeqBSTRM_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBSTRM    *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*yarray,*zarray;
  MatScalar      *v1, *v2, *v3, *v4;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscBool      usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx   = a->j;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,4*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }  

  PetscInt slen = 4*(ii[mbs]-ii[0]);
  v1 = bstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 4*ridx[i];
      y = yarray + 4*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3];
    for (j=0; j<n; j++) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3  + v1[3]*x4;
      sum2 += v2[0]*x1 + v2[1]*x2 + v2[2]*x3  + v2[3]*x4;
      sum3 += v3[0]*x1 + v3[1]*x2 + v3[2]*x3  + v3[3]*x4;
      sum4 += v4[0]*x1 + v4[1]*x2 + v4[2]*x3  + v4[3]*x4;
      v1 += 4; v2 += 4; v3 += 4; v4 += 4;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    if (!usecprow){
      z += 4; y += 4;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(32.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBSTRM_5"
PetscErrorCode MatMultAdd_SeqBSTRM_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqBSTRM   *bstrm = (Mat_SeqBSTRM *)A->spptr;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5;
  PetscScalar    *yarray,*zarray;
  MatScalar      *v1,*v2,*v3,*v4,*v5;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscBool      usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }


  idx = a->j;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,5*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  PetscInt slen = 5*(ii[mbs]-ii[0]);
  v1 = bstrm->as;
  v2 = v1 + slen;
  v3 = v2 + slen;
  v4 = v3 + slen;
  v5 = v4 + slen;


  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 5*ridx[i];
      y = yarray + 5*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3]; sum5 = y[4];
    PetscPrefetchBlock(idx+n,n,0,PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v1+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v2+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v3+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v4+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    PetscPrefetchBlock(v5+5*n,5*n,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */


    for (j=0; j<n; j++) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v1[0]*x1 + v1[1]*x2 + v1[2]*x3 + v1[3]*x4 + v1[4]*x5;
      sum2 += v2[0]*x1 + v2[1]*x2 + v2[2]*x3 + v2[3]*x4 + v2[4]*x5;
      sum3 += v3[0]*x1 + v3[1]*x2 + v3[2]*x3 + v3[3]*x4 + v3[4]*x5;
      sum4 += v4[0]*x1 + v4[1]*x2 + v4[2]*x3 + v4[3]*x4 + v4[4]*x5;
      sum5 += v5[0]*x1 + v5[1]*x2 + v5[2]*x3 + v5[3]*x4 + v5[4]*x5;
      v1 += 5; v2 += 5; v3 += 5; v4 += 5; v5 += 5;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    if (!usecprow){
      z += 5; y += 5;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(50.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/ 
#undef __FUNCT__
#define __FUNCT__ "SeqBSTRM_create_bstrm"
PetscErrorCode SeqBSTRM_create_bstrm(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *) A->data;
  Mat_SeqBSTRM   *bstrm = (Mat_SeqBSTRM*) A->spptr;
  PetscInt       MROW = a->mbs, bs = A->rmap->bs;
  PetscInt       *aj = a->j, *ai = a->i, *ilen=a->ilen;
  PetscInt       i,j,k, rmax = a->rmax, *idiag, nnz;
  MatScalar      *aa = a->a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bstrm->rbs = bstrm->cbs = bs;
  PetscInt  bs2, rbs,  cbs, blen, slen;

  rbs = cbs = bs;
  bs2 = rbs*cbs; 
  blen = ai[MROW]-ai[0];
  slen = blen*cbs;

  ierr  = PetscMalloc(bs2*blen*sizeof(PetscScalar), &bstrm->as);CHKERRQ(ierr);

  PetscScalar **asp;
  ierr  = PetscMalloc(rbs*sizeof(PetscScalar *), &asp);CHKERRQ(ierr);
   
  for(i=0;i<rbs;i++) asp[i] = bstrm->as + i*slen; 
   
  for (k=0; k<blen; k++) {
    for (j=0; j<cbs; j++) 
    for (i=0; i<rbs; i++) 
        asp[i][k*cbs+j] = aa[k*bs2+j*rbs+i];
  }

  /* ierr = PetscFree(a->a);CHKERRQ(ierr);  */

  switch (bs){
    case 4:
       A->ops->mult          = MatMult_SeqBSTRM_4;
       A->ops->multadd       = MatMultAdd_SeqBSTRM_4;
       A->ops->multtranspose = MatMultTranspose_SeqBSTRM_4;
       /**/ A->ops->sor      = MatSOR_SeqBSTRM_4;  /**/
       break; 
    case 5:
       A->ops->mult          = MatMult_SeqBSTRM_5;
       A->ops->multadd       = MatMultAdd_SeqBSTRM_5;
       A->ops->multtranspose = MatMultTranspose_SeqBSTRM_5;
       /**/ A->ops->sor      = MatSOR_SeqBSTRM_5;  /**/
       break; 
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
  PetscFunctionReturn(0);
}
/*=========================================================*/ 

/*=========================================================*/ 
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBSTRMTransform"
PetscErrorCode MatSeqBSTRMTransform(Mat A)
{
  PetscFunctionBegin;
    SeqBSTRM_convert_bstrm(A);
  PetscFunctionReturn(0);
}
EXTERN_C_END

