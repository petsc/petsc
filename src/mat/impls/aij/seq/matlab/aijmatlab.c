
/* 
        Provides an interface for the MATLAB engine sparse solver

*/
#include <../src/mat/impls/aij/seq/aij.h>

#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJToMatlab"
mxArray *MatSeqAIJToMatlab(Mat B)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)B->data;
  mwIndex        *ii,*jj;
  mxArray        *mat;
  PetscInt       i;

  PetscFunctionBegin;
  mat  = mxCreateSparse(B->cmap->n,B->rmap->n,aij->nz,mxREAL);
  ierr = PetscMemcpy(mxGetPr(mat),aij->a,aij->nz*sizeof(PetscScalar));
  /* MATLAB stores by column, not row so we pass in the transpose of the matrix */
  jj = mxGetIr(mat);
  for (i=0; i<aij->nz; i++) jj[i] = aij->j[i];
  ii = mxGetJc(mat);
  for (i=0; i<B->rmap->n+1; i++) ii[i] = aij->i[i];
  
  PetscFunctionReturn(mat);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatlabEnginePut_SeqAIJ"
PetscErrorCode  MatlabEnginePut_SeqAIJ(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  mxArray        *mat; 

  PetscFunctionBegin;
  mat = MatSeqAIJToMatlab((Mat)obj);if (!mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot create MATLAB matrix");
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine *)mengine,obj->name,mat);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJFromMatlab"
/*@C
    MatSeqAIJFromMatlab - Given a MATLAB sparse matrix, fills a SeqAIJ matrix with its transpose.

   Not Collective

   Input Parameters:
+     mmat - a MATLAB sparse matris
-     mat - a already created MATSEQAIJ

  Level: intermediate

@*/
PetscErrorCode  MatSeqAIJFromMatlab(mxArray *mmat,Mat mat)
{
  PetscErrorCode ierr;
  PetscInt       nz,n,m,*i,*j,k;
  mwIndex        nnz,nn,nm,*ii,*jj;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)mat->data;

  PetscFunctionBegin;
  nn  = mxGetN(mmat);   /* rows of transpose of matrix */
  nm  = mxGetM(mmat);
  nnz = (mxGetJc(mmat))[nn];
  ii  = mxGetJc(mmat);
  jj  = mxGetIr(mmat);
  n   = (PetscInt) nn;  
  m   = (PetscInt) nm;
  nz  = (PetscInt) nnz;

  if (mat->rmap->n < 0 && mat->cmap->n < 0) {
    /* matrix has not yet had its size set */
    ierr = MatSetSizes(mat,n,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetUp(mat);CHKERRQ(ierr);
  } else {
    if (mat->rmap->n != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change size of PETSc matrix %D to %D",mat->rmap->n,n);
    if (mat->cmap->n != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change size of PETSc matrix %D to %D",mat->cmap->n,m);
  }
  if (nz != aij->nz) {
    /* number of nonzeros in matrix has changed, so need new data structure */
    ierr = MatSeqXAIJFreeAIJ(mat,&aij->a,&aij->j,&aij->i);CHKERRQ(ierr);
    aij->nz           = nz;
    ierr  = PetscMalloc3(aij->nz,PetscScalar,&aij->a,aij->nz,PetscInt,&aij->j,mat->rmap->n+1,PetscInt,&aij->i);CHKERRQ(ierr);
    aij->singlemalloc = PETSC_TRUE;
  }

  ierr = PetscMemcpy(aij->a,mxGetPr(mmat),aij->nz*sizeof(PetscScalar));CHKERRQ(ierr);
  /* MATLAB stores by column, not row so we pass in the transpose of the matrix */
  i = aij->i;
  for (k=0; k<n+1; k++) {
    i[k] = (PetscInt) ii[k];
  }
  j = aij->j;
  for (k=0; k<nz; k++) {
    j[k] = (PetscInt) jj[k];
  }

  for (k=0; k<mat->rmap->n; k++) {
    aij->ilen[k] = aij->imax[k] = aij->i[k+1] - aij->i[k];
  }

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatlabEngineGet_SeqAIJ"
PetscErrorCode  MatlabEngineGet_SeqAIJ(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  Mat            mat = (Mat)obj;
  mxArray        *mmat; 

  PetscFunctionBegin;
  mmat = engGetVariable((Engine *)mengine,obj->name);
  ierr = MatSeqAIJFromMatlab(mmat,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_Matlab"
PetscErrorCode MatSolve_Matlab(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;
  const char     *_A,*_b,*_x;

  PetscFunctionBegin;
  /* make sure objects have names; use default if not */
  ierr = PetscObjectName((PetscObject)b);CHKERRQ(ierr);
  ierr = PetscObjectName((PetscObject)x);CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject)A,&_A);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)b,&_b);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)x,&_x);CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = u%s\\(l%s\\(p%s*%s));",_x,_A,_A,_A,_b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = 0;",_b);CHKERRQ(ierr);
  /* ierr = PetscMatlabEnginePrintOutput(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),stdout);CHKERRQ(ierr);  */
  ierr = PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_Matlab"
PetscErrorCode MatLUFactorNumeric_Matlab(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  size_t         len;
  char           *_A,*name;
  PetscReal      dtcol = info->dtcol;

  PetscFunctionBegin;
  if (F->factortype == MAT_FACTOR_ILU || info->dt > 0) {
    if (info->dtcol == PETSC_DEFAULT)  dtcol = .01;
    F->ops->solve           = MatSolve_Matlab;
    F->factortype           = MAT_FACTOR_LU;
    ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)A);CHKERRQ(ierr);
    _A   = ((PetscObject)A)->name;
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"info_%s = struct('droptol',%g,'thresh',%g);",_A,info->dt,dtcol);CHKERRQ(ierr);
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"[l_%s,u_%s,p_%s] = luinc(%s',info_%s);",_A,_A,_A,_A,_A);CHKERRQ(ierr);
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = 0;",_A);CHKERRQ(ierr);

    ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
    ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
    sprintf(name,"_%s",_A);
    ierr = PetscObjectSetName((PetscObject)F,name);CHKERRQ(ierr);
    ierr = PetscFree(name);CHKERRQ(ierr);
  } else {
    ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)A);CHKERRQ(ierr);
    _A   = ((PetscObject)A)->name;
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"[l_%s,u_%s,p_%s] = lu(%s',%g);",_A,_A,_A,_A,dtcol);CHKERRQ(ierr);
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = 0;",_A);CHKERRQ(ierr);
    ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
    ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
    sprintf(name,"_%s",_A);
    ierr = PetscObjectSetName((PetscObject)F,name);CHKERRQ(ierr);
    ierr = PetscFree(name);CHKERRQ(ierr);
    F->ops->solve              = MatSolve_Matlab;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_Matlab"
PetscErrorCode MatLUFactorSymbolic_Matlab(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  F->ops->lufactornumeric    = MatLUFactorNumeric_Matlab;
  F->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_matlab"
PetscErrorCode MatFactorGetSolverPackage_seqaij_matlab(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMATLAB;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_matlab"
PetscErrorCode MatGetFactor_seqaij_matlab(Mat A,MatFactorType ftype,Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                         = MatCreate(((PetscObject)A)->comm,F);CHKERRQ(ierr);
  ierr                         = MatSetSizes(*F,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr                         = MatSetType(*F,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr                         = MatSeqAIJSetPreallocation(*F,0,PETSC_NULL);CHKERRQ(ierr);
  (*F)->ops->lufactorsymbolic  = MatLUFactorSymbolic_Matlab;
  (*F)->ops->ilufactorsymbolic = MatLUFactorSymbolic_Matlab;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)(*F),"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_matlab",MatFactorGetSolverPackage_seqaij_matlab);CHKERRQ(ierr);

  (*F)->factortype             = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_Matlab"
PetscErrorCode MatFactorInfo_Matlab(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin; 
  ierr = PetscViewerASCIIPrintf(viewer,"MATLAB run parameters:  -- not written yet!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Matlab"
PetscErrorCode MatView_Matlab(Mat A,PetscViewer viewer) 
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MatView_SeqAIJ(A,viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_Matlab(A,viewer);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERMATLAB - "matlab" - Providing direct solvers (LU and QR) and drop tolerance
  based ILU factorization (ILUDT) for sequential matrices via the external package MATLAB.


  Works with MATSEQAIJ matrices.

  Options Database Keys:
. -pc_factor_mat_solver_type matlab - selects MATLAB to do the sparse factorization


  Level: beginner

.seealso: PCLU

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage
M*/

