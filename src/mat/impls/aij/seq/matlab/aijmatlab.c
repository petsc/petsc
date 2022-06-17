
/*
        Provides an interface for the MATLAB engine sparse solver

*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <petscmatlab.h>
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */

PETSC_EXTERN mxArray *MatSeqAIJToMatlab(Mat B)
{
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)B->data;
  mwIndex        *ii,*jj;
  mxArray        *mat;
  PetscInt       i;

  PetscFunctionBegin;
  mat  = mxCreateSparse(B->cmap->n,B->rmap->n,aij->nz,mxREAL);
  if (PetscArraycpy(mxGetPr(mat),aij->a,aij->nz)) return NULL;
  /* MATLAB stores by column, not row so we pass in the transpose of the matrix */
  jj = mxGetIr(mat);
  for (i=0; i<aij->nz; i++) jj[i] = aij->j[i];
  ii = mxGetJc(mat);
  for (i=0; i<B->rmap->n+1; i++) ii[i] = aij->i[i];
  PetscFunctionReturn(mat);
}

PETSC_EXTERN PetscErrorCode MatlabEnginePut_SeqAIJ(PetscObject obj,void *mengine)
{
  mxArray        *mat;

  PetscFunctionBegin;
  mat  = MatSeqAIJToMatlab((Mat)obj);PetscCheck(mat,PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot create MATLAB matrix");
  PetscCall(PetscObjectName(obj));
  engPutVariable((Engine*)mengine,obj->name,mat);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqAIJFromMatlab(mxArray *mmat,Mat mat)
{
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
    PetscCall(MatSetSizes(mat,n,m,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetUp(mat));
  } else {
    PetscCheck(mat->rmap->n == n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change size of PETSc matrix %" PetscInt_FMT " to %" PetscInt_FMT,mat->rmap->n,n);
    PetscCheck(mat->cmap->n == m,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change size of PETSc matrix %" PetscInt_FMT " to %" PetscInt_FMT,mat->cmap->n,m);
  }
  if (nz != aij->nz) {
    /* number of nonzeros in matrix has changed, so need new data structure */
    PetscCall(MatSeqXAIJFreeAIJ(mat,&aij->a,&aij->j,&aij->i));
    aij->nz = nz;
    PetscCall(PetscMalloc3(aij->nz,&aij->a,aij->nz,&aij->j,mat->rmap->n+1,&aij->i));

    aij->singlemalloc = PETSC_TRUE;
  }

  PetscCall(PetscArraycpy(aij->a,mxGetPr(mmat),aij->nz));
  /* MATLAB stores by column, not row so we pass in the transpose of the matrix */
  i = aij->i;
  for (k=0; k<n+1; k++) i[k] = (PetscInt) ii[k];
  j = aij->j;
  for (k=0; k<nz; k++) j[k] = (PetscInt) jj[k];

  for (k=0; k<mat->rmap->n; k++) aij->ilen[k] = aij->imax[k] = aij->i[k+1] - aij->i[k];

  mat->nonzerostate++; /* since the nonzero structure can change anytime force the Inode information to always be rebuilt */
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode  MatlabEngineGet_SeqAIJ(PetscObject obj,void *mengine)
{
  Mat            mat = (Mat)obj;
  mxArray        *mmat;

  PetscFunctionBegin;
  mmat = engGetVariable((Engine*)mengine,obj->name);
  PetscCall(MatSeqAIJFromMatlab(mmat,mat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_Matlab(Mat A,Vec b,Vec x)
{
  const char     *_A,*_b,*_x;

  PetscFunctionBegin;
  /* make sure objects have names; use default if not */
  PetscCall(PetscObjectName((PetscObject)b));
  PetscCall(PetscObjectName((PetscObject)x));

  PetscCall(PetscObjectGetName((PetscObject)A,&_A));
  PetscCall(PetscObjectGetName((PetscObject)b,&_b));
  PetscCall(PetscObjectGetName((PetscObject)x,&_x));
  PetscCall(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),(PetscObject)b));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"%s = u%s\\(l%s\\(p%s*%s));",_x,_A,_A,_A,_b));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"%s = 0;",_b));
  /* PetscCall(PetscMatlabEnginePrintOutput(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),stdout));  */
  PetscCall(PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),(PetscObject)x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_Matlab(Mat F,Mat A,const MatFactorInfo *info)
{
  size_t         len;
  char           *_A,*name;
  PetscReal      dtcol = info->dtcol;

  PetscFunctionBegin;
  if (F->factortype == MAT_FACTOR_ILU || info->dt > 0) {
    /* the ILU form is not currently registered */
    if (info->dtcol == PETSC_DEFAULT) dtcol = .01;
    F->ops->solve = MatSolve_Matlab;
    F->factortype = MAT_FACTOR_LU;

    PetscCall(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),(PetscObject)A));
    _A   = ((PetscObject)A)->name;
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"info_%s = struct('droptol',%g,'thresh',%g);",_A,info->dt,dtcol));
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"[l_%s,u_%s,p_%s] = luinc(%s',info_%s);",_A,_A,_A,_A,_A));
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"%s = 0;",_A));

    PetscCall(PetscStrlen(_A,&len));
    PetscCall(PetscMalloc1(len+2,&name));
    sprintf(name,"_%s",_A);
    PetscCall(PetscObjectSetName((PetscObject)F,name));
    PetscCall(PetscFree(name));
  } else {
    PetscCall(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),(PetscObject)A));
    _A   = ((PetscObject)A)->name;
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"[l_%s,u_%s,p_%s] = lu(%s',%g);",_A,_A,_A,_A,dtcol));
    PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"%s = 0;",_A));
    PetscCall(PetscStrlen(_A,&len));
    PetscCall(PetscMalloc1(len+2,&name));
    sprintf(name,"_%s",_A);
    PetscCall(PetscObjectSetName((PetscObject)F,name));
    PetscCall(PetscFree(name));

    F->ops->solve = MatSolve_Matlab;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_Matlab(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCheck(A->cmap->N == A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square");
  F->ops->lufactornumeric = MatLUFactorNumeric_Matlab;
  F->assembled            = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_matlab(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMATLAB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_matlab(Mat A)
{
  const char     *_A;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)A,&_A));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PetscObjectComm((PetscObject)A)),"delete %s l_%s u_%s;",_A,_A,_A));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_matlab(Mat A,MatFactorType ftype,Mat *F)
{
  PetscFunctionBegin;
  PetscCheck(A->cmap->N == A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"matrix must be square");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),F));
  PetscCall(MatSetSizes(*F,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  PetscCall(PetscStrallocpy("matlab",&((PetscObject)*F)->type_name));
  PetscCall(MatSetUp(*F));

  (*F)->ops->destroy           = MatDestroy_matlab;
  (*F)->ops->getinfo           = MatGetInfo_External;
  (*F)->trivialsymbolic        = PETSC_TRUE;
  (*F)->ops->lufactorsymbolic  = MatLUFactorSymbolic_Matlab;
  (*F)->ops->ilufactorsymbolic = MatLUFactorSymbolic_Matlab;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*F),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_matlab));

  (*F)->factortype = ftype;
  PetscCall(PetscFree((*F)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMATLAB,&(*F)->solvertype));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Matlab(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERMATLAB,MATSEQAIJ,        MAT_FACTOR_LU,MatGetFactor_seqaij_matlab));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

PetscErrorCode MatView_Info_Matlab(Mat A,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer,"MATLAB run parameters:  -- not written yet!\n"));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_Matlab(Mat A,PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(MatView_SeqAIJ(A,viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) PetscCall(MatView_Info_Matlab(A,viewer));
  }
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERMATLAB - "matlab" - Providing direct solver LU for sequential aij matrix via the external package MATLAB.

  Works with MATSEQAIJ matrices.

  Options Database Keys:
. -pc_factor_mat_solver_type matlab - selects MATLAB to do the sparse factorization

  Notes:
    You must ./configure with the options --with-matlab --with-matlab-engine

  Level: beginner

.seealso: `PCLU`

.seealso: `PCFactorSetMatSolverType()`, `MatSolverType`
M*/
