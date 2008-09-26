#define PETSCMAT_DLL

/* 
        Provides an interface for the Matlab engine sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"

#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMatlabEnginePut_Matlab"
PetscErrorCode PETSCMAT_DLLEXPORT MatMatlabEnginePut_Matlab(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  Mat            B = (Mat)obj;
  mxArray        *mat; 
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)B->data;

  PetscFunctionBegin;
  mat  = mxCreateSparse(B->cmap->n,B->rmap->n,aij->nz,mxREAL);
  //mat  = mxCreateSparse(((PetscObject)B)->cmap.n,((PetscObject)B)->rmap.n,((Mat_SeqAIJ*)aij)->nz,mxREAL); 
  ierr = PetscMemcpy(mxGetPr(mat),aij->a,aij->nz*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Matlab stores by column, not row so we pass in the transpose of the matrix */
  ierr = PetscMemcpy(mxGetIr(mat),aij->j,aij->nz*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(mxGetJc(mat),aij->i,(B->rmap->n+1)*sizeof(int));CHKERRQ(ierr);

  /* Matlab indices start at 0 for sparse (what a surprise) */
  
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine *)mengine,obj->name,mat);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMatlabEngineGet_Matlab"
PetscErrorCode PETSCMAT_DLLEXPORT MatMatlabEngineGet_Matlab(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  int            ii;
  Mat            mat = (Mat)obj;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)mat->data;
  mxArray        *mmat; 

  PetscFunctionBegin;
  ierr = MatSeqXAIJFreeAIJ(mat,&aij->a,&aij->j,&aij->i);CHKERRQ(ierr);

  mmat = engGetVariable((Engine *)mengine,obj->name);

  aij->nz           = (mxGetJc(mmat))[mat->rmap->n];
  ierr  = PetscMalloc3(aij->nz,PetscScalar,&aij->a,aij->nz,PetscInt,&aij->j,mat->rmap->n+1,PetscInt,&aij->i);CHKERRQ(ierr);
  aij->singlemalloc = PETSC_TRUE;

  ierr = PetscMemcpy(aij->a,mxGetPr(mmat),aij->nz*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Matlab stores by column, not row so we pass in the transpose of the matrix */
  ierr = PetscMemcpy(aij->j,mxGetIr(mmat),aij->nz*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(aij->i,mxGetJc(mmat),(mat->rmap->n+1)*sizeof(int));CHKERRQ(ierr);

  for (ii=0; ii<mat->rmap->n; ii++) {
    aij->ilen[ii] = aij->imax[ii] = aij->i[ii+1] - aij->i[ii];
  }

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

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
PetscErrorCode MatLUFactorNumeric_Matlab(Mat F,Mat A,MatFactorInfo *info)
{
  PetscErrorCode ierr;
  size_t         len;
  char           *_A,*name;

  PetscFunctionBegin;
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = ((PetscObject)A)->name;
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"[l_%s,u_%s,p_%s] = lu(%s',%g);",_A,_A,_A,_A,info->dtcol);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = 0;",_A);CHKERRQ(ierr);
  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  F->ops->solve              = MatSolve_Matlab;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_Matlab"
PetscErrorCode MatLUFactorSymbolic_Matlab(Mat F,Mat A,IS r,IS c,MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  F->ops->lufactornumeric    = MatLUFactorNumeric_Matlab;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_matlab"
PetscErrorCode MatGetFactor_seqaij_matlab(Mat A,MatFactorType ftype,Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                        = MatCreate(((PetscObject)A)->comm,F);CHKERRQ(ierr);
  ierr                        = MatSetSizes(*F,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr                        = MatSetType(*F,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr                        = MatSeqAIJSetPreallocation(*F,0,PETSC_NULL);CHKERRQ(ierr);
  (*F)->ops->lufactorsymbolic = MatLUFactorSymbolic_Matlab;

  (*F)->factor                = MAT_FACTOR_LU;
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactor_Matlab"
PetscErrorCode MatILUDTFactor_Matlab(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *F)
{
  PetscErrorCode ierr;
  size_t         len;
  char           *_A,*name;

  PetscFunctionBegin;
  if (info->dt == PETSC_DEFAULT)      info->dt      = .005;
  if (info->dtcol == PETSC_DEFAULT)   info->dtcol   = .01;
  if (A->cmap->N != A->rmap->N) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                       = MatCreate(((PetscObject)A)->comm,F);CHKERRQ(ierr);
  ierr                       = MatSetSizes(*F,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr                       = MatSetType(*F,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr                       = MatSeqAIJSetPreallocation(*F,0,PETSC_NULL);CHKERRQ(ierr);
  (*F)->ops->solve           = MatSolve_Matlab;
  (*F)->factor               = MAT_FACTOR_LU;
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = ((PetscObject)A)->name;
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"info_%s = struct('droptol',%g,'thresh',%g);",_A,info->dt,info->dtcol);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"[l_%s,u_%s,p_%s] = luinc(%s',info_%s);",_A,_A,_A,_A,_A);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(((PetscObject)A)->comm),"%s = 0;",_A);CHKERRQ(ierr);

  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)*F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_Matlab"
PetscErrorCode MatFactorInfo_Matlab(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin; 
  ierr = PetscViewerASCIIPrintf(viewer,"Matlab run parameters:  -- not written yet!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Matlab"
PetscErrorCode MatView_Matlab(Mat A,PetscViewer viewer) 
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MatView_SeqAIJ(A,viewer);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_Matlab(A,viewer);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
  MATMATLAB - MATMATLAB = "matlab" - A matrix type providing direct solvers (LU and QR) and drop tolerance
  based ILU factorization (ILUDT) for sequential matrices via the external package Matlab.

  If Matlab is instaled (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes Matlab solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATMATLAB).
  This matrix type is only supported for double precision real.

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQAIJ type without data copy.

  Options Database Keys:
+ -mat_type matlab - sets the matrix type to "matlab" during a call to MatSetFromOptions()
- -mat_matlab_qr   - sets the direct solver to be QR instead of LU

  Level: beginner

.seealso: PCLU
M*/

