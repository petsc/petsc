/*$Id: aijmatlab.c,v 1.12 2001/08/06 21:15:14 bsmith Exp $*/

/* 
        Provides an interface for the Matlab engine sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_Matlab"
int MatSolve_SeqAIJ_Matlab(Mat A,Vec b,Vec x)
{
  int             ierr;
  char            *_A,*_b,*_x;

  PetscFunctionBegin;
  /* make sure objects have names; use default if not */
  ierr = PetscObjectName((PetscObject)b);CHKERRQ(ierr);
  ierr = PetscObjectName((PetscObject)x);CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject)A,&_A);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)b,&_b);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)x,&_x);CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = u%s\\(l%s\\(p%s*%s));",_x,_A,_A,_A,_b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = 0;",_b);CHKERRQ(ierr);
  /* ierr = PetscMatlabEnginePrintOutput(PETSC_MATLAB_ENGINE_(A->comm),stdout);CHKERRQ(ierr);  */
  ierr = PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_Matlab"
int MatLUFactorNumeric_SeqAIJ_Matlab(Mat A,Mat *F)
{
  Mat_SeqAIJ      *f = (Mat_SeqAIJ*)(*F)->data;
  int             ierr,len;
  char            *_A,*name;

  PetscFunctionBegin;
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = A->name;
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"[l_%s,u_%s,p_%s] = lu(%s',%g);",_A,_A,_A,_A,f->lu_dtcol);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = 0;",_A);CHKERRQ(ierr);
  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)*F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Matlab"
int MatLUFactorSymbolic_SeqAIJ_Matlab(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  int        ierr;
  Mat_SeqAIJ *f;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                       = MatCreateSeqAIJ(A->comm,A->m,A->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  (*F)->ops->solve           = MatSolve_SeqAIJ_Matlab;
  (*F)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Matlab;
  (*F)->factor               = FACTOR_LU;
  f                          = (Mat_SeqAIJ*)(*F)->data;
  f->lu_dtcol = info->dtcol;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_Matlab_QR"
int MatSolve_SeqAIJ_Matlab_QR(Mat A,Vec b,Vec x)
{
  int             ierr;
  char            *_A,*_b,*_x;

  PetscFunctionBegin;
  /* make sure objects have names; use default if not */
  ierr = PetscObjectName((PetscObject)b);CHKERRQ(ierr);
  ierr = PetscObjectName((PetscObject)x);CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject)A,&_A);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)b,&_b);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)x,&_x);CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = r%s\\(r%s'\\(%s*%s));",_x,_A,_A,_A+1,_b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = 0;",_b);CHKERRQ(ierr);
  /* ierr = PetscMatlabEnginePrintOutput(PETSC_MATLAB_ENGINE_(A->comm),stdout);CHKERRQ(ierr);  */
  ierr = PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_Matlab_QR"
int MatLUFactorNumeric_SeqAIJ_Matlab_QR(Mat A,Mat *F)
{
  int             ierr,len;
  char            *_A,*name;

  PetscFunctionBegin;
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = A->name;
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"r_%s = qr(%s');",_A,_A);CHKERRQ(ierr);
  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)*F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Matlab_QR"
int MatLUFactorSymbolic_SeqAIJ_Matlab_QR(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  int  ierr;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                       = MatCreateSeqAIJ(A->comm,A->m,A->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  (*F)->ops->solve           = MatSolve_SeqAIJ_Matlab_QR;
  (*F)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Matlab_QR;
  (*F)->factor               = FACTOR_LU;
  (*F)->assembled            = PETSC_TRUE;  /* required by -sles_view */

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactor_SeqAIJ_Matlab"
int MatILUDTFactor_SeqAIJ_Matlab(Mat A,MatILUInfo *info,IS isrow,IS iscol,Mat *F)
{
  int        ierr,len;
  char       *_A,*name;

  PetscFunctionBegin;
  if (info->dt == PETSC_DEFAULT)      info->dt      = .005;
  if (info->dtcol == PETSC_DEFAULT)   info->dtcol   = .01;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr                       = MatCreateSeqAIJ(A->comm,A->m,A->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  (*F)->ops->solve           = MatSolve_SeqAIJ_Matlab;
  (*F)->factor               = FACTOR_LU;
  ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(A->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = A->name;
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"info_%s = struct('droptol',%g,'thresh',%g);",_A,info->dt,info->dtcol);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"[l_%s,u_%s,p_%s] = luinc(%s',info_%s);",_A,_A,_A,_A,_A);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(A->comm),"%s = 0;",_A);CHKERRQ(ierr);

  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+2)*sizeof(char),&name);CHKERRQ(ierr);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)*F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int MatSeqAIJFactorInfo_Matlab(Mat A,PetscViewer viewer)
{
  int ierr;
  
  PetscFunctionBegin; 
  /* check if matrix is matlab type */
  /* if (A->ops->solve != MatSolve_SeqAIJ_Matlab) PetscFunctionReturn(0); */

  ierr = PetscViewerASCIIPrintf(viewer,"Matlab run parameters:  -- not written yet!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseMatlab_SeqAIJ"
int MatUseMatlab_SeqAIJ(Mat A)
{
  PetscTruth qr;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_matlab_qr",&qr);CHKERRQ(ierr);
  if (qr) {
    A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Matlab_QR;
    PetscLogInfo(0,"Using Matlab QR with iterative refinement for SeqAIJ LU factorization and solves");
  } else {
    A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Matlab;
    PetscLogInfo(0,"Using Matlab for SeqAIJ LU factorization and solves");
  }
  A->ops->iludtfactor      = MatILUDTFactor_SeqAIJ_Matlab;
  PetscLogInfo(0,"Using Matlab for SeqAIJ ILUDT factorization and solves");
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseMatlab_SeqAIJ"
int MatUseMatlab_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


