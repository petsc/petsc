/*$Id: const.c,v 1.12 2000/09/28 21:16:15 bsmith Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "petscpf.h" I*/

#undef __FUNC__  
#define __FUNC__ "PFApply_Constant"
int PFApply_Constant(void *value,int n,Scalar *x,Scalar *y)
{
  int    i;
  Scalar v = ((Scalar*)value)[0];

  PetscFunctionBegin;
  n *= (int) PetscRealPart(((Scalar*)value)[1]);
  for (i=0; i<n; i++) {
    y[i] = v;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFApplyVec_Constant"
int PFApplyVec_Constant(void *value,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecSet((Scalar*)value,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PFView_Constant"
int PFView_Constant(void *value,PetscViewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"Constant = %g\n",*(double*)value);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"Constant = %g + %gi\n",PetscRealPart(*(Scalar*)value),PetscImaginaryPart(*(Scalar*)value));CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PFDestroy_Constant"
int PFDestroy_Constant(void *value)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFSetFromOptions_Constant"
int PFSetFromOptions_Constant(PF pf)
{
  int        ierr;
  Scalar     *value = (Scalar *)pf->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Constant function options");CHKERRQ(ierr);
    ierr = PetscOptionsScalar("-pf_constant","The constant value","None",*value,value,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PFCreate_Constant"
int PFCreate_Constant(PF pf,void *value)
{
  int    ierr;
  Scalar *loc;

  PetscFunctionBegin;
ierr = PetscMalloc(2*sizeof(Scalar),&(  loc    ));CHKERRQ(ierr);
  if (value) loc[0] = *(Scalar*)value; else loc[0] = 0.0;
  loc[1] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Constant,PFApplyVec_Constant,PFView_Constant,PFDestroy_Constant,loc);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Constant;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PFCreate_Quick"
int PFCreate_Quick(PF pf,void* function)
{
  int  ierr;

  PetscFunctionBegin;

  ierr = PFSet(pf,(int (*)(void*,int,Scalar*,Scalar*))function,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PFApply_Identity"
int PFApply_Identity(void *value,int n,Scalar *x,Scalar *y)
{
  int    i;

  PetscFunctionBegin;
  n *= *(int*)value;
  for (i=0; i<n; i++) {
    y[i] = x[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFApplyVec_Identity"
int PFApplyVec_Identity(void *value,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PFView_Identity"
int PFView_Identity(void *value,PetscViewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Identity function\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PFDestroy_Identity"
int PFDestroy_Identity(void *value)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PFCreate_Identity"
int PFCreate_Identity(PF pf,void *value)
{
  int    ierr,*loc;

  PetscFunctionBegin;
  if (pf->dimout != pf->dimin) {
    SETERRQ2(1,"Input dimension must match output dimension for Identity function, dimin = %d dimout = %d\n",pf->dimin,pf->dimout);
  }
ierr = PetscMalloc(sizeof(int),&(  loc    ));CHKERRQ(ierr);
  loc[0] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Identity,PFApplyVec_Identity,PFView_Identity,PFDestroy_Identity,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
