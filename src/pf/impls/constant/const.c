/*$Id: const.c,v 1.10 2000/08/25 16:45:17 balay Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "petscpf.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PFApply_Constant"></a>*/"PFApply_Constant"
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
#define __FUNC__ /*<a name="PFApplyVec_Constant"></a>*/"PFApplyVec_Constant"
int PFApplyVec_Constant(void *value,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecSet((Scalar*)value,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="PFView_Constant"></a>*/"PFView_Constant"
int PFView_Constant(void *value,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = ViewerASCIIPrintf(viewer,"Constant = %g\n",*(double*)value);CHKERRQ(ierr);
#else
    ierr = ViewerASCIIPrintf(viewer,"Constant = %g + %gi\n",PetscRealPart(*(Scalar*)value),PetscImaginaryPart(*(Scalar*)value));CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="PFDestroy_Constant"></a>*/"PFDestroy_Constant"
int PFDestroy_Constant(void *value)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFSetFromOptions_Constant"></a>*/"PFSetFromOptions_Constant"
int PFSetFromOptions_Constant(PF pf)
{
  int        ierr;
  Scalar     *value = (Scalar *)pf->data;

  PetscFunctionBegin;
  ierr = OptionsHead("Constant function options");CHKERRQ(ierr);
    ierr = OptionsScalar("-pf_constant","The constant value","None",*value,value,0);CHKERRQ(ierr);
  ierr = OptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PFCreate_Constant"></a>*/"PFCreate_Constant"
int PFCreate_Constant(PF pf,void *value)
{
  int    ierr;
  Scalar *loc;

  PetscFunctionBegin;
  loc    = (Scalar*)PetscMalloc(2*sizeof(Scalar));CHKPTRQ(loc);
  if (value) loc[0] = *(Scalar*)value; else loc[0] = 0.0;
  loc[1] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Constant,PFApplyVec_Constant,PFView_Constant,PFDestroy_Constant,loc);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Constant;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PFCreate_Quick"></a>*/"PFCreate_Quick"
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
#define __FUNC__ /*<a name="PFApply_Identity"></a>*/"PFApply_Identity"
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
#define __FUNC__ /*<a name="PFApplyVec_Identity"></a>*/"PFApplyVec_Identity"
int PFApplyVec_Identity(void *value,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="PFView_Identity"></a>*/"PFView_Identity"
int PFView_Identity(void *value,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"Identity function\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ /*<a name="PFDestroy_Identity"></a>*/"PFDestroy_Identity"
int PFDestroy_Identity(void *value)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PFCreate_Identity"></a>*/"PFCreate_Identity"
int PFCreate_Identity(PF pf,void *value)
{
  int    ierr,*loc;

  PetscFunctionBegin;
  if (pf->dimout != pf->dimin) {
    SETERRQ2(1,1,"Input dimension must match output dimension for Identity function, dimin = %d dimout = %d\n",pf->dimin,pf->dimout);
  }
  loc    = (int*)PetscMalloc(sizeof(int));CHKPTRQ(loc);
  loc[0] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Identity,PFApplyVec_Identity,PFView_Identity,PFDestroy_Identity,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
