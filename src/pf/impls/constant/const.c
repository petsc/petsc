/*$Id: const.c,v 1.3 2000/04/09 03:11:43 bsmith Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "pf.h" I*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PFApply_Constant"
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
#define  __FUNC__ /*<a name=""></a>*/"PFApplyVec_Constant"
int PFApplyVec_Constant(void *value,Vec x,Vec y)
{
  int ierr;
  PetscFunctionBegin;
  ierr = VecSet((Scalar*)value,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PFView_Constant"
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
#define  __FUNC__ /*<a name=""></a>*/"PFDestroy_Constant"
int PFDestroy_Constant(void *value)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PFSetFromOptions_Constant"
int PFSetFromOptions_Constant(PF pf)
{
  int        ierr;
  PetscTruth flag;
  Scalar     *value = (Scalar *)pf->data;

  PetscFunctionBegin;
  ierr = OptionsGetScalar(pf->prefix,"-pf_constant",value,&flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PFCreate_Constant"
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
#define  __FUNC__ /*<a name=""></a>*/"PFCreate_Quick"
int PFCreate_Quick(PF pf,void* function)
{
  int  ierr;

  PetscFunctionBegin;

  ierr = PFSet(pf,(int (*)(void*,int,Scalar*,Scalar*))function,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
