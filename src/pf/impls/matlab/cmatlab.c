/*$Id: cmatlab.c,v 1.2 2000/05/20 20:29:56 bsmith Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "petscpf.h" I*/

/*
        Ths PF generates a Matlab function on the fly
*/
typedef struct {
  int               dimin,dimout;
  PetscMatlabEngine engine;
  char              *string;
} PF_Matlab;
  
#undef __FUNC__  
#define __FUNC__ /*<a name="PFView_Matlab"></a>*/"PFView_Matlab"
int PFView_Matlab(void *value,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;
  PF_Matlab  *matlab = (PF_Matlab*)value;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"Matlab Matlab = %s\n",matlab->string);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFDestroy_Matlab"></a>*/"PFDestroy_Matlab"
int PFDestroy_Matlab(void *value)
{
  int        ierr;
  PF_Matlab  *matlab = (PF_Matlab*)value;

  PetscFunctionBegin;
  ierr = PetscStrfree(matlab->string);CHKERRQ(ierr);
  ierr = PetscMatlabEngineDestroy(matlab->engine);CHKERRQ(ierr);
  ierr = PetscFree(matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFApply_Matlab"></a>*/"PFApply_Matlab"
int PFApply_Matlab(void *value,int n,Scalar *in,Scalar *out)
{
  PF_Matlab  *matlab = (PF_Matlab*)value;
  int        ierr;

  PetscFunctionBegin;
  if (!value) SETERRQ(1,1,"Need to set string for Matlab function, via -pf_matlab string");
  ierr = PetscMatlabEnginePutArray(matlab->engine,matlab->dimin,n,in,"x");CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(matlab->engine,matlab->string);CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(matlab->engine,matlab->dimout,n,out,"f");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFSetFromOptions_Matlab"></a>*/"PFSetFromOptions_Matlab"
int PFSetFromOptions_Matlab(PF pf)
{
  int        ierr;
  PetscTruth flag;
  char       value[256];
  PF_Matlab  *matlab = (PF_Matlab*)pf->data;

  PetscFunctionBegin;
  ierr = OptionsBegin(pf->comm,pf->prefix,"Matlab function options");CHKERRQ(ierr);
    ierr = OptionsString("-pf_matlab","Matlab function","None","",value,256,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
    }
  ierr = OptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PFCreate_Matlab"></a>*/"PFCreate_Matlab"
int PFCreate_Matlab(PF pf,void *value)
{
  int       ierr;
  PF_Matlab *matlab;

  PetscFunctionBegin;
  matlab = PetscNew(PF_Matlab);CHKPTRQ(matlab);
  matlab->dimin  = pf->dimin;
  matlab->dimout = pf->dimout;

  ierr = PetscMatlabEngineCreate(pf->comm,PETSC_NULL,&matlab->engine);CHKERRQ(ierr);
    
  if (value) {
    ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
  }
  ierr   = PFSet(pf,PFApply_Matlab,PETSC_NULL,PFView_Matlab,PFDestroy_Matlab,matlab);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Matlab;
  PetscFunctionReturn(0);
}
EXTERN_C_END





