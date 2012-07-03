
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/

/*
        Ths PF generates a MATLAB function on the fly
*/
typedef struct {
  PetscInt          dimin,dimout;
  PetscMatlabEngine mengine;
  char              *string;
} PF_Matlab;
  
#undef __FUNCT__  
#define __FUNCT__ "PFView_Matlab"
PetscErrorCode PFView_Matlab(void *value,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PF_Matlab      *matlab = (PF_Matlab*)value;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Matlab Matlab = %s\n",matlab->string);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFDestroy_Matlab"
PetscErrorCode PFDestroy_Matlab(void *value)
{
  PetscErrorCode ierr;
  PF_Matlab      *matlab = (PF_Matlab*)value;

  PetscFunctionBegin;
  ierr = PetscFree(matlab->string);CHKERRQ(ierr);
  ierr = PetscMatlabEngineDestroy(&matlab->mengine);CHKERRQ(ierr);
  ierr = PetscFree(matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFApply_Matlab"
PetscErrorCode PFApply_Matlab(void *value,PetscInt n,const PetscScalar *in,PetscScalar *out)
{
  PF_Matlab      *matlab = (PF_Matlab*)value;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!value) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Need to set string for MATLAB function, via -pf_matlab string");
  ierr = PetscMatlabEnginePutArray(matlab->mengine,matlab->dimin,n,in,"x");CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(matlab->mengine,matlab->string);CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(matlab->mengine,matlab->dimout,n,out,"f");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFSetFromOptions_Matlab"
PetscErrorCode PFSetFromOptions_Matlab(PF pf)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  char           value[256];
  PF_Matlab      *matlab = (PF_Matlab*)pf->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Matlab function options");CHKERRQ(ierr);
    ierr = PetscOptionsString("-pf_matlab","Matlab function","None","",value,256,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PFCreate_Matlab"
PetscErrorCode  PFCreate_Matlab(PF pf,void *value)
{
  PetscErrorCode ierr;
  PF_Matlab      *matlab;

  PetscFunctionBegin;
  ierr           = PetscNewLog(pf,PF_Matlab,&matlab);CHKERRQ(ierr);
  matlab->dimin  = pf->dimin;
  matlab->dimout = pf->dimout;

  ierr = PetscMatlabEngineCreate(((PetscObject)pf)->comm,PETSC_NULL,&matlab->mengine);CHKERRQ(ierr);
    
  if (value) {
    ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
  }
  ierr   = PFSet(pf,PFApply_Matlab,PETSC_NULL,PFView_Matlab,PFDestroy_Matlab,matlab);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Matlab;
  PetscFunctionReturn(0);
}
EXTERN_C_END





