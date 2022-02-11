
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/
#include <petscmatlab.h>   /*I  "petscmatlab.h"  I*/

/*
        This PF generates a MATLAB function on the fly
*/
typedef struct {
  PetscInt          dimin,dimout;
  PetscMatlabEngine mengine;
  char              *string;
} PF_Matlab;

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

PetscErrorCode PFApply_Matlab(void *value,PetscInt n,const PetscScalar *in,PetscScalar *out)
{
  PF_Matlab      *matlab = (PF_Matlab*)value;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!value,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Need to set string for MATLAB function, via -pf_matlab string");
  ierr = PetscMatlabEnginePutArray(matlab->mengine,matlab->dimin,n,in,"x");CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(matlab->mengine,matlab->string);CHKERRQ(ierr);
  ierr = PetscMatlabEngineGetArray(matlab->mengine,matlab->dimout,n,out,"f");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PFSetFromOptions_Matlab(PetscOptionItems *PetscOptionsObject,PF pf)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  char           value[256];
  PF_Matlab      *matlab = (PF_Matlab*)pf->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Matlab function options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-pf_matlab","Matlab function","None","",value,sizeof(value),&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PFCreate_Matlab(PF pf,void *value)
{
  PetscErrorCode ierr;
  PF_Matlab      *matlab;

  PetscFunctionBegin;
  ierr           = PetscNewLog(pf,&matlab);CHKERRQ(ierr);
  matlab->dimin  = pf->dimin;
  matlab->dimout = pf->dimout;

  ierr = PetscMatlabEngineCreate(PetscObjectComm((PetscObject)pf),NULL,&matlab->mengine);CHKERRQ(ierr);

  if (value) {
    ierr = PetscStrallocpy((char*)value,&matlab->string);CHKERRQ(ierr);
  }
  ierr = PFSet(pf,PFApply_Matlab,NULL,PFView_Matlab,PFDestroy_Matlab,matlab);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Matlab;
  PetscFunctionReturn(0);
}

