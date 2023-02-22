
#include <../src/vec/pf/pfimpl.h> /*I "petscpf.h" I*/
#include <petscmatlab.h>          /*I  "petscmatlab.h"  I*/

/*
        This PF generates a MATLAB function on the fly
*/
typedef struct {
  PetscInt          dimin, dimout;
  PetscMatlabEngine mengine;
  char             *string;
} PF_Matlab;

PetscErrorCode PFView_Matlab(void *value, PetscViewer viewer)
{
  PetscBool  iascii;
  PF_Matlab *matlab = (PF_Matlab *)value;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "Matlab Matlab = %s\n", matlab->string));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PFDestroy_Matlab(void *value)
{
  PF_Matlab *matlab = (PF_Matlab *)value;

  PetscFunctionBegin;
  PetscCall(PetscFree(matlab->string));
  PetscCall(PetscMatlabEngineDestroy(&matlab->mengine));
  PetscCall(PetscFree(matlab));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PFApply_Matlab(void *value, PetscInt n, const PetscScalar *in, PetscScalar *out)
{
  PF_Matlab *matlab = (PF_Matlab *)value;

  PetscFunctionBegin;
  PetscCheck(value, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Need to set string for MATLAB function, via -pf_matlab string");
  PetscCall(PetscMatlabEnginePutArray(matlab->mengine, matlab->dimin, n, in, "x"));
  PetscCall(PetscMatlabEngineEvaluate(matlab->mengine, matlab->string));
  PetscCall(PetscMatlabEngineGetArray(matlab->mengine, matlab->dimout, n, out, "f"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PFSetFromOptions_Matlab(PF pf, PetscOptionItems *PetscOptionsObject)
{
  PetscBool  flag;
  char       value[256];
  PF_Matlab *matlab = (PF_Matlab *)pf->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Matlab function options");
  PetscCall(PetscOptionsString("-pf_matlab", "Matlab function", "None", "", value, sizeof(value), &flag));
  if (flag) PetscCall(PetscStrallocpy((char *)value, &matlab->string));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PFCreate_Matlab(PF pf, void *value)
{
  PF_Matlab *matlab;

  PetscFunctionBegin;
  PetscCall(PetscNew(&matlab));
  matlab->dimin  = pf->dimin;
  matlab->dimout = pf->dimout;

  PetscCall(PetscMatlabEngineCreate(PetscObjectComm((PetscObject)pf), NULL, &matlab->mengine));

  if (value) PetscCall(PetscStrallocpy((char *)value, &matlab->string));
  PetscCall(PFSet(pf, PFApply_Matlab, NULL, PFView_Matlab, PFDestroy_Matlab, matlab));

  pf->ops->setfromoptions = PFSetFromOptions_Matlab;
  PetscFunctionReturn(PETSC_SUCCESS);
}
