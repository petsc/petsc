
#include <../src/vec/pf/pfimpl.h> /*I "petscpf.h" I*/

static PetscErrorCode PFApply_Constant(void *value, PetscInt n, const PetscScalar *x, PetscScalar *y)
{
  PetscInt    i;
  PetscScalar v = ((PetscScalar *)value)[0];

  PetscFunctionBegin;
  n *= (PetscInt)PetscRealPart(((PetscScalar *)value)[1]);
  for (i = 0; i < n; i++) y[i] = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFApplyVec_Constant(void *value, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecSet(y, *((PetscScalar *)value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode PFView_Constant(void *value, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer, "Constant = %g\n", *(double *)value));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "Constant = %g + %gi\n", (double)PetscRealPart(*(PetscScalar *)value), (double)PetscImaginaryPart(*(PetscScalar *)value)));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PFDestroy_Constant(void *value)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFSetFromOptions_Constant(PF pf, PetscOptionItems *PetscOptionsObject)
{
  PetscScalar *value = (PetscScalar *)pf->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Constant function options");
  PetscCall(PetscOptionsScalar("-pf_constant", "The constant value", "None", *value, value, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PFCreate_Constant(PF pf, void *value)
{
  PetscScalar *loc;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(2, &loc));
  if (value) loc[0] = *(PetscScalar *)value;
  else loc[0] = 0.0;
  loc[1] = pf->dimout;
  PetscCall(PFSet(pf, PFApply_Constant, PFApplyVec_Constant, PFView_Constant, PFDestroy_Constant, loc));

  pf->ops->setfromoptions = PFSetFromOptions_Constant;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*typedef PetscErrorCode (*FCN)(void*,PetscInt,const PetscScalar*,PetscScalar*);  force argument to next function to not be extern C*/

PETSC_EXTERN PetscErrorCode PFCreate_Quick(PF pf, PetscErrorCode (*function)(void *, PetscInt, const PetscScalar *, PetscScalar *))
{
  PetscFunctionBegin;
  PetscCall(PFSet(pf, function, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------------------------------------------------------------------------------------------------------------------*/
static PetscErrorCode PFApply_Identity(void *value, PetscInt n, const PetscScalar *x, PetscScalar *y)
{
  PetscInt i;

  PetscFunctionBegin;
  n *= *(PetscInt *)value;
  for (i = 0; i < n; i++) y[i] = x[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PFApplyVec_Identity(void *value, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PFView_Identity(void *value, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "Identity function\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PFDestroy_Identity(void *value)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PFCreate_Identity(PF pf, void *value)
{
  PetscInt *loc;

  PetscFunctionBegin;
  PetscCheck(pf->dimout == pf->dimin, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Input dimension must match output dimension for Identity function, dimin = %" PetscInt_FMT " dimout = %" PetscInt_FMT, pf->dimin, pf->dimout);
  PetscCall(PetscNew(&loc));
  loc[0] = pf->dimout;
  PetscCall(PFSet(pf, PFApply_Identity, PFApplyVec_Identity, PFView_Identity, PFDestroy_Identity, loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
