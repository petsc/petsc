
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/

static PetscErrorCode PFApply_Constant(void *value,PetscInt n,const PetscScalar *x,PetscScalar *y)
{
  PetscInt    i;
  PetscScalar v = ((PetscScalar*)value)[0];

  PetscFunctionBegin;
  n *= (PetscInt) PetscRealPart(((PetscScalar*)value)[1]);
  for (i=0; i<n; i++) y[i] = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode PFApplyVec_Constant(void *value,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,*((PetscScalar*)value));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode PFView_Constant(void *value,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"Constant = %g\n",*(double*)value);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"Constant = %g + %gi\n",PetscRealPart(*(PetscScalar*)value),PetscImaginaryPart(*(PetscScalar*)value));CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PFDestroy_Constant(void *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PFSetFromOptions_Constant(PetscOptionItems *PetscOptionsObject,PF pf)
{
  PetscErrorCode ierr;
  PetscScalar    *value = (PetscScalar*)pf->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Constant function options");CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-pf_constant","The constant value","None",*value,value,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PFCreate_Constant(PF pf,void *value)
{
  PetscErrorCode ierr;
  PetscScalar    *loc;

  PetscFunctionBegin;
  ierr = PetscMalloc1(2,&loc);CHKERRQ(ierr);
  if (value) loc[0] = *(PetscScalar*)value;
  else loc[0] = 0.0;
  loc[1] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Constant,PFApplyVec_Constant,PFView_Constant,PFDestroy_Constant,loc);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_Constant;
  PetscFunctionReturn(0);
}

/*typedef PetscErrorCode (*FCN)(void*,PetscInt,const PetscScalar*,PetscScalar*);  force argument to next function to not be extern C*/

PETSC_EXTERN PetscErrorCode PFCreate_Quick(PF pf,PetscErrorCode (*function)(void*,PetscInt,const PetscScalar*,PetscScalar*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PFSet(pf,function,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------------------------------*/
static PetscErrorCode PFApply_Identity(void *value,PetscInt n,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i;

  PetscFunctionBegin;
  n *= *(PetscInt*)value;
  for (i=0; i<n; i++) y[i] = x[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode PFApplyVec_Identity(void *value,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PFView_Identity(void *value,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Identity function\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PFDestroy_Identity(void *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PFCreate_Identity(PF pf,void *value)
{
  PetscErrorCode ierr;
  PetscInt       *loc;

  PetscFunctionBegin;
  if (pf->dimout != pf->dimin) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Input dimension must match output dimension for Identity function, dimin = %" PetscInt_FMT " dimout = %" PetscInt_FMT "\n",pf->dimin,pf->dimout);
  ierr   = PetscNew(&loc);CHKERRQ(ierr);
  loc[0] = pf->dimout;
  ierr   = PFSet(pf,PFApply_Identity,PFApplyVec_Identity,PFView_Identity,PFDestroy_Identity,loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
