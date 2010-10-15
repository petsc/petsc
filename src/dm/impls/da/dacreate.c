#define PETSCDM_DLL
#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMSetFromOptions_DA"
PetscErrorCode PETSCDM_DLLEXPORT DMSetFromOptions_DA(DM da)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  char           typeName[256];
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);

  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA Options","DA");CHKERRQ(ierr);
    /* Handle DA grid sizes */
    if (dd->M < 0) {
      PetscInt newM = -dd->M;
      ierr = PetscOptionsInt("-da_grid_x","Number of grid points in x direction","DASetSizes",newM,&newM,PETSC_NULL);CHKERRQ(ierr);
      dd->M = newM;
    }
    if (dd->dim > 1 && dd->N < 0) {
      PetscInt newN = -dd->N;
      ierr = PetscOptionsInt("-da_grid_y","Number of grid points in y direction","DASetSizes",newN,&newN,PETSC_NULL);CHKERRQ(ierr);
      dd->N = newN;
    }
    if (dd->dim > 2 && dd->P < 0) {
      PetscInt newP = -dd->P;
      ierr = PetscOptionsInt("-da_grid_z","Number of grid points in z direction","DASetSizes",newP,&newP,PETSC_NULL);CHKERRQ(ierr);
      dd->P = newP;
    }
    /* Handle DA parallel distibution */
    ierr = PetscOptionsInt("-da_processors_x","Number of processors in x direction","DASetNumProcs",dd->m,&dd->m,PETSC_NULL);CHKERRQ(ierr);
    if (dd->dim > 1) {ierr = PetscOptionsInt("-da_processors_y","Number of processors in y direction","DASetNumProcs",dd->n,&dd->n,PETSC_NULL);CHKERRQ(ierr);}
    if (dd->dim > 2) {ierr = PetscOptionsInt("-da_processors_z","Number of processors in z direction","DASetNumProcs",dd->p,&dd->p,PETSC_NULL);CHKERRQ(ierr);}
    /* Handle DA refinement */
    ierr = PetscOptionsInt("-da_refine_x","Refinement ratio in x direction","DASetRefinementFactor",dd->refine_x,&dd->refine_x,PETSC_NULL);CHKERRQ(ierr);
    if (dd->dim > 1) {ierr = PetscOptionsInt("-da_refine_y","Refinement ratio in y direction","DASetRefinementFactor",dd->refine_y,&dd->refine_y,PETSC_NULL);CHKERRQ(ierr);}
    if (dd->dim > 2) {ierr = PetscOptionsInt("-da_refine_z","Refinement ratio in z direction","DASetRefinementFactor",dd->refine_z,&dd->refine_z,PETSC_NULL);CHKERRQ(ierr);}

    if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-da_vec_type","Vector type used for created vectors","DMSetVecType",VecList,da->vectype,typeName,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMSetVecType(da,typeName);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)da);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

extern PetscErrorCode PETSCDM_DLLEXPORT DMCreateGlobalVector_DA(DM,Vec*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMCreateLocalVector_DA(DM,Vec*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGlobalToLocalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGlobalToLocalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGetInterpolation_DA(DM,DM,Mat*,Vec*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGetColoring_DA(DM,ISColoringType,const MatType,ISColoring*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGetMatrix_DA(DM,const MatType,Mat*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMRefine_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMCoarsen_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMRefineHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode PETSCDM_DLLEXPORT DMCoarsenHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGetInjection_DA(DM,DM,VecScatter*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMGetAggregates_DA(DM,DM,Mat*);
extern PetscErrorCode PETSCDM_DLLEXPORT DMView_DA(DM,PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT DMSetUp_DA(DM);
extern PetscErrorCode PETSCDM_DLLEXPORT DMDestroy_DA(DM);

#undef __FUNCT__  
#define __FUNCT__ "DACreate"
/*@
  DACreate - Creates a DA object. 

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DA object

  Output Parameter:
. da  - The DA object

  Level: beginner

.keywords: DA, create
.seealso:  DASetSizes(), DADuplicate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreate(MPI_Comm comm, DM *da)
{
  DM             d;
  PetscErrorCode ierr;
  DM_DA          *dd;

  PetscFunctionBegin;
  PetscValidPointer(da,2);
  *da = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(d, _p_DM, struct _DMOps, DM_CLASSID, 0, "DM", comm, DMDestroy, DMView);CHKERRQ(ierr);
  ierr = PetscMemzero(d->ops, sizeof(struct _DMOps));CHKERRQ(ierr);
  ierr = PetscNewLog(d,DM_DA,&dd);CHKERRQ(ierr);
  d->data = dd;

  dd->dim        = -1;
  dd->interptype = DA_Q1;
  dd->refine_x   = 2;
  dd->refine_y   = 2;
  dd->refine_z   = 2;
  dd->fieldname  = PETSC_NULL;
  dd->nlocal     = -1;
  dd->Nlocal     = -1;
  dd->M          = -1;
  dd->N          = -1;
  dd->P          = -1;
  dd->m          = -1;
  dd->n          = -1;
  dd->p          = -1;
  dd->w          = -1;
  dd->s          = -1;
  dd->xs = -1; dd->xe = -1; dd->ys = -1; dd->ye = -1; dd->zs = -1; dd->ze = -1;
  dd->Xs = -1; dd->Xe = -1; dd->Ys = -1; dd->Ye = -1; dd->Zs = -1; dd->Ze = -1;

  dd->gtol         = PETSC_NULL;
  dd->ltog         = PETSC_NULL;
  dd->ltol         = PETSC_NULL;
  dd->ltogmap      = PETSC_NULL;
  dd->ltogmapb     = PETSC_NULL;
  dd->ao           = PETSC_NULL;
  dd->base         = -1;
  dd->wrap         = DA_NONPERIODIC;
  dd->stencil_type = DA_STENCIL_BOX;
  dd->interptype   = DA_Q1;
  dd->idx          = PETSC_NULL;
  dd->Nl           = -1;
  dd->lx           = PETSC_NULL;
  dd->ly           = PETSC_NULL;
  dd->lz           = PETSC_NULL;

  ierr = PetscStrallocpy(VECSTANDARD,&d->vectype);CHKERRQ(ierr);
  d->ops->globaltolocalbegin = DMGlobalToLocalBegin_DA;
  d->ops->globaltolocalend   = DMGlobalToLocalEnd_DA;
  d->ops->localtoglobalbegin = DMLocalToGlobalBegin_DA;
  d->ops->localtoglobalend   = DMLocalToGlobalEnd_DA;
  d->ops->createglobalvector = DMCreateGlobalVector_DA;
  d->ops->createlocalvector  = DMCreateLocalVector_DA;
  d->ops->getinterpolation   = DMGetInterpolation_DA;
  d->ops->getcoloring        = DMGetColoring_DA;
  d->ops->getmatrix          = DMGetMatrix_DA;
  d->ops->refine             = DMRefine_DA;
  d->ops->coarsen            = DMCoarsen_DA;
  d->ops->refinehierarchy    = DMRefineHierarchy_DA;
  d->ops->coarsenhierarchy   = DMCoarsenHierarchy_DA;
  d->ops->getinjection       = DMGetInjection_DA;
  d->ops->getaggregates      = DMGetAggregates_DA;
  d->ops->destroy            = DMDestroy_DA;
  d->ops->view               = 0;
  d->ops->setfromoptions     = DMSetFromOptions_DA;
  d->ops->setup              = DMSetUp_DA;

  *da = d; 
  PetscFunctionReturn(0);
}
