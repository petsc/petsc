#define PETSCDM_DLL
#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef  __FUNCT__
#define __FUNCT__ "DAViewFromOptions"
/*@
  DAViewFromOptions - This function visualizes the DA based upon user options.

  Collective on DA

  Input Parameters:
+ da   - The DA
- title - The title (currently ignored)

  Level: intermediate

.keywords: DA, view, options, database
.seealso: DASetFromOptions(), DAView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAViewFromOptions(DA da, const char title[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAView_Private(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetTypeFromOptions_Private"
/*
  DASetTypeFromOptions_Private - Sets the type of DA from user options. Defaults to a 1D DA.

  Collective on Vec

  Input Parameter:
. da - The DA

  Level: intermediate

.keywords: DA, set, options, database, type
.seealso: DASetFromOptions(), DASetType()
*/
static PetscErrorCode DASetTypeFromOptions_Private(DA da)
{
  const DAType   defaultType = DA1D;
  char           typeName[256];
  PetscBool      opt = PETSC_FALSE;
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  switch (dd->dim) {
    case 1: defaultType = DA1D; break;
    case 2: defaultType = DA2D; break;
    case 3: defaultType = DA3D; break;
  }
  if (((PetscObject)da)->type_name) {
    defaultType = ((PetscObject)da)->type_name;
  }
  if (!DARegisterAllCalled) {ierr = DARegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  if (dd->dim == PETSC_DECIDE) {
    ierr = PetscOptionsList("-da_type","DA type","DASetType",DAList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  }
  if (opt) {
    ierr = DASetType(da, typeName);CHKERRQ(ierr);
  } else {
    ierr = DASetType(da, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetFromOptions"
/*@
  DASetFromOptions - Configures the vector from the options database.

  Collective on DA

  Input Parameter:
. da - The DA

  Notes:  To see all options, run your program with the -help option, or consult the <A href="../../docs/manual.pdf">Users Manual</A>.
          Must be called after DACreate() but before the DA is used.

  Level: beginner

  Concepts: DA^setting options
  Concepts: DA^setting type

.keywords: DA, set, options, database
.seealso: DACreate(), DASetOptionsPrefix()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetFromOptions(DA da)
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
    /* Handle DA type options; only makes sense to call if dimension has not yet been set  */
    ierr = DASetTypeFromOptions_Private(da);CHKERRQ(ierr);

    if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-da_vec_type","Vector type used for created vectors","DASetVecType",VecList,da->vectype,typeName,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DASetVecType(da,typeName);CHKERRQ(ierr);
    }
   
    /* Handle specific DA options */
    if (da->ops->setfromoptions) {
      ierr = (*da->ops->setfromoptions)(da);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)da);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DAViewFromOptions(da, ((PetscObject)da)->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACreate"
/*@
  DACreate - Creates an empty DA object. The type can then be set with DASetType(),
  or DASetFromOptions().

   If you never  call DASetType() or DASetFromOptions() it will generate an 
   error when you try to use the DA.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DA object

  Output Parameter:
. da  - The DA object

  Level: beginner

.keywords: DA, create
.seealso: DASetType(), DASetSizes(), DADuplicate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreate(MPI_Comm comm, DA *da)
{
  DA             d;
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
  d->ops->globaltolocalbegin = DAGlobalToLocalBegin;
  d->ops->globaltolocalend   = DAGlobalToLocalEnd;
  d->ops->localtoglobal      = DALocalToGlobal;
  d->ops->createglobalvector = DACreateGlobalVector;
  d->ops->createlocalvector  = DACreateLocalVector;
  d->ops->getinterpolation   = DAGetInterpolation;
  d->ops->getcoloring        = DAGetColoring;
  d->ops->getmatrix          = DAGetMatrix;
  d->ops->refine             = DARefine;
  d->ops->coarsen            = DACoarsen;
  d->ops->refinehierarchy    = DARefineHierarchy;
  d->ops->coarsenhierarchy   = DACoarsenHierarchy;
  d->ops->getinjection       = DAGetInjection;
  d->ops->getaggregates      = DAGetAggregates;
  d->ops->destroy            = DADestroy;
  d->ops->view               = DAView;

  *da = d; 
  PetscFunctionReturn(0);
}
