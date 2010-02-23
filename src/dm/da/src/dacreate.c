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
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (da->dim) {
    case 1: defaultType = DA1D; break;
    case 2: defaultType = DA2D; break;
    case 3: defaultType = DA3D; break;
  }
  if (((PetscObject)da)->type_name) {
    defaultType = ((PetscObject)da)->type_name;
  }
  if (!DARegisterAllCalled) {ierr = DARegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsList("-da_type","DA type","DASetType",DAList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
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

  Notes:  To see all options, run your program with the -help option, or consult the users manual.
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
  PetscInt       dim;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);

  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA Options","DA");CHKERRQ(ierr);
    /* Handle DA dimensions */
    ierr = PetscOptionsInt("-da_dim","Number of dimensions","DASetDim",da->dim,&dim,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DASetDim(da,dim);CHKERRQ(ierr);
    }
    /* Handle DA grid sizes */
    if (da->M < 0) {
      PetscInt newM = -da->M;
      ierr = PetscOptionsInt("-da_grid_x","Number of grid points in x direction","DASetSizes",newM,&newM,PETSC_NULL);CHKERRQ(ierr);
      da->M = newM;
    }
    if (da->N < 0) {
      PetscInt newN = -da->N;
      ierr = PetscOptionsInt("-da_grid_y","Number of grid points in y direction","DASetSizes",newN,&newN,PETSC_NULL);CHKERRQ(ierr);
      da->N = newN;
    }
    if (da->P < 0) {
      PetscInt newP = -da->P;
      ierr = PetscOptionsInt("-da_grid_z","Number of grid points in z direction","DASetSizes",newP,&newP,PETSC_NULL);CHKERRQ(ierr);
      da->P = newP;
    }
    /* Handle DA parallel distibution */
    ierr = PetscOptionsInt("-da_processors_x","Number of processors in x direction","DASetNumProcs",da->m,&da->m,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_processors_y","Number of processors in y direction","DASetNumProcs",da->n,&da->n,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_processors_z","Number of processors in z direction","DASetNumProcs",da->p,&da->p,PETSC_NULL);CHKERRQ(ierr);
    /* Handle DA refinement */
    ierr = PetscOptionsInt("-da_refine_x","Refinement ratio in x direction","DASetRefinementFactor",da->refine_x,&da->refine_x,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_refine_y","Refinement ratio in y direction","DASetRefinementFactor",da->refine_y,&da->refine_y,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-da_refine_z","Refinement ratio in z direction","DASetRefinementFactor",da->refine_z,&da->refine_z,PETSC_NULL);CHKERRQ(ierr);
    /* Handle DA type options */
    ierr = DASetTypeFromOptions_Private(da);CHKERRQ(ierr);
    /* Handle specific DA options */
    if (da->ops->setfromoptions) {
      ierr = (*da->ops->setfromoptions)(da);CHKERRQ(ierr);
    }
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

  PetscFunctionBegin;
  PetscValidPointer(da,2);
  *da = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(d, _p_DA, struct _DAOps, DM_COOKIE, 0, "DM", comm, DADestroy, DAView);CHKERRQ(ierr);
  ierr = PetscMemzero(d->ops, sizeof(struct _DAOps));CHKERRQ(ierr);

  d->dim        = -1;
  d->interptype = DA_Q1;
  d->refine_x   = 2;
  d->refine_y   = 2;
  d->refine_z   = 2;
  d->fieldname  = PETSC_NULL;
  d->nlocal     = -1;
  d->Nlocal     = -1;
  d->M          = -1;
  d->N          = -1;
  d->P          = -1;
  d->m          = -1;
  d->n          = -1;
  d->p          = -1;
  d->w          = -1;
  d->s          = -1;
  d->xs = -1; d->xe = -1; d->ys = -1; d->ye = -1; d->zs = -1; d->ze = -1;
  d->Xs = -1; d->Xe = -1; d->Ys = -1; d->Ye = -1; d->Zs = -1; d->Ze = -1;

  d->gtol         = PETSC_NULL;
  d->ltog         = PETSC_NULL;
  d->ltol         = PETSC_NULL;
  d->ltogmap      = PETSC_NULL;
  d->ltogmapb     = PETSC_NULL;
  d->ao           = PETSC_NULL;
  d->base         = -1;
  d->wrap         = DA_NONPERIODIC;
  d->stencil_type = DA_STENCIL_BOX;
  d->interptype   = DA_Q1;
  d->idx          = PETSC_NULL;
  d->Nl           = -1;
  d->lx           = PETSC_NULL;
  d->ly           = PETSC_NULL;
  d->lz           = PETSC_NULL;

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

  *da = d; 
  PetscFunctionReturn(0);
}
