
#include <private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMSetFromOptions_DA"
PetscErrorCode  DMSetFromOptions_DA(DM da)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  char           typeName[256];
  DM_DA          *dd = (DM_DA*)da->data;
  PetscInt       refine = 0;
  PetscBool      negativeMNP = PETSC_FALSE,bM = PETSC_FALSE,bN = PETSC_FALSE, bP = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);

  if (dd->M < 0) {
    dd->M       = -dd->M;
    bM          = PETSC_TRUE;
    negativeMNP = PETSC_TRUE;
  }
  if (dd->dim > 1 && dd->N < 0) {
    dd->N       = -dd->N;
    bN          = PETSC_TRUE;
    negativeMNP = PETSC_TRUE;
  }
  if (dd->dim > 2 && dd->P < 0) {
    dd->P       = -dd->P;
    bP          = PETSC_TRUE;
    negativeMNP = PETSC_TRUE;
  }

  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DMDA Options","DMDA");CHKERRQ(ierr);
    if (bM) {ierr = PetscOptionsInt("-da_grid_x","Number of grid points in x direction","DMDASetSizes",dd->M,&dd->M,PETSC_NULL);CHKERRQ(ierr);}
    if (bN) {ierr = PetscOptionsInt("-da_grid_y","Number of grid points in y direction","DMDASetSizes",dd->N,&dd->N,PETSC_NULL);CHKERRQ(ierr);}
    if (bP) {ierr = PetscOptionsInt("-da_grid_z","Number of grid points in z direction","DMDASetSizes",dd->P,&dd->P,PETSC_NULL);CHKERRQ(ierr);}
    /* Handle DMDA parallel distibution */
    ierr = PetscOptionsInt("-da_processors_x","Number of processors in x direction","DMDASetNumProcs",dd->m,&dd->m,PETSC_NULL);CHKERRQ(ierr);
    if (dd->dim > 1) {ierr = PetscOptionsInt("-da_processors_y","Number of processors in y direction","DMDASetNumProcs",dd->n,&dd->n,PETSC_NULL);CHKERRQ(ierr);}
    if (dd->dim > 2) {ierr = PetscOptionsInt("-da_processors_z","Number of processors in z direction","DMDASetNumProcs",dd->p,&dd->p,PETSC_NULL);CHKERRQ(ierr);}
    /* Handle DMDA refinement */
    ierr = PetscOptionsInt("-da_refine_x","Refinement ratio in x direction","DMDASetRefinementFactor",dd->refine_x,&dd->refine_x,PETSC_NULL);CHKERRQ(ierr);
    if (dd->dim > 1) {ierr = PetscOptionsInt("-da_refine_y","Refinement ratio in y direction","DMDASetRefinementFactor",dd->refine_y,&dd->refine_y,PETSC_NULL);CHKERRQ(ierr);}
    if (dd->dim > 2) {ierr = PetscOptionsInt("-da_refine_z","Refinement ratio in z direction","DMDASetRefinementFactor",dd->refine_z,&dd->refine_z,PETSC_NULL);CHKERRQ(ierr);}

    if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-da_vec_type","Vector type used for created vectors","DMSetVecType",VecList,da->vectype,typeName,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMSetVecType(da,typeName);CHKERRQ(ierr);
    }
    if (negativeMNP) {ierr = PetscOptionsInt("-da_refine","Uniformly refine DA one or more times","None",refine,&refine,PETSC_NULL);CHKERRQ(ierr);}

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)da);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  while (refine--) {
    if (dd->bx == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
      dd->M = dd->refine_x*dd->M;
    } else {
      dd->M = 1 + dd->refine_x*(dd->M - 1);
    }
    if (dd->by == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
      dd->N = dd->refine_y*dd->N;
    } else {
      dd->N = 1 + dd->refine_y*(dd->N - 1);
    }
    if (dd->bz == DMDA_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0){
      dd->P = dd->refine_z*dd->P;
    } else {
      dd->P = 1 + dd->refine_z*(dd->P - 1);
    }
    da->levelup++;
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode  DMCreateGlobalVector_DA(DM,Vec*);
extern PetscErrorCode  DMCreateLocalVector_DA(DM,Vec*);
extern PetscErrorCode  DMGlobalToLocalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMGlobalToLocalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToGlobalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToGlobalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMGetInterpolation_DA(DM,DM,Mat*,Vec*);
extern PetscErrorCode  DMGetColoring_DA(DM,ISColoringType,const MatType,ISColoring*);
extern PetscErrorCode  DMGetMatrix_DA(DM,const MatType,Mat*);
extern PetscErrorCode  DMRefine_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode  DMCoarsen_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode  DMRefineHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode  DMCoarsenHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode  DMGetInjection_DA(DM,DM,VecScatter*);
extern PetscErrorCode  DMGetAggregates_DA(DM,DM,Mat*);
extern PetscErrorCode  DMView_DA(DM,PetscViewer);
extern PetscErrorCode  DMSetUp_DA(DM);
extern PetscErrorCode  DMDestroy_DA(DM);

#undef __FUNCT__  
#define __FUNCT__ "DMLoad_DA"
PetscErrorCode DMLoad_DA(DM da,PetscViewer viewer)
{
  PetscErrorCode   ierr;
  PetscInt         dim,m,n,p,dof,swidth,M,N,P;
  DMDAStencilType  stencil;
  DMDABoundaryType bx,by,bz;
  PetscInt         classid = DM_FILE_CLASSID,subclassid = DMDA_FILE_CLASSID ;

  PetscFunctionBegin;  
  ierr = PetscViewerBinaryRead(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
  if (classid != DM_FILE_CLASSID) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Not DM next in file");
  ierr = PetscViewerBinaryRead(viewer,&subclassid,1,PETSC_INT);CHKERRQ(ierr);
  if (subclassid != DMDA_FILE_CLASSID) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_ARG_WRONG,"Not DM DA next in file");
  ierr = PetscViewerBinaryRead(viewer,&dim,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&m,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&n,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&p,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&dof,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&swidth,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&bx,1,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&by,1,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&bz,1,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&stencil,1,PETSC_ENUM);CHKERRQ(ierr);

  ierr = DMDASetDim(da, 3);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, M, N, P);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, bx, by, bz);CHKERRQ(ierr);
  ierr = DMDASetDof(da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da, stencil);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, swidth);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "DMCreate_DA"
PetscErrorCode  DMCreate_DA(DM da)
{
  PetscErrorCode ierr;
  DM_DA          *dd;

  PetscFunctionBegin;
  PetscValidPointer(da,1);
  ierr = PetscNewLog(da,DM_DA,&dd);CHKERRQ(ierr);
  da->data = dd;

  dd->dim        = -1;
  dd->interptype = DMDA_Q1;
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
  dd->ao           = PETSC_NULL;
  dd->base         = -1;
  dd->bx         = DMDA_BOUNDARY_NONE;
  dd->by         = DMDA_BOUNDARY_NONE;
  dd->bz         = DMDA_BOUNDARY_NONE;
  dd->stencil_type = DMDA_STENCIL_BOX;
  dd->interptype   = DMDA_Q1;
  dd->idx          = PETSC_NULL;
  dd->Nl           = -1;
  dd->lx           = PETSC_NULL;
  dd->ly           = PETSC_NULL;
  dd->lz           = PETSC_NULL;

  dd->elementtype  = DMDA_ELEMENT_Q1;

  ierr = PetscStrallocpy(VECSTANDARD,&da->vectype);CHKERRQ(ierr);
  da->ops->globaltolocalbegin = DMGlobalToLocalBegin_DA;
  da->ops->globaltolocalend   = DMGlobalToLocalEnd_DA;
  da->ops->localtoglobalbegin = DMLocalToGlobalBegin_DA;
  da->ops->localtoglobalend   = DMLocalToGlobalEnd_DA;
  da->ops->createglobalvector = DMCreateGlobalVector_DA;
  da->ops->createlocalvector  = DMCreateLocalVector_DA;
  da->ops->getinterpolation   = DMGetInterpolation_DA;
  da->ops->getcoloring        = DMGetColoring_DA;
  da->ops->getmatrix          = DMGetMatrix_DA;
  da->ops->refine             = DMRefine_DA;
  da->ops->coarsen            = DMCoarsen_DA;
  da->ops->refinehierarchy    = DMRefineHierarchy_DA;
  da->ops->coarsenhierarchy   = DMCoarsenHierarchy_DA;
  da->ops->getinjection       = DMGetInjection_DA;
  da->ops->getaggregates      = DMGetAggregates_DA;
  da->ops->destroy            = DMDestroy_DA;
  da->ops->view               = 0;
  da->ops->setfromoptions     = DMSetFromOptions_DA;
  da->ops->setup              = DMSetUp_DA;
  da->ops->load               = DMLoad_DA;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DMDACreate"
/*@
  DMDACreate - Creates a DMDA object. 

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMDA object

  Output Parameter:
. da  - The DMDA object

  Level: advanced

  Developers Note: Since there exists DMDACreate1/2/3d() should this routine even exist?

.keywords: DMDA, create
.seealso:  DMDASetSizes(), DMDADuplicate(),  DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode  DMDACreate(MPI_Comm comm, DM *da)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(da,2);
  ierr = DMCreate(comm,da);CHKERRQ(ierr);
  ierr = DMSetType(*da,DMDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


