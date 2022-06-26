
#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  DMSetFromOptions_DA(PetscOptionItems *PetscOptionsObject,DM da)
{
  DM_DA          *dd    = (DM_DA*)da->data;
  PetscInt       refine = 0,dim = da->dim,maxnlevels = 100,refx[100],refy[100],refz[100],n,i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,2);

  PetscCheck(dd->M >= 0,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  PetscCheck(dd->N >= 0,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  PetscCheck(dd->P >= 0,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");

  PetscOptionsHeadBegin(PetscOptionsObject,"DMDA Options");
  PetscCall(PetscOptionsBoundedInt("-da_grid_x","Number of grid points in x direction","DMDASetSizes",dd->M,&dd->M,NULL,1));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_grid_y","Number of grid points in y direction","DMDASetSizes",dd->N,&dd->N,NULL,1));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_grid_z","Number of grid points in z direction","DMDASetSizes",dd->P,&dd->P,NULL,1));

  PetscCall(PetscOptionsBoundedInt("-da_overlap","Decomposition overlap in all directions","DMDASetOverlap",dd->xol,&dd->xol,&flg,0));
  if (flg) PetscCall(DMDASetOverlap(da,dd->xol,dd->xol,dd->xol));
  PetscCall(PetscOptionsBoundedInt("-da_overlap_x","Decomposition overlap in x direction","DMDASetOverlap",dd->xol,&dd->xol,NULL,0));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_overlap_y","Decomposition overlap in y direction","DMDASetOverlap",dd->yol,&dd->yol,NULL,0));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_overlap_z","Decomposition overlap in z direction","DMDASetOverlap",dd->zol,&dd->zol,NULL,0));

  PetscCall(PetscOptionsBoundedInt("-da_local_subdomains","","DMDASetNumLocalSubdomains",dd->Nsub,&dd->Nsub,&flg,PETSC_DECIDE));
  if (flg) PetscCall(DMDASetNumLocalSubDomains(da,dd->Nsub));

  /* Handle DMDA parallel distribution */
  PetscCall(PetscOptionsBoundedInt("-da_processors_x","Number of processors in x direction","DMDASetNumProcs",dd->m,&dd->m,NULL,PETSC_DECIDE));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_processors_y","Number of processors in y direction","DMDASetNumProcs",dd->n,&dd->n,NULL,PETSC_DECIDE));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_processors_z","Number of processors in z direction","DMDASetNumProcs",dd->p,&dd->p,NULL,PETSC_DECIDE));
  /* Handle DMDA refinement */
  PetscCall(PetscOptionsBoundedInt("-da_refine_x","Refinement ratio in x direction","DMDASetRefinementFactor",dd->refine_x,&dd->refine_x,NULL,1));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_refine_y","Refinement ratio in y direction","DMDASetRefinementFactor",dd->refine_y,&dd->refine_y,NULL,1));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_refine_z","Refinement ratio in z direction","DMDASetRefinementFactor",dd->refine_z,&dd->refine_z,NULL,1));
  dd->coarsen_x = dd->refine_x; dd->coarsen_y = dd->refine_y; dd->coarsen_z = dd->refine_z;

  /* Get refinement factors, defaults taken from the coarse DMDA */
  PetscCall(DMDAGetRefinementFactor(da,&refx[0],&refy[0],&refz[0]));
  for (i=1; i<maxnlevels; i++) {
    refx[i] = refx[0];
    refy[i] = refy[0];
    refz[i] = refz[0];
  }
  n    = maxnlevels;
  PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_x","Refinement factor for each level","None",refx,&n,&flg));
  if (flg) {
    dd->refine_x = refx[0];
    dd->refine_x_hier_n = n;
    PetscCall(PetscMalloc1(n,&dd->refine_x_hier));
    PetscCall(PetscArraycpy(dd->refine_x_hier,refx,n));
  }
  if (dim > 1) {
    n    = maxnlevels;
    PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_y","Refinement factor for each level","None",refy,&n,&flg));
    if (flg) {
      dd->refine_y = refy[0];
      dd->refine_y_hier_n = n;
      PetscCall(PetscMalloc1(n,&dd->refine_y_hier));
      PetscCall(PetscArraycpy(dd->refine_y_hier,refy,n));
    }
  }
  if (dim > 2) {
    n    = maxnlevels;
    PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_z","Refinement factor for each level","None",refz,&n,&flg));
    if (flg) {
      dd->refine_z = refz[0];
      dd->refine_z_hier_n = n;
      PetscCall(PetscMalloc1(n,&dd->refine_z_hier));
      PetscCall(PetscArraycpy(dd->refine_z_hier,refz,n));
    }
  }

  PetscCall(PetscOptionsBoundedInt("-da_refine","Uniformly refine DA one or more times","None",refine,&refine,NULL,0));
  PetscOptionsHeadEnd();

  while (refine--) {
    if (dd->bx == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      PetscCall(PetscIntMultError(dd->refine_x,dd->M,&dd->M));
    } else {
      PetscCall(PetscIntMultError(dd->refine_x,dd->M-1,&dd->M));
      dd->M += 1;
    }
    if (dd->by == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      PetscCall(PetscIntMultError(dd->refine_y,dd->N,&dd->N));
    } else {
      PetscCall(PetscIntMultError(dd->refine_y,dd->N-1,&dd->N));
      dd->N += 1;
    }
    if (dd->bz == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      PetscCall(PetscIntMultError(dd->refine_z,dd->P,&dd->P));
    } else {
      PetscCall(PetscIntMultError(dd->refine_z,dd->P-1,&dd->P));
      dd->P += 1;
    }
    da->levelup++;
    if (da->levelup - da->leveldown >= 0) {
      dd->refine_x = refx[da->levelup - da->leveldown];
      dd->refine_y = refy[da->levelup - da->leveldown];
      dd->refine_z = refz[da->levelup - da->leveldown];
    }
    if (da->levelup - da->leveldown >= 1) {
      dd->coarsen_x = refx[da->levelup - da->leveldown - 1];
      dd->coarsen_y = refy[da->levelup - da->leveldown - 1];
      dd->coarsen_z = refz[da->levelup - da->leveldown - 1];
    }
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode  DMCreateGlobalVector_DA(DM,Vec*);
extern PetscErrorCode  DMCreateLocalVector_DA(DM,Vec*);
extern PetscErrorCode  DMGlobalToLocalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMGlobalToLocalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToGlobalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToGlobalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToLocalBegin_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMLocalToLocalEnd_DA(DM,Vec,InsertMode,Vec);
extern PetscErrorCode  DMCreateInterpolation_DA(DM,DM,Mat*,Vec*);
extern PetscErrorCode  DMCreateColoring_DA(DM,ISColoringType,ISColoring*);
extern PetscErrorCode  DMCreateMatrix_DA(DM,Mat*);
extern PetscErrorCode  DMCreateCoordinateDM_DA(DM,DM*);
extern PetscErrorCode  DMRefine_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode  DMCoarsen_DA(DM,MPI_Comm,DM*);
extern PetscErrorCode  DMRefineHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode  DMCoarsenHierarchy_DA(DM,PetscInt,DM[]);
extern PetscErrorCode  DMCreateInjection_DA(DM,DM,Mat*);
extern PetscErrorCode  DMView_DA(DM,PetscViewer);
extern PetscErrorCode  DMSetUp_DA(DM);
extern PetscErrorCode  DMDestroy_DA(DM);
extern PetscErrorCode  DMCreateDomainDecomposition_DA(DM,PetscInt*,char***,IS**,IS**,DM**);
extern PetscErrorCode  DMCreateDomainDecompositionScatters_DA(DM,PetscInt,DM*,VecScatter**,VecScatter**,VecScatter**);
PETSC_INTERN PetscErrorCode DMGetCompatibility_DA(DM,DM,PetscBool*,PetscBool*);

PetscErrorCode DMLoad_DA(DM da,PetscViewer viewer)
{
  PetscInt         dim,m,n,p,dof,swidth;
  DMDAStencilType  stencil;
  DMBoundaryType   bx,by,bz;
  PetscBool        coors;
  DM               dac;
  Vec              c;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer,&dim,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&m,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&n,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&p,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&dof,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&swidth,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,&bx,1,NULL,PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer,&by,1,NULL,PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer,&bz,1,NULL,PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer,&stencil,1,NULL,PETSC_ENUM));

  PetscCall(DMSetDimension(da, dim));
  PetscCall(DMDASetSizes(da, m,n,p));
  PetscCall(DMDASetBoundaryType(da, bx, by, bz));
  PetscCall(DMDASetDof(da, dof));
  PetscCall(DMDASetStencilType(da, stencil));
  PetscCall(DMDASetStencilWidth(da, swidth));
  PetscCall(DMSetUp(da));
  PetscCall(PetscViewerBinaryRead(viewer,&coors,1,NULL,PETSC_ENUM));
  if (coors) {
    PetscCall(DMGetCoordinateDM(da,&dac));
    PetscCall(DMCreateGlobalVector(dac,&c));
    PetscCall(VecLoad(c,viewer));
    PetscCall(DMSetCoordinates(da,c));
    PetscCall(VecDestroy(&c));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_DA(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  DM_DA         *da = (DM_DA*) dm->data;

  PetscFunctionBegin;
  if (subdm) {
    PetscSF sf;
    Vec     coords;
    void   *ctx;
    /* Cannot use DMClone since the dof stuff is mixed in. Ugh
    PetscCall(DMClone(dm, subdm)); */
    PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(DMSetPointSF(*subdm, sf));
    PetscCall(DMGetApplicationContext(dm, &ctx));
    PetscCall(DMSetApplicationContext(*subdm, ctx));
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    if (coords) {
      PetscCall(DMSetCoordinatesLocal(*subdm, coords));
    } else {
      PetscCall(DMGetCoordinates(dm, &coords));
      if (coords) PetscCall(DMSetCoordinates(*subdm, coords));
    }

    PetscCall(DMSetType(*subdm, DMDA));
    PetscCall(DMSetDimension(*subdm, dm->dim));
    PetscCall(DMDASetSizes(*subdm, da->M, da->N, da->P));
    PetscCall(DMDASetNumProcs(*subdm, da->m, da->n, da->p));
    PetscCall(DMDASetBoundaryType(*subdm, da->bx, da->by, da->bz));
    PetscCall(DMDASetDof(*subdm, numFields));
    PetscCall(DMDASetStencilType(*subdm, da->stencil_type));
    PetscCall(DMDASetStencilWidth(*subdm, da->s));
    PetscCall(DMDASetOwnershipRanges(*subdm, da->lx, da->ly, da->lz));
  }
  if (is) {
    PetscInt *indices, cnt = 0, dof = da->w, i, j;

    PetscCall(PetscMalloc1(da->Nlocal*numFields/dof, &indices));
    for (i = da->base/dof; i < (da->base+da->Nlocal)/dof; ++i) {
      for (j = 0; j < numFields; ++j) {
        indices[cnt++] = dof*i + fields[j];
      }
    }
    PetscCheck(cnt == da->Nlocal*numFields/dof,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count %" PetscInt_FMT " does not equal expected value %" PetscInt_FMT, cnt, da->Nlocal*numFields/dof);
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject) dm), cnt, indices, PETSC_OWN_POINTER, is));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateFieldDecomposition_DA(DM dm, PetscInt *len,char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt       i;
  DM_DA          *dd = (DM_DA*)dm->data;
  PetscInt       dof = dd->w;

  PetscFunctionBegin;
  if (len) *len = dof;
  if (islist) {
    Vec      v;
    PetscInt rstart,n;

    PetscCall(DMGetGlobalVector(dm,&v));
    PetscCall(VecGetOwnershipRange(v,&rstart,NULL));
    PetscCall(VecGetLocalSize(v,&n));
    PetscCall(DMRestoreGlobalVector(dm,&v));
    PetscCall(PetscMalloc1(dof,islist));
    for (i=0; i<dof; i++) {
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)dm),n/dof,rstart+i,dof,&(*islist)[i]));
    }
  }
  if (namelist) {
    PetscCall(PetscMalloc1(dof, namelist));
    if (dd->fieldname) {
      for (i=0; i<dof; i++) {
        PetscCall(PetscStrallocpy(dd->fieldname[i],&(*namelist)[i]));
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently DMDA must have fieldnames");
  }
  if (dmlist) {
    DM da;

    PetscCall(DMDACreate(PetscObjectComm((PetscObject)dm), &da));
    PetscCall(DMSetDimension(da, dm->dim));
    PetscCall(DMDASetSizes(da, dd->M, dd->N, dd->P));
    PetscCall(DMDASetNumProcs(da, dd->m, dd->n, dd->p));
    PetscCall(DMDASetBoundaryType(da, dd->bx, dd->by, dd->bz));
    PetscCall(DMDASetDof(da, 1));
    PetscCall(DMDASetStencilType(da, dd->stencil_type));
    PetscCall(DMDASetStencilWidth(da, dd->s));
    PetscCall(DMSetUp(da));
    PetscCall(PetscMalloc1(dof,dmlist));
    for (i=0; i<dof-1; i++) PetscCall(PetscObjectReference((PetscObject)da));
    for (i=0; i<dof; i++) (*dmlist)[i] = da;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMClone_DA(DM dm, DM *newdm)
{
  DM_DA         *da = (DM_DA *) dm->data;

  PetscFunctionBegin;
  PetscCall(DMSetType(*newdm, DMDA));
  PetscCall(DMSetDimension(*newdm, dm->dim));
  PetscCall(DMDASetSizes(*newdm, da->M, da->N, da->P));
  PetscCall(DMDASetNumProcs(*newdm, da->m, da->n, da->p));
  PetscCall(DMDASetBoundaryType(*newdm, da->bx, da->by, da->bz));
  PetscCall(DMDASetDof(*newdm, da->w));
  PetscCall(DMDASetStencilType(*newdm, da->stencil_type));
  PetscCall(DMDASetStencilWidth(*newdm, da->s));
  PetscCall(DMDASetOwnershipRanges(*newdm, da->lx, da->ly, da->lz));
  PetscCall(DMSetUp(*newdm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMHasCreateInjection_DA(DM dm, PetscBool *flg)
{
  DM_DA          *da = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = da->interptype == DMDA_Q1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetDimPoints_DA(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  PetscCall(DMDAGetDepthStratum(dm, dim, pStart, pEnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetNeighbors_DA(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscInt dim;
  DMDAStencilType st;

  PetscFunctionBegin;
  PetscCall(DMDAGetNeighbors(dm,ranks));
  PetscCall(DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&st));

  switch (dim) {
    case 1:
      *nranks = 3;
      /* if (st == DMDA_STENCIL_STAR) { *nranks = 3; } */
      break;
    case 2:
      *nranks = 9;
      /* if (st == DMDA_STENCIL_STAR) { *nranks = 5; } */
      break;
    case 3:
      *nranks = 27;
      /* if (st == DMDA_STENCIL_STAR) { *nranks = 7; } */
      break;
    default:
      break;
  }
  PetscFunctionReturn(0);
}

/*MC
   DMDA = "da" - A DM object that is used to manage data for a structured grid in 1, 2, or 3 dimensions.
         In the global representation of the vector each process stores a non-overlapping rectangular (or slab in 3d) portion of the grid points.
         In the local representation these rectangular regions (slabs) are extended in all directions by a stencil width.

         The vectors can be thought of as either cell centered or vertex centered on the mesh. But some variables cannot be cell centered and others
         vertex centered; see the documentation for DMSTAG, a similar DM implementation which supports these staggered grids.

  Level: intermediate

.seealso: `DMType`, `DMCOMPOSITE`, `DMSTAG`, `DMDACreate()`, `DMCreate()`, `DMSetType()`
M*/

extern PetscErrorCode DMLocatePoints_DA_Regular(DM,Vec,DMPointLocationType,PetscSF);
PETSC_INTERN PetscErrorCode DMSetUpGLVisViewer_DMDA(PetscObject,PetscViewer);

PETSC_EXTERN PetscErrorCode DMCreate_DA(DM da)
{
  DM_DA          *dd;

  PetscFunctionBegin;
  PetscValidPointer(da,1);
  PetscCall(PetscNewLog(da,&dd));
  da->data = dd;

  da->dim        = -1;
  dd->interptype = DMDA_Q1;
  dd->refine_x   = 2;
  dd->refine_y   = 2;
  dd->refine_z   = 2;
  dd->coarsen_x  = 2;
  dd->coarsen_y  = 2;
  dd->coarsen_z  = 2;
  dd->fieldname  = NULL;
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

  dd->Nsub            = 1;
  dd->xol             = 0;
  dd->yol             = 0;
  dd->zol             = 0;
  dd->xo              = 0;
  dd->yo              = 0;
  dd->zo              = 0;
  dd->Mo              = -1;
  dd->No              = -1;
  dd->Po              = -1;

  dd->gtol         = NULL;
  dd->ltol         = NULL;
  dd->ao           = NULL;
  PetscStrallocpy(AOBASIC,(char**)&dd->aotype);
  dd->base         = -1;
  dd->bx           = DM_BOUNDARY_NONE;
  dd->by           = DM_BOUNDARY_NONE;
  dd->bz           = DM_BOUNDARY_NONE;
  dd->stencil_type = DMDA_STENCIL_BOX;
  dd->interptype   = DMDA_Q1;
  dd->lx           = NULL;
  dd->ly           = NULL;
  dd->lz           = NULL;

  dd->elementtype = DMDA_ELEMENT_Q1;

  da->ops->globaltolocalbegin          = DMGlobalToLocalBegin_DA;
  da->ops->globaltolocalend            = DMGlobalToLocalEnd_DA;
  da->ops->localtoglobalbegin          = DMLocalToGlobalBegin_DA;
  da->ops->localtoglobalend            = DMLocalToGlobalEnd_DA;
  da->ops->localtolocalbegin           = DMLocalToLocalBegin_DA;
  da->ops->localtolocalend             = DMLocalToLocalEnd_DA;
  da->ops->createglobalvector          = DMCreateGlobalVector_DA;
  da->ops->createlocalvector           = DMCreateLocalVector_DA;
  da->ops->createinterpolation         = DMCreateInterpolation_DA;
  da->ops->getcoloring                 = DMCreateColoring_DA;
  da->ops->creatematrix                = DMCreateMatrix_DA;
  da->ops->refine                      = DMRefine_DA;
  da->ops->coarsen                     = DMCoarsen_DA;
  da->ops->refinehierarchy             = DMRefineHierarchy_DA;
  da->ops->coarsenhierarchy            = DMCoarsenHierarchy_DA;
  da->ops->createinjection             = DMCreateInjection_DA;
  da->ops->hascreateinjection          = DMHasCreateInjection_DA;
  da->ops->destroy                     = DMDestroy_DA;
  da->ops->view                        = NULL;
  da->ops->setfromoptions              = DMSetFromOptions_DA;
  da->ops->setup                       = DMSetUp_DA;
  da->ops->clone                       = DMClone_DA;
  da->ops->load                        = DMLoad_DA;
  da->ops->createcoordinatedm          = DMCreateCoordinateDM_DA;
  da->ops->createsubdm                 = DMCreateSubDM_DA;
  da->ops->createfielddecomposition    = DMCreateFieldDecomposition_DA;
  da->ops->createdomaindecomposition   = DMCreateDomainDecomposition_DA;
  da->ops->createddscatters            = DMCreateDomainDecompositionScatters_DA;
  da->ops->getdimpoints                = DMGetDimPoints_DA;
  da->ops->getneighbors                = DMGetNeighbors_DA;
  da->ops->locatepoints                = DMLocatePoints_DA_Regular;
  da->ops->getcompatibility            = DMGetCompatibility_DA;
  PetscCall(PetscObjectComposeFunction((PetscObject)da,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_DMDA));
  PetscFunctionReturn(0);
}

/*@
  DMDACreate - Creates a DMDA object.

  Collective

  Input Parameter:
. comm - The communicator for the DMDA object

  Output Parameter:
. da  - The DMDA object

  Level: advanced

  Developers Note:
  Since there exists DMDACreate1/2/3d() should this routine even exist?

.seealso: `DMDASetSizes()`, `DMClone()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode  DMDACreate(MPI_Comm comm, DM *da)
{
  PetscFunctionBegin;
  PetscValidPointer(da,2);
  PetscCall(DMCreate(comm,da));
  PetscCall(DMSetType(*da,DMDA));
  PetscFunctionReturn(0);
}
