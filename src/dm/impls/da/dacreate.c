
#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  DMSetFromOptions_DA(PetscOptionItems *PetscOptionsObject,DM da)
{
  PetscErrorCode ierr;
  DM_DA          *dd    = (DM_DA*)da->data;
  PetscInt       refine = 0,dim = da->dim,maxnlevels = 100,refx[100],refy[100],refz[100],n,i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,2);

  if (dd->M < 0) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  if (dd->N < 0) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  if (dd->P < 0) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");

  ierr = PetscOptionsHead(PetscOptionsObject,"DMDA Options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-da_grid_x","Number of grid points in x direction","DMDASetSizes",dd->M,&dd->M,NULL,1);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsBoundedInt("-da_grid_y","Number of grid points in y direction","DMDASetSizes",dd->N,&dd->N,NULL,1);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsBoundedInt("-da_grid_z","Number of grid points in z direction","DMDASetSizes",dd->P,&dd->P,NULL,1);CHKERRQ(ierr);}

  ierr = PetscOptionsBoundedInt("-da_overlap","Decomposition overlap in all directions","DMDASetOverlap",dd->xol,&dd->xol,&flg,0);CHKERRQ(ierr);
  if (flg) {ierr = DMDASetOverlap(da,dd->xol,dd->xol,dd->xol);CHKERRQ(ierr);}
  ierr = PetscOptionsBoundedInt("-da_overlap_x","Decomposition overlap in x direction","DMDASetOverlap",dd->xol,&dd->xol,NULL,0);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsBoundedInt("-da_overlap_y","Decomposition overlap in y direction","DMDASetOverlap",dd->yol,&dd->yol,NULL,0);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsBoundedInt("-da_overlap_z","Decomposition overlap in z direction","DMDASetOverlap",dd->zol,&dd->zol,NULL,0);CHKERRQ(ierr);}

  ierr = PetscOptionsBoundedInt("-da_local_subdomains","","DMDASetNumLocalSubdomains",dd->Nsub,&dd->Nsub,&flg,PETSC_DECIDE);CHKERRQ(ierr);
  if (flg) {ierr = DMDASetNumLocalSubDomains(da,dd->Nsub);CHKERRQ(ierr);}

  /* Handle DMDA parallel distribution */
  ierr = PetscOptionsBoundedInt("-da_processors_x","Number of processors in x direction","DMDASetNumProcs",dd->m,&dd->m,NULL,PETSC_DECIDE);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsBoundedInt("-da_processors_y","Number of processors in y direction","DMDASetNumProcs",dd->n,&dd->n,NULL,PETSC_DECIDE);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsBoundedInt("-da_processors_z","Number of processors in z direction","DMDASetNumProcs",dd->p,&dd->p,NULL,PETSC_DECIDE);CHKERRQ(ierr);}
  /* Handle DMDA refinement */
  ierr = PetscOptionsBoundedInt("-da_refine_x","Refinement ratio in x direction","DMDASetRefinementFactor",dd->refine_x,&dd->refine_x,NULL,1);CHKERRQ(ierr);
  if (dim > 1) {ierr = PetscOptionsBoundedInt("-da_refine_y","Refinement ratio in y direction","DMDASetRefinementFactor",dd->refine_y,&dd->refine_y,NULL,1);CHKERRQ(ierr);}
  if (dim > 2) {ierr = PetscOptionsBoundedInt("-da_refine_z","Refinement ratio in z direction","DMDASetRefinementFactor",dd->refine_z,&dd->refine_z,NULL,1);CHKERRQ(ierr);}
  dd->coarsen_x = dd->refine_x; dd->coarsen_y = dd->refine_y; dd->coarsen_z = dd->refine_z;

  /* Get refinement factors, defaults taken from the coarse DMDA */
  ierr = DMDAGetRefinementFactor(da,&refx[0],&refy[0],&refz[0]);CHKERRQ(ierr);
  for (i=1; i<maxnlevels; i++) {
    refx[i] = refx[0];
    refy[i] = refy[0];
    refz[i] = refz[0];
  }
  n    = maxnlevels;
  ierr = PetscOptionsIntArray("-da_refine_hierarchy_x","Refinement factor for each level","None",refx,&n,&flg);CHKERRQ(ierr);
  if (flg) {
    dd->refine_x = refx[0];
    dd->refine_x_hier_n = n;
    ierr = PetscMalloc1(n,&dd->refine_x_hier);CHKERRQ(ierr);
    ierr = PetscArraycpy(dd->refine_x_hier,refx,n);CHKERRQ(ierr);
  }
  if (dim > 1) {
    n    = maxnlevels;
    ierr = PetscOptionsIntArray("-da_refine_hierarchy_y","Refinement factor for each level","None",refy,&n,&flg);CHKERRQ(ierr);
    if (flg) {
      dd->refine_y = refy[0];
      dd->refine_y_hier_n = n;
      ierr = PetscMalloc1(n,&dd->refine_y_hier);CHKERRQ(ierr);
      ierr = PetscArraycpy(dd->refine_y_hier,refy,n);CHKERRQ(ierr);
    }
  }
  if (dim > 2) {
    n    = maxnlevels;
    ierr = PetscOptionsIntArray("-da_refine_hierarchy_z","Refinement factor for each level","None",refz,&n,&flg);CHKERRQ(ierr);
    if (flg) {
      dd->refine_z = refz[0];
      dd->refine_z_hier_n = n;
      ierr = PetscMalloc1(n,&dd->refine_z_hier);CHKERRQ(ierr);
      ierr = PetscArraycpy(dd->refine_z_hier,refz,n);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsBoundedInt("-da_refine","Uniformly refine DA one or more times","None",refine,&refine,NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  while (refine--) {
    if (dd->bx == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      dd->M = dd->refine_x*dd->M;
    } else {
      dd->M = 1 + dd->refine_x*(dd->M - 1);
    }
    if (dd->by == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      dd->N = dd->refine_y*dd->N;
    } else {
      dd->N = 1 + dd->refine_y*(dd->N - 1);
    }
    if (dd->bz == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      dd->P = dd->refine_z*dd->P;
    } else {
      dd->P = 1 + dd->refine_z*(dd->P - 1);
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
  PetscErrorCode   ierr;
  PetscInt         dim,m,n,p,dof,swidth;
  DMDAStencilType  stencil;
  DMBoundaryType   bx,by,bz;
  PetscBool        coors;
  DM               dac;
  Vec              c;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryRead(viewer,&dim,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&m,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&n,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&p,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&dof,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&swidth,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&bx,1,NULL,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&by,1,NULL,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&bz,1,NULL,PETSC_ENUM);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&stencil,1,NULL,PETSC_ENUM);CHKERRQ(ierr);

  ierr = DMSetDimension(da, dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, m,n,p);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, bx, by, bz);CHKERRQ(ierr);
  ierr = DMDASetDof(da, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da, stencil);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, swidth);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&coors,1,NULL,PETSC_ENUM);CHKERRQ(ierr);
  if (coors) {
    ierr = DMGetCoordinateDM(da,&dac);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dac,&c);CHKERRQ(ierr);
    ierr = VecLoad(c,viewer);CHKERRQ(ierr);
    ierr = DMSetCoordinates(da,c);CHKERRQ(ierr);
    ierr = VecDestroy(&c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_DA(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  DM_DA         *da = (DM_DA*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {
    PetscSF sf;
    Vec     coords;
    void   *ctx;
    /* Cannot use DMClone since the dof stuff is mixed in. Ugh
    ierr = DMClone(dm, subdm);CHKERRQ(ierr); */
    ierr = DMCreate(PetscObjectComm((PetscObject)dm), subdm);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = DMSetPointSF(*subdm, sf);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(*subdm, ctx);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
    if (coords) {
      ierr = DMSetCoordinatesLocal(*subdm, coords);CHKERRQ(ierr);
    } else {
      ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
      if (coords) {ierr = DMSetCoordinates(*subdm, coords);CHKERRQ(ierr);}
    }

    ierr = DMSetType(*subdm, DMDA);CHKERRQ(ierr);
    ierr = DMSetDimension(*subdm, dm->dim);CHKERRQ(ierr);
    ierr = DMDASetSizes(*subdm, da->M, da->N, da->P);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(*subdm, da->m, da->n, da->p);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(*subdm, da->bx, da->by, da->bz);CHKERRQ(ierr);
    ierr = DMDASetDof(*subdm, numFields);CHKERRQ(ierr);
    ierr = DMDASetStencilType(*subdm, da->stencil_type);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(*subdm, da->s);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(*subdm, da->lx, da->ly, da->lz);CHKERRQ(ierr);
  }
  if (is) {
    PetscInt *indices, cnt = 0, dof = da->w, i, j;

    ierr = PetscMalloc1(da->Nlocal*numFields/dof, &indices);CHKERRQ(ierr);
    for (i = da->base/dof; i < (da->base+da->Nlocal)/dof; ++i) {
      for (j = 0; j < numFields; ++j) {
        indices[cnt++] = dof*i + fields[j];
      }
    }
    if (cnt != da->Nlocal*numFields/dof) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count %D does not equal expected value %D", cnt, da->Nlocal*numFields/dof);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), cnt, indices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateFieldDecomposition_DA(DM dm, PetscInt *len,char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt       i;
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)dm->data;
  PetscInt       dof = dd->w;

  PetscFunctionBegin;
  if (len) *len = dof;
  if (islist) {
    Vec      v;
    PetscInt rstart,n;

    ierr = DMGetGlobalVector(dm,&v);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(v,&rstart,NULL);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&v);CHKERRQ(ierr);
    ierr = PetscMalloc1(dof,islist);CHKERRQ(ierr);
    for (i=0; i<dof; i++) {
      ierr = ISCreateStride(PetscObjectComm((PetscObject)dm),n/dof,rstart+i,dof,&(*islist)[i]);CHKERRQ(ierr);
    }
  }
  if (namelist) {
    ierr = PetscMalloc1(dof, namelist);CHKERRQ(ierr);
    if (dd->fieldname) {
      for (i=0; i<dof; i++) {
        ierr = PetscStrallocpy(dd->fieldname[i],&(*namelist)[i]);CHKERRQ(ierr);
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently DMDA must have fieldnames");
  }
  if (dmlist) {
    DM da;

    ierr = DMDACreate(PetscObjectComm((PetscObject)dm), &da);CHKERRQ(ierr);
    ierr = DMSetDimension(da, dm->dim);CHKERRQ(ierr);
    ierr = DMDASetSizes(da, dd->M, dd->N, dd->P);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(da, dd->m, dd->n, dd->p);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(da, dd->bx, dd->by, dd->bz);CHKERRQ(ierr);
    ierr = DMDASetDof(da, 1);CHKERRQ(ierr);
    ierr = DMDASetStencilType(da, dd->stencil_type);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(da, dd->s);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);
    ierr = PetscMalloc1(dof,dmlist);CHKERRQ(ierr);
    for (i=0; i<dof-1; i++) {ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);}
    for (i=0; i<dof; i++) (*dmlist)[i] = da;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMClone_DA(DM dm, DM *newdm)
{
  DM_DA         *da = (DM_DA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetType(*newdm, DMDA);CHKERRQ(ierr);
  ierr = DMSetDimension(*newdm, dm->dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(*newdm, da->M, da->N, da->P);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*newdm, da->m, da->n, da->p);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*newdm, da->bx, da->by, da->bz);CHKERRQ(ierr);
  ierr = DMDASetDof(*newdm, da->w);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*newdm, da->stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*newdm, da->s);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*newdm, da->lx, da->ly, da->lz);CHKERRQ(ierr);
  ierr = DMSetUp(*newdm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetDepthStratum(dm, dim, pStart, pEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetNeighbors_DA(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscErrorCode ierr;
  PetscInt dim;
  DMDAStencilType st;

  PetscFunctionBegin;
  ierr = DMDAGetNeighbors(dm,ranks);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&st);CHKERRQ(ierr);

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

.seealso: DMType, DMCOMPOSITE, DMSTAG, DMDACreate(), DMCreate(), DMSetType()
M*/

extern PetscErrorCode DMLocatePoints_DA_Regular(DM,Vec,DMPointLocationType,PetscSF);
PETSC_INTERN PetscErrorCode DMSetUpGLVisViewer_DMDA(PetscObject,PetscViewer);

PETSC_EXTERN PetscErrorCode DMCreate_DA(DM da)
{
  PetscErrorCode ierr;
  DM_DA          *dd;

  PetscFunctionBegin;
  PetscValidPointer(da,1);
  ierr     = PetscNewLog(da,&dd);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunction((PetscObject)da,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_DMDA);CHKERRQ(ierr);
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

.seealso:  DMDASetSizes(), DMClone(),  DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
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
