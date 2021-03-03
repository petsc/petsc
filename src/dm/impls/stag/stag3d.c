/* Functions specific to the 3-dimensional implementation of DMStag */
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagCreate3d - Create an object to manage data living on the faces, edges, and vertices of a parallelized regular 3D grid.

  Collective

  Input Parameters:
+ comm - MPI communicator
. bndx,bndy,bndz - boundary type: DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, or DM_BOUNDARY_GHOSTED
. M,N,P - global number of grid points in x,y directions
. m,n,p - number of ranks in the x,y directions (may be PETSC_DECIDE)
. dof0 - number of degrees of freedom per vertex/point/node/0-cell
. dof1 - number of degrees of freedom per edge/1-cell
. dof2 - number of degrees of freedom per face/2-cell
. dof3 - number of degrees of freedom per element/3-cell
. stencilType - ghost/halo region type: DMSTAG_STENCIL_NONE, DMSTAG_STENCIL_BOX, or DMSTAG_STENCIL_STAR
. stencilWidth - width, in elements, of halo/ghost region
- lx,ly,lz - array sof local x,y,z element counts, of length equal to m,n,p, summing to M,N,P

  Output Parameter:
. dm - the new DMStag object

  Options Database Keys:
+ -dm_view - calls DMViewFromOptions() a the conclusion of DMSetUp()
. -stag_grid_x <nx> - number of elements in the x direction
. -stag_grid_y <ny> - number of elements in the y direction
. -stag_grid_z <nz> - number of elements in the z direction
. -stag_ranks_x <rx> - number of ranks in the x direction
. -stag_ranks_y <ry> - number of ranks in the y direction
. -stag_ranks_z <rz> - number of ranks in the z direction
. -stag_ghost_stencil_width - width of ghost region, in elements
. -stag_boundary_type x <none,ghosted,periodic> - DMBoundaryType value
. -stag_boundary_type y <none,ghosted,periodic> - DMBoundaryType value
- -stag_boundary_type z <none,ghosted,periodic> - DMBoundaryType value

  Notes:
  You must call DMSetUp() after this call before using the DM.
  If you wish to use the options database (see the keys above) to change values in the DMStag, you must call
  DMSetFromOptions() after this function but before DMSetUp().

  Level: beginner

.seealso: DMSTAG, DMStagCreate1d(), DMStagCreate2d(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateLocalVector(), DMLocalToGlobalBegin(), DMDACreate3d()
@*/
PETSC_EXTERN PetscErrorCode DMStagCreate3d(MPI_Comm comm,DMBoundaryType bndx,DMBoundaryType bndy,DMBoundaryType bndz,PetscInt M,PetscInt N,PetscInt P,PetscInt m,PetscInt n,PetscInt p,PetscInt dof0,PetscInt dof1,PetscInt dof2,PetscInt dof3,DMStagStencilType stencilType,PetscInt stencilWidth,const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],DM* dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm,3);CHKERRQ(ierr);
  ierr = DMStagInitialize(bndx,bndy,bndz,M,N,P,m,n,p,dof0,dof1,dof2,dof3,stencilType,stencilWidth,lx,ly,lz,*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_3d(DM dm,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  PetscErrorCode ierr;
  DM_Stag        *stagCoord;
  DM             dmCoord;
  Vec            coordLocal,coord;
  PetscReal      h[3],min[3];
  PetscScalar    ****arr;
  PetscInt       ind[3],start[3],n[3],nExtra[3],s,c;
  PetscInt       ibackdownleft,ibackdown,ibackleft,iback,idownleft,idown,ileft,ielement;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
  stagCoord = (DM_Stag*) dmCoord->data;
  for (s=0; s<4; ++s) {
    if (stagCoord->dof[s] !=0 && stagCoord->dof[s] != 3) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Coordinate DM in 3 dimensions must have 0 or 3 dof on each stratum, but stratum %d has %d dof",s,stagCoord->dof[s]);
  }
  ierr = DMGetLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  if (stagCoord->dof[0]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_BACK_DOWN_LEFT,0,&ibackdownleft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[1]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_BACK_DOWN     ,0,&ibackdown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_BACK_LEFT     ,0,&ibackleft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_DOWN_LEFT     ,0,&idownleft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[2]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_BACK          ,0,&iback);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_DOWN          ,0,&idown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_LEFT          ,0,&ileft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[3]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_ELEMENT       ,0,&ielement);CHKERRQ(ierr);
  }
  ierr = DMStagGetCorners(dmCoord,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&nExtra[0],&nExtra[1],&nExtra[2]);CHKERRQ(ierr);
  min[0] = xmin; min[1]= ymin; min[2] = zmin;
  h[0] = (xmax-xmin)/stagCoord->N[0];
  h[1] = (ymax-ymin)/stagCoord->N[1];
  h[2] = (zmax-zmin)/stagCoord->N[2];

  for (ind[2]=start[2]; ind[2]<start[2] + n[2] + nExtra[2]; ++ind[2]) {
    for (ind[1]=start[1]; ind[1]<start[1] + n[1] + nExtra[1]; ++ind[1]) {
      for (ind[0]=start[0]; ind[0]<start[0] + n[0] + nExtra[0]; ++ind[0]) {
        if (stagCoord->dof[0]) {
          const PetscReal offs[3] = {0.0,0.0,0.0};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][ibackdownleft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[1]) {
          const PetscReal offs[3] = {0.5,0.0,0.0};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][ibackdown + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[1]) {
          const PetscReal offs[3] = {0.0,0.5,0.0};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][ibackleft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[2]) {
          const PetscReal offs[3] = {0.5,0.5,0.0};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][iback + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[1]) {
          const PetscReal offs[3] = {0.0,0.0,0.5};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][idownleft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[2]) {
          const PetscReal offs[3] = {0.5,0.0,0.5};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][idown + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[2]) {
          const PetscReal offs[3] = {0.0,0.5,0.5};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][ileft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
        if (stagCoord->dof[3]) {
          const PetscReal offs[3] = {0.5,0.5,0.5};
          for (c=0; c<3; ++c) {
            arr[ind[2]][ind[1]][ind[0]][ielement + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
          }
        }
      }
    }
  }
  ierr = DMStagVecRestoreArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmCoord,&coord);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmCoord,coordLocal,INSERT_VALUES,coord);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmCoord,coordLocal,INSERT_VALUES,coord);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm,coord);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)coord);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&coord);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Helper functions used in DMSetUp_Stag() */
static PetscErrorCode DMStagSetUpBuildRankGrid_3d(DM);
static PetscErrorCode DMStagSetUpBuildNeighbors_3d(DM);
static PetscErrorCode DMStagSetUpBuildGlobalOffsets_3d(DM,PetscInt**);
static PetscErrorCode DMStagSetUpBuildScatter_3d(DM,const PetscInt*);
static PetscErrorCode DMStagSetUpBuildL2G_3d(DM,const PetscInt*);
static PetscErrorCode DMStagComputeLocationOffsets_3d(DM);

PETSC_INTERN PetscErrorCode DMSetUp_Stag_3d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscMPIInt     rank;
  PetscInt        i,j,d;
  PetscInt        *globalOffsets;
  const PetscInt  dim = 3;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

  /* Rank grid sizes (populates stag->nRanks) */
  ierr = DMStagSetUpBuildRankGrid_3d(dm);CHKERRQ(ierr);

  /* Determine location of rank in grid */
    stag->rank[0] = rank % stag->nRanks[0];
    stag->rank[1] = rank % (stag->nRanks[0] * stag->nRanks[1]) / stag->nRanks[0];
    stag->rank[2] = rank / (stag->nRanks[0] * stag->nRanks[1]);
    for (d=0; d<dim; ++d) {
      stag->firstRank[d] = PetscNot(stag->rank[d]);
      stag->lastRank[d]  = (PetscBool)(stag->rank[d] == stag->nRanks[d]-1);
    }

  /* Determine locally owned region (if not already prescribed).
   Divide equally, giving lower ranks in each dimension and extra element if needbe.
   Note that this uses O(P) storage. If this ever becomes an issue, this could
   be refactored to not keep this data around.  */
  for (i=0; i<dim; ++i) {
    if (!stag->l[i]) {
      const PetscInt Ni = stag->N[i], nRanksi = stag->nRanks[i];
      ierr = PetscMalloc1(stag->nRanks[i],&stag->l[i]);CHKERRQ(ierr);
      for (j=0; j<stag->nRanks[i]; ++j){
        stag->l[i][j] = Ni/nRanksi + ((Ni % nRanksi) > j);
      }
    }
  }

  /* Retrieve local size in stag->n */
  for (i=0; i<dim; ++i) stag->n[i] = stag->l[i][stag->rank[i]];
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<dim; ++i) {
      PetscInt Ncheck,j;
      Ncheck = 0;
      for (j=0; j<stag->nRanks[i]; ++j) Ncheck += stag->l[i][j];
      if (Ncheck != stag->N[i]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local sizes in dimension %d don't add up. %d != %d\n",i,Ncheck,stag->N[i]);
    }
  }

  /* Compute starting elements */
  for (i=0; i<dim; ++i) {
    stag->start[i] = 0;
    for (j=0;j<stag->rank[i];++j){
      stag->start[i] += stag->l[i][j];
    }
  }

  /* Determine ranks of neighbors */
  ierr = DMStagSetUpBuildNeighbors_3d(dm);CHKERRQ(ierr);

  /* Define useful sizes */
  {
    PetscInt entriesPerEdge,entriesPerFace,entriesPerCorner,entriesPerElementRow,entriesPerElementLayer;
    PetscBool dummyEnd[3];
    for (d=0; d<3; ++d) dummyEnd[d] = (PetscBool)(stag->lastRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
    stag->entriesPerElement = stag->dof[0] + 3*stag->dof[1] + 3*stag->dof[2] + stag->dof[3];
    entriesPerFace          = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
    entriesPerEdge          = stag->dof[0] + stag->dof[1];
    entriesPerCorner        = stag->dof[0];
    entriesPerElementRow    = stag->n[0]*stag->entriesPerElement
                              + (dummyEnd[0]                               ? entriesPerFace                       : 0);
    entriesPerElementLayer  = stag->n[1]*entriesPerElementRow
                              + (dummyEnd[1]                               ? stag->n[0] * entriesPerFace          : 0)
                              + (dummyEnd[1] && dummyEnd[0]                ? entriesPerEdge                       : 0);
    stag->entries           = stag->n[2]*entriesPerElementLayer
                              + (dummyEnd[2]                               ? stag->n[0]*stag->n[1]*entriesPerFace : 0)
                              + (dummyEnd[2] && dummyEnd[0]                ? stag->n[1]*entriesPerEdge            : 0)
                              + (dummyEnd[2] && dummyEnd[1]                ? stag->n[0]*entriesPerEdge            : 0)
                              + (dummyEnd[2] && dummyEnd[1] && dummyEnd[0] ? entriesPerCorner                     : 0);
  }

  /* Check that we will not overflow 32 bit indices (slightly overconservative) */
  if (!PetscDefined(USE_64BIT_INDICES)) {
    if (((PetscInt64) stag->n[0])*((PetscInt64) stag->n[1])*((PetscInt64) stag->n[2])*((PetscInt64) stag->entriesPerElement) > (PetscInt64) PETSC_MPI_INT_MAX) SETERRQ4(PetscObjectComm((PetscObject)dm),PETSC_ERR_INT_OVERFLOW,"Mesh of %D x %D x %D with %D entries per (interior) element is likely too large for 32 bit indices",stag->n[0],stag->n[1],stag->n[2],stag->entriesPerElement);
  }

  /* Compute offsets for each rank into global vectors

    This again requires O(P) storage, which could be replaced with some global
    communication.
  */
  ierr = DMStagSetUpBuildGlobalOffsets_3d(dm,&globalOffsets);CHKERRQ(ierr);

  for (d=0; d<dim; ++d) if (stag->boundaryType[d] != DM_BOUNDARY_NONE && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->boundaryType[d] != DM_BOUNDARY_GHOSTED) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported boundary type");

  /* Define ghosted/local sizes */
  for (d=0; d<dim; ++d) {
    switch (stag->boundaryType[d]) {
      case DM_BOUNDARY_NONE:
        /* Note: for a elements-only DMStag, the extra elements on the edges aren't necessary but we include them anyway */
        switch (stag->stencilType) {
          case DMSTAG_STENCIL_NONE : /* only the extra one on the right/top edges */
            stag->nGhost[d] = stag->n[d];
            stag->startGhost[d] = stag->start[d];
            if (stag->lastRank[d]) stag->nGhost[d] += 1;
            break;
          case DMSTAG_STENCIL_STAR : /* allocate the corners but don't use them */
          case DMSTAG_STENCIL_BOX :
            stag->nGhost[d] = stag->n[d];
            stag->startGhost[d] = stag->start[d];
            if (!stag->firstRank[d]) {
              stag->nGhost[d]     += stag->stencilWidth; /* add interior ghost elements */
              stag->startGhost[d] -= stag->stencilWidth;
            }
            if (!stag->lastRank[d]) {
              stag->nGhost[d] += stag->stencilWidth; /* add interior ghost elements */
            } else {
              stag->nGhost[d] += 1; /* one element on the boundary to complete blocking */
            }
            break;
          default :
            SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
        }
        break;
      case DM_BOUNDARY_GHOSTED:
        switch (stag->stencilType) {
          case DMSTAG_STENCIL_NONE :
            stag->startGhost[d] = stag->start[d];
            stag->nGhost[d]     = stag->n[d] + (stag->lastRank[d] ? 1 : 0);
            break;
          case DMSTAG_STENCIL_STAR :
          case DMSTAG_STENCIL_BOX :
            stag->startGhost[d] = stag->start[d] - stag->stencilWidth; /* This value may be negative */
            stag->nGhost[d]     = stag->n[d] + 2*stag->stencilWidth + (stag->lastRank[d] && stag->stencilWidth == 0 ? 1 : 0);
            break;
          default :
            SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
        }
        break;
      case DM_BOUNDARY_PERIODIC:
        switch (stag->stencilType) {
          case DMSTAG_STENCIL_NONE : /* only the extra one on the right/top edges */
            stag->nGhost[d] = stag->n[d];
            stag->startGhost[d] = stag->start[d];
            break;
          case DMSTAG_STENCIL_STAR :
          case DMSTAG_STENCIL_BOX :
            stag->nGhost[d] = stag->n[d] + 2*stag->stencilWidth;
            stag->startGhost[d] = stag->start[d] - stag->stencilWidth;
            break;
          default :
            SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
        }
        break;
      default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported boundary type in dimension %D",d);
    }
  }
  stag->entriesGhost = stag->nGhost[0]*stag->nGhost[1]*stag->nGhost[2]*stag->entriesPerElement;

  /* Create global-->local VecScatter and local->global ISLocalToGlobalMapping */
  ierr = DMStagSetUpBuildScatter_3d(dm,globalOffsets);CHKERRQ(ierr);
  ierr = DMStagSetUpBuildL2G_3d(dm,globalOffsets);CHKERRQ(ierr);

  /* In special cases, create a dedicated injective local-to-global map */
  if ((stag->boundaryType[0] == DM_BOUNDARY_PERIODIC && stag->nRanks[0] == 1) ||
      (stag->boundaryType[1] == DM_BOUNDARY_PERIODIC && stag->nRanks[1] == 1) ||
      (stag->boundaryType[2] == DM_BOUNDARY_PERIODIC && stag->nRanks[2] == 1)) {
    ierr = DMStagPopulateLocalToGlobalInjective(dm);CHKERRQ(ierr);
  }

  /* Free global offsets */
  ierr = PetscFree(globalOffsets);CHKERRQ(ierr);

  /* Precompute location offsets */
  ierr = DMStagComputeLocationOffsets_3d(dm);CHKERRQ(ierr);

  /* View from Options */
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* adapted from da3.c */
static PetscErrorCode DMStagSetUpBuildRankGrid_3d(DM dm)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        m,n,p,pm;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt M = stag->N[0];
  const PetscInt N = stag->N[1];
  const PetscInt P = stag->N[2];

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);

  m = stag->nRanks[0];
  n = stag->nRanks[1];
  p = stag->nRanks[2];

  if (m != PETSC_DECIDE) {
    if (m < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in X direction: %D",m);
    else if (m > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in X direction: %D %d",m,size);
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Y direction: %D",n);
    else if (n > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Y direction: %D %d",n,size);
  }
  if (p != PETSC_DECIDE) {
    if (p < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of processors in Z direction: %D",p);
    else if (p > size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many processors in Z direction: %D %d",p,size);
  }
  if ((m > 0) && (n > 0) && (p > 0) && (m*n*p != size)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m %D * n %D * p %D != size %d",m,n,p,size);

  /* Partition the array among the processors */
  if (m == PETSC_DECIDE && n != PETSC_DECIDE && p != PETSC_DECIDE) {
    m = size/(n*p);
  } else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    n = size/(m*p);
  } else if (m != PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    p = size/(m*n);
  } else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p != PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)N*p)));
    if (!m) m = 1;
    while (m > 0) {
      n = size/(m*p);
      if (m*n*p == size) break;
      m--;
    }
    if (!m) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad p value: p = %D",p);
    if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}
  } else if (m == PETSC_DECIDE && n != PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    m = (int)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)P*n)));
    if (!m) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (!m) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad n value: n = %D",n);
    if (M > P && m < p) {PetscInt _m = m; m = p; p = _m;}
  } else if (m != PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (int)(0.5 + PetscSqrtReal(((PetscReal)N)*((PetscReal)size)/((PetscReal)P*m)));
    if (!n) n = 1;
    while (n > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      n--;
    }
    if (!n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"bad m value: m = %D",n);
    if (N > P && n < p) {PetscInt _n = n; n = p; p = _n;}
  } else if (m == PETSC_DECIDE && n == PETSC_DECIDE && p == PETSC_DECIDE) {
    /* try for squarish distribution */
    n = (PetscInt)(0.5 + PetscPowReal(((PetscReal)N*N)*((PetscReal)size)/((PetscReal)P*M),(PetscReal)(1./3.)));
    if (!n) n = 1;
    while (n > 0) {
      pm = size/n;
      if (n*pm == size) break;
      n--;
    }
    if (!n) n = 1;
    m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)P*n)));
    if (!m) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (M > P && m < p) {PetscInt _m = m; m = p; p = _m;}
  } else if (m*n*p != size) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition");

  if (m*n*p != size) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Could not find good partition");
  if (M < m) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);
  if (P < p) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Partition in z direction is too fine! %D %D",P,p);

  stag->nRanks[0] = m;
  stag->nRanks[1] = n;
  stag->nRanks[2] = p;
  PetscFunctionReturn(0);
}

/* Determine ranks of neighbors, using DMDA's convention

        n24 n25 n26
        n21 n22 n23
        n18 n19 n20 (Front, bigger z)

        n15 n16 n17
        n12     n14   ^ y
        n9  n10 n11   |
                      +--> x
        n6  n7  n8
        n3  n4  n5
        n0  n1  n2 (Back, smaller z) */
static PetscErrorCode DMStagSetUpBuildNeighbors_3d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        d,i;
  PetscBool       per[3],first[3],last[3];
  PetscInt        neighborRank[27][3],r[3],n[3];
  const PetscInt  dim = 3;

  PetscFunctionBegin;
  for (d=0; d<dim; ++d) if (stag->boundaryType[d] != DM_BOUNDARY_NONE && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->boundaryType[d] != DM_BOUNDARY_GHOSTED) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Neighbor determination not implemented for %s",DMBoundaryTypes[stag->boundaryType[d]]);

  /* Assemble some convenience variables */
  for (d=0; d<dim; ++d) {
    per[d]   = (PetscBool)(stag->boundaryType[d] == DM_BOUNDARY_PERIODIC);
    first[d] = stag->firstRank[d];
    last[d]  = stag->lastRank[d];
    r[d]     = stag->rank[d];
    n[d]     = stag->nRanks[d];
  }

  /* First, compute the position in the rank grid for all neighbors */

  neighborRank[0][0]  = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  down back  */
  neighborRank[0][1]  = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[0][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[1][0]  =                                      r[0]    ; /*       down back  */
  neighborRank[1][1]  = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[1][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[2][0]  = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right down back  */
  neighborRank[2][1]  = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[2][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[3][0]  = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left       back  */
  neighborRank[3][1]  =                                      r[1]    ;
  neighborRank[3][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[4][0]  =                                      r[0]    ; /*            back  */
  neighborRank[4][1]  =                                      r[1]    ;
  neighborRank[4][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[5][0]  = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right      back  */
  neighborRank[5][1]  =                                      r[1]    ;
  neighborRank[5][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[6][0]  = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  up   back  */
  neighborRank[6][1]  = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[6][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[7][0]  =                                      r[0]    ; /*       up   back  */
  neighborRank[7][1]  = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[7][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[8][0]  = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right up   back  */
  neighborRank[8][1]  = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[8][2]  = first[2] ? (per[2] ?  n[2]-1 : -1) : r[2] - 1;

  neighborRank[9][0]  = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  down       */
  neighborRank[9][1]  = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[9][2]  =                                      r[2]    ;

  neighborRank[10][0] =                                      r[0]    ; /*       down       */
  neighborRank[10][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[10][2] =                                      r[2]    ;

  neighborRank[11][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right down       */
  neighborRank[11][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[11][2] =                                      r[2]    ;

  neighborRank[12][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left             */
  neighborRank[12][1] =                                      r[1]    ;
  neighborRank[12][2] =                                      r[2]    ;

  neighborRank[13][0] =                                      r[0]    ; /*                  */
  neighborRank[13][1] =                                      r[1]    ;
  neighborRank[13][2] =                                      r[2]    ;

  neighborRank[14][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right            */
  neighborRank[14][1] =                                      r[1]    ;
  neighborRank[14][2] =                                      r[2]    ;

  neighborRank[15][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  up         */
  neighborRank[15][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[15][2] =                                      r[2]    ;

  neighborRank[16][0] =                                      r[0]    ; /*       up         */
  neighborRank[16][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[16][2] =                                      r[2]    ;

  neighborRank[17][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right up         */
  neighborRank[17][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[17][2] =                                      r[2]    ;

  neighborRank[18][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  down front */
  neighborRank[18][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[18][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[19][0] =                                      r[0]    ; /*       down front */
  neighborRank[19][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[19][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[20][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right down front */
  neighborRank[20][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;
  neighborRank[20][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[21][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left       front */
  neighborRank[21][1] =                                      r[1]    ;
  neighborRank[21][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[22][0] =                                      r[0]    ; /*            front */
  neighborRank[22][1] =                                      r[1]    ;
  neighborRank[22][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[23][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right      front */
  neighborRank[23][1] =                                      r[1]    ;
  neighborRank[23][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[24][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  up   front */
  neighborRank[24][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[24][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[25][0] =                                      r[0]    ; /*       up   front */
  neighborRank[25][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[25][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  neighborRank[26][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right up   front */
  neighborRank[26][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;
  neighborRank[26][2] = last[2]  ? (per[2] ?  0      : -1) : r[2] + 1;

  /* Then, compute the rank of each in the linear ordering */
  ierr = PetscMalloc1(27,&stag->neighbors);CHKERRQ(ierr);
  for (i=0; i<27; ++i){
    if  (neighborRank[i][0] >= 0 && neighborRank[i][1] >=0 && neighborRank[i][2] >=0) {
      stag->neighbors[i] = neighborRank[i][0] + n[0]*neighborRank[i][1] + n[0]*n[1]*neighborRank[i][2];
    } else {
      stag->neighbors[i] = -1;
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagSetUpBuildGlobalOffsets_3d(DM dm,PetscInt **pGlobalOffsets)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              *globalOffsets;
  PetscInt              i,j,k,d,entriesPerEdge,entriesPerFace,count;
  PetscMPIInt           size;
  PetscBool             extra[3];

  PetscFunctionBegin;
  for (d=0; d<3; ++d) extra[d] = (PetscBool)(stag->boundaryType[d] != DM_BOUNDARY_PERIODIC); /* Extra points in global rep */
  entriesPerFace               = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
  entriesPerEdge               = stag->dof[0] + stag->dof[1];
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,pGlobalOffsets);CHKERRQ(ierr);
  globalOffsets = *pGlobalOffsets;
  globalOffsets[0] = 0;
  count = 1; /* note the count is offset by 1 here. We add the size of the previous rank */
  for (k=0; k<stag->nRanks[2]-1; ++k){
    const PetscInt nnk = stag->l[2][k];
    for (j=0; j<stag->nRanks[1]-1; ++j) {
      const PetscInt nnj = stag->l[1][j];
      for (i=0; i<stag->nRanks[0]-1; ++i) {
        const PetscInt nni = stag->l[0][i];
        /* Interior : No right/top/front boundaries */
        globalOffsets[count] = globalOffsets[count-1] + nni*nnj*nnk*stag->entriesPerElement;
        ++count;
      }
      {
        /* Right boundary - extra faces */
        /* i = stag->nRanks[0]-1; */
        const PetscInt nni = stag->l[0][i];
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[0] ? nnj*nnk*entriesPerFace : 0);
        ++count;
      }
    }
    {
      /* j = stag->nRanks[1]-1; */
      const PetscInt nnj = stag->l[1][j];
      for (i=0; i<stag->nRanks[0]-1; ++i) {
        const PetscInt nni = stag->l[0][i];
        /* Up boundary - extra faces */
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[1] ? nni*nnk*entriesPerFace : 0);
        ++count;
      }
      {
        /* i = stag->nRanks[0]-1; */
        const PetscInt nni = stag->l[0][i];
        /* Up right boundary - 2x extra faces and extra edges */
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[0]             ? nnj*nnk*entriesPerFace : 0)
                               + (extra[1]             ? nni*nnk*entriesPerFace : 0)
                               + (extra[0] && extra[1] ? nnk*entriesPerEdge     : 0);
        ++count;
      }
    }
  }
  {
    /* k = stag->nRanks[2]-1; */
    const PetscInt nnk = stag->l[2][k];
    for (j=0; j<stag->nRanks[1]-1; ++j) {
      const PetscInt nnj = stag->l[1][j];
      for (i=0; i<stag->nRanks[0]-1; ++i) {
        const PetscInt nni = stag->l[0][i];
        /* Front boundary - extra faces */
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[2] ? nni*nnj*entriesPerFace : 0);
        ++count;
      }
      {
        /* i = stag->nRanks[0]-1; */
        const PetscInt nni = stag->l[0][i];
        /* Front right boundary - 2x extra faces and extra edges */
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[0]             ? nnk*nnj*entriesPerFace : 0)
                               + (extra[2]             ? nni*nnj*entriesPerFace : 0)
                               + (extra[0] && extra[2] ? nnj*entriesPerEdge     : 0);
        ++count;
      }
    }
    {
      /* j = stag->nRanks[1]-1; */
      const PetscInt nnj = stag->l[1][j];
      for (i=0; i<stag->nRanks[0]-1; ++i) {
        const PetscInt nni = stag->l[0][i];
        /* Front up boundary - 2x extra faces and extra edges */
        globalOffsets[count] = globalOffsets[count-1] + nnj*nni*nnk*stag->entriesPerElement
                               + (extra[1]             ? nnk*nni*entriesPerFace : 0)
                               + (extra[2]             ? nnj*nni*entriesPerFace : 0)
                               + (extra[1] && extra[2] ? nni*entriesPerEdge     : 0);
        ++count;
      }
      /* Don't need to compute entries in last element */
    }
  }

  PetscFunctionReturn(0);
}

/* A helper function to reduce code duplication as we loop over various ranges.
   i,j,k refer to element numbers on the rank where an element lives in the global
   representation (without ghosts) to be offset by a global offset per rank.
   ig,jg,kg refer to element numbers in the local (this rank) ghosted numbering.
   Note that this function could easily be converted to a macro (it does not compute
   anything except loop indices and the values of idxGlobal and idxLocal).  */
static PetscErrorCode DMStagSetUpBuildScatterPopulateIdx_3d(DM_Stag *stag,PetscInt *count,PetscInt *idxLocal,PetscInt *idxGlobal,PetscInt entriesPerEdge, PetscInt entriesPerFace,PetscInt eprNeighbor,PetscInt eplNeighbor,PetscInt eprGhost,PetscInt eplGhost,PetscInt epFaceRow,PetscInt globalOffset,PetscInt startx,PetscInt starty,PetscInt startz,PetscInt startGhostx,PetscInt startGhosty,PetscInt startGhostz,PetscInt endGhostx,PetscInt endGhosty,PetscInt endGhostz, PetscBool extrax,PetscBool extray,PetscBool extraz)
{
  PetscInt ig,jg,kg,d,c;

  PetscFunctionBegin;
  c = *count;
  for (kg = startGhostz; kg < endGhostz; ++kg) {
    const PetscInt k = kg - startGhostz + startz;
    for (jg = startGhosty; jg < endGhosty; ++jg) {
      const PetscInt j = jg - startGhosty + starty;
      for (ig = startGhostx; ig<endGhostx; ++ig) {
        const PetscInt i = ig - startGhostx + startx;
        for (d=0; d<stag->entriesPerElement; ++d,++c) {
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *stag->entriesPerElement + d;
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + d;
        }
      }
      if (extrax) {
        PetscInt dLocal;
        const PetscInt i = endGhostx - startGhostx + startx;
        ig = endGhostx;
        for (d=0,dLocal=0; d<stag->dof[0]; ++d, ++dLocal, ++c) { /* Vertex */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j*eprNeighbor + i *stag->entriesPerElement + d;
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost   + ig*stag->entriesPerElement + dLocal;
        }
        dLocal += stag->dof[1]; /* Skip back down edge */
        for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++c) { /* Back left edge */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *stag->entriesPerElement + d;
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        dLocal += stag->dof[2]; /* Skip back face */
        for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++c) { /* Down left edge */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *stag->entriesPerElement + d;
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        dLocal += stag->dof[2]; /* Skip down face */
        for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++c) { /* Left face */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *stag->entriesPerElement + d;
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        /* Skip element */
      }
    }
    if (extray) {
      const PetscInt j = endGhosty - startGhosty + starty;
      jg = endGhosty;
      for (ig = startGhostx; ig<endGhostx; ++ig) {
        const PetscInt i = ig - startGhostx + startx;
        /* Vertex and Back down edge */
        PetscInt dLocal;
        for (d=0,dLocal=0; d<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++c) { /* Vertex */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *entriesPerFace          + d; /* Note face increment in  x */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        /* Skip back left edge and back face */
        dLocal += stag->dof[1] + stag->dof[2];
        /* Down face and down left edge */
        for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++c) { /* Back left edge */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j*eprNeighbor + i *entriesPerFace          + d; /* Note face increment in x */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost   + ig*stag->entriesPerElement + dLocal;
        }
        /* Skip left face and element */
      }
      if (extrax) {
        PetscInt dLocal;
        const PetscInt i = endGhostx - startGhostx + startx;
        ig = endGhostx;
        for (d=0,dLocal=0; d<stag->dof[0]; ++d, ++dLocal, ++c) { /* Vertex */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *entriesPerFace          + d; /* Note face increment in x */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        dLocal += 2*stag->dof[1]+stag->dof[2]; /* Skip back down edge, back face, and back left edge */
        for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++c) { /* Down left edge */
          idxGlobal[c] = globalOffset + k *eplNeighbor + j *eprNeighbor + i *entriesPerFace          + d; /* Note face increment in x */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost    + ig*stag->entriesPerElement + dLocal;
        }
        /* Skip remaining entries */
      }
    }
  }
  if (extraz) {
    const PetscInt k = endGhostz - startGhostz + startz;
    kg = endGhostz;
    for (jg = startGhosty; jg < endGhosty; ++jg) {
      const PetscInt j = jg - startGhosty + starty;
      for (ig = startGhostx; ig<endGhostx; ++ig) {
        const PetscInt i = ig - startGhostx + startx;
        for (d=0; d<entriesPerFace; ++d, ++c) {
          idxGlobal[c] = globalOffset + k*eplNeighbor  + j *epFaceRow + i *entriesPerFace          + d; /* Note face-based x and y increments */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost  + ig*stag->entriesPerElement + d;
        }
      }
      if (extrax) {
        PetscInt dLocal;
        const PetscInt i = endGhostx - startGhostx + startx;
        ig = endGhostx;
        for (d=0,dLocal=0; d<stag->dof[0]; ++d, ++dLocal, ++c) { /* Vertex */
          idxGlobal[c] = globalOffset + k*eplNeighbor + j *epFaceRow + i *entriesPerFace          + d; /* Note face-based x and y increments */
          idxLocal[c]  =                kg*eplGhost   + jg*eprGhost  + ig*stag->entriesPerElement + dLocal;
        }
        dLocal += stag->dof[1]; /* Skip back down edge */
        for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++c) { /* Back left edge */
          idxGlobal[c] = globalOffset + k*eplNeighbor  + j *epFaceRow + i*entriesPerFace           + d; /* Note face-based x and y increments */
          idxLocal[c]  =                kg*eplGhost    + jg*eprGhost  + ig*stag->entriesPerElement + dLocal;
        }
        /* Skip the rest */
      }
    }
    if (extray) {
      const PetscInt j = endGhosty - startGhosty + starty;
      jg = endGhosty;
      for (ig = startGhostx; ig<endGhostx; ++ig) {
        const PetscInt i = ig - startGhostx + startx;
        for (d=0; d<entriesPerEdge; ++d,++c) {
          idxGlobal[c] = globalOffset + k*eplNeighbor + j *epFaceRow + i *entriesPerEdge          + d; /* Note face-based y increment and edge-based x increment */
          idxLocal[c]  =                kg*eplGhost   + jg*eprGhost  + ig*stag->entriesPerElement + d;
        }
      }
      if (extrax) {
        const PetscInt i = endGhostx - startGhostx + startx;
        ig = endGhostx;
        for (d=0; d<stag->dof[0]; ++d, ++c) { /* Vertex (only) */
          idxGlobal[c] = globalOffset + k*eplNeighbor + j *epFaceRow + i *entriesPerEdge          + d; /* Note face-based y increment and edge-based x increment */
          idxLocal[c]  =                kg*eplGhost   + jg*eprGhost  + ig*stag->entriesPerElement + d;
        }
      }
    }
  }
  *count = c;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagSetUpBuildScatter_3d(DM dm,const PetscInt *globalOffsets)
{
  PetscErrorCode ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt       d,ghostOffsetStart[3],ghostOffsetEnd[3],entriesPerCorner,entriesPerEdge,entriesPerFace,entriesToTransferTotal,count,eprGhost,eplGhost;
  PetscInt       *idxLocal,*idxGlobal;
  PetscMPIInt    rank;
  PetscInt       nNeighbors[27][3];
  PetscBool      star,nextToDummyEnd[3],dummyStart[3],dummyEnd[3];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  if (stag->n[0] < stag->stencilWidth || stag->n[1] < stag->stencilWidth || stag->n[2] < stag->stencilWidth) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStag 3d setup does not support local sizes (%D x %D x %D) smaller than the elementwise stencil width (%D)",stag->n[0],stag->n[1],stag->n[2],stag->stencilWidth);
  }

  /* Check stencil type */
  if (stag->stencilType != DMSTAG_STENCIL_NONE && stag->stencilType != DMSTAG_STENCIL_BOX && stag->stencilType != DMSTAG_STENCIL_STAR) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported stencil type %s",DMStagStencilTypes[stag->stencilType]);
  if (stag->stencilType == DMSTAG_STENCIL_NONE && stag->stencilWidth != 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMStag 3d setup requires stencil width 0 with stencil type none");
  star = (PetscBool)(stag->stencilType == DMSTAG_STENCIL_STAR);

  /* Compute numbers of elements on each neighbor */
  {
    PetscInt i;
    for (i=0; i<27; ++i) {
      const PetscInt neighborRank = stag->neighbors[i];
      if (neighborRank >= 0) { /* note we copy the values for our own rank (neighbor 13) */
        nNeighbors[i][0] =  stag->l[0][neighborRank % stag->nRanks[0]];
        nNeighbors[i][1] =  stag->l[1][neighborRank % (stag->nRanks[0] * stag->nRanks[1]) / stag->nRanks[0]];
        nNeighbors[i][2] =  stag->l[2][neighborRank / (stag->nRanks[0] * stag->nRanks[1])];
      } /* else leave uninitialized - error if accessed */
    }
  }

  /* These offsets should always be non-negative, and describe how many
     ghost elements exist at each boundary. These are not always equal to the stencil width,
     because we may have different numbers of ghost elements at the boundaries. In particular,
     in the non-periodic casewe always have at least one ghost (dummy) element at the right/top/front. */
  for (d=0; d<3; ++d) ghostOffsetStart[d] = stag->start[d] - stag->startGhost[d];
  for (d=0; d<3; ++d) ghostOffsetEnd[d]   = stag->startGhost[d]+stag->nGhost[d] - (stag->start[d]+stag->n[d]);

  /* Determine whether the ghost region includes dummies or not. This is currently
     equivalent to having a non-periodic boundary. If not, then
     ghostOffset{Start,End}[d] elements correspond to elements on the neighbor.
     If true, then
     - at the start, there are ghostOffsetStart[d] ghost elements
     - at the end, there is a layer of extra "physical" points inside a layer of
       ghostOffsetEnd[d] ghost elements
     Note that this computation should be updated if any boundary types besides
     NONE, GHOSTED, and PERIODIC are supported.  */
  for (d=0; d<3; ++d) dummyStart[d] = (PetscBool)(stag->firstRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
  for (d=0; d<3; ++d) dummyEnd[d]   = (PetscBool)(stag->lastRank[d]  && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);

  /* Convenience variables */
  entriesPerFace                      = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
  entriesPerEdge                      = stag->dof[0] + stag->dof[1];
  entriesPerCorner                    = stag->dof[0];
  for (d=0;d<3;++d) nextToDummyEnd[d] = (PetscBool)(stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->rank[d] == stag->nRanks[d]-2);
  eprGhost                            = stag->nGhost[0]*stag->entriesPerElement; /* epr = entries per (element) row   */
  eplGhost                            = stag->nGhost[1]*eprGhost;                /* epl = entries per (element) layer */

  /* Compute the number of local entries which correspond to any global entry */
  {
    PetscInt nNonDummyGhost[3];
    for (d=0; d<3; ++d) nNonDummyGhost[d] = stag->nGhost[d] - (dummyStart[d] ? ghostOffsetStart[d] : 0) - (dummyEnd[d] ? ghostOffsetEnd[d] : 0);
    if (star) {
      entriesToTransferTotal = (
          nNonDummyGhost[0] * stag->n[1]        * stag->n[2]
        + stag->n[0]        * nNonDummyGhost[1] * stag->n[2]
        + stag->n[0]        * stag->n[1]        * nNonDummyGhost[2]
        - 2 * (stag->n[0] * stag->n[1] * stag->n[2])) * stag->entriesPerElement
        + (dummyEnd[0]                               ? (nNonDummyGhost[1] * stag->n[2] + stag->n[1] * nNonDummyGhost[2] - stag->n[1] * stag->n[2]) * entriesPerFace   : 0)
        + (dummyEnd[1]                               ? (nNonDummyGhost[0] * stag->n[2] + stag->n[0] * nNonDummyGhost[2] - stag->n[0] * stag->n[2]) * entriesPerFace   : 0)
        + (dummyEnd[2]                               ? (nNonDummyGhost[0] * stag->n[1] + stag->n[0] * nNonDummyGhost[1] - stag->n[0] * stag->n[1]) * entriesPerFace   : 0)
        + (dummyEnd[0] && dummyEnd[1]                ? nNonDummyGhost[2]                     * entriesPerEdge   : 0)
        + (dummyEnd[2] && dummyEnd[0]                ? nNonDummyGhost[1]                     * entriesPerEdge   : 0)
        + (dummyEnd[1] && dummyEnd[2]                ? nNonDummyGhost[0]                     * entriesPerEdge   : 0)
        + (dummyEnd[0] && dummyEnd[1] && dummyEnd[2] ?                                         entriesPerCorner : 0);
    } else {
      entriesToTransferTotal  =  nNonDummyGhost[0] * nNonDummyGhost[1] * nNonDummyGhost[2] * stag->entriesPerElement
        + (dummyEnd[0]                               ? nNonDummyGhost[1] * nNonDummyGhost[2] * entriesPerFace   : 0)
        + (dummyEnd[1]                               ? nNonDummyGhost[2] * nNonDummyGhost[0] * entriesPerFace   : 0)
        + (dummyEnd[2]                               ? nNonDummyGhost[0] * nNonDummyGhost[1] * entriesPerFace   : 0)
        + (dummyEnd[0] && dummyEnd[1]                ? nNonDummyGhost[2]                     * entriesPerEdge   : 0)
        + (dummyEnd[2] && dummyEnd[0]                ? nNonDummyGhost[1]                     * entriesPerEdge   : 0)
        + (dummyEnd[1] && dummyEnd[2]                ? nNonDummyGhost[0]                     * entriesPerEdge   : 0)
        + (dummyEnd[0] && dummyEnd[1] && dummyEnd[2] ?                                         entriesPerCorner : 0);
    }
  }

  /* Allocate arrays to populate */
  ierr = PetscMalloc1(entriesToTransferTotal,&idxLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(entriesToTransferTotal,&idxGlobal);CHKERRQ(ierr);

  /* Counts into idxLocal/idxGlobal */
  count = 0;

  /*  Loop over each of the 27 neighbor, populating idxLocal and idxGlobal. These
      cases are principally distinguished by

      1. The loop bounds (i/ighost,j/jghost,k/kghost)
      2. the strides used in the loop body, which depend on whether the current
      rank and/or its neighbor are a non-periodic right/top/front boundary, which has additional
      points in the global representation.
      - If the neighboring rank is a right/top/front boundary,
      then eprNeighbor (entries per element row on the neighbor) and/or eplNeighbor (entries per element layer on the neighbor)
      are different.
      - If this rank is a non-periodic right/top/front boundary (checking entries of stag->lastRank),
      there is an extra loop over 1-3 boundary faces)

      Here, we do not include "dummy" dof (points in the local representation which
      do not correspond to any global dof). This, and the fact that we iterate over points in terms of
      increasing global ordering, are the main two differences from the construction of
      the Local-to-global map, which iterates over points in local order, and does include dummy points. */

  /* LEFT DOWN BACK */
  if (!star && !dummyStart[0] && !dummyStart[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 0;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* DOWN BACK */
  if (!star && !dummyStart[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 1;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT DOWN BACK */
  if (!star && !dummyEnd[0] && !dummyStart[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 2;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT BACK */
  if (!star && !dummyStart[0] && !dummyStart[2]) {
    const PetscInt  neighbor     = 3;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0);  /* May be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* BACK */
  if (!dummyStart[2]) {
    const PetscInt  neighbor     = 4;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may  be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* We+neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT BACK */
  if (!star && !dummyEnd[0] && !dummyStart[2]) {
    const PetscInt  neighbor     = 5;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT UP BACK */
  if (!star && !dummyStart[0] && !dummyEnd[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 6;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* UP BACK */
  if (!star && !dummyEnd[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 7;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = stag->nGhost[1]-ghostOffsetEnd[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT UP BACK */
  if (!star && !dummyEnd[0] && !dummyEnd[1] && !dummyStart[2]) {
    const PetscInt  neighbor     = 8;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = nNeighbors[neighbor][2] - ghostOffsetStart[2];
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = 0;
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = ghostOffsetStart[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT DOWN */
  if (!star && !dummyStart[0] && !dummyStart[1]) {
    const PetscInt  neighbor     = 9;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0]; /* Note that we can't be a right boundary */
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* DOWN */
  if (!dummyStart[1]) {
    const PetscInt  neighbor     = 10;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT DOWN */
  if (!star && !dummyEnd[0] && !dummyStart[1]) {
    const PetscInt  neighbor     = 11;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT */
  if (!dummyStart[0]) {
    const PetscInt  neighbor     = 12;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0);  /* We+neighbor may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0];
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* (HERE) */
  {
    const PetscInt  neighbor     = 13;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* We may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT */
  if (!dummyEnd[0]) {
    const PetscInt  neighbor     = 14;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT UP */
  if (!star && !dummyStart[0] && !dummyEnd[1]) {
    const PetscInt  neighbor     = 15;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0];
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* UP */
  if (!dummyEnd[1]) {
    const PetscInt  neighbor     = 16;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT UP */
  if (!star && !dummyEnd[0] && !dummyEnd[1]) {
    const PetscInt  neighbor     = 17;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT DOWN FRONT */
  if (!star && !dummyStart[0] && !dummyStart[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 18;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* DOWN FRONT */
  if (!star && !dummyStart[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 19;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT DOWN FRONT */
  if (!star && !dummyEnd[0] && !dummyStart[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 20;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1];
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = nNeighbors[neighbor][1] - ghostOffsetStart[1];
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = 0;
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = ghostOffsetStart[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT FRONT */
  if (!star && !dummyStart[0] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 21;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0);  /* We+neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* FRONT */
  if (!dummyEnd[2]) {
    const PetscInt  neighbor     = 22;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* neighbor is a right boundary if we are*/
    const PetscInt  eplNeighbor  = eprNeighbor  * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* May be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT FRONT */
  if (!star && !dummyEnd[0] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 23;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (dummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* LEFT UP FRONT */
  if (!star && !dummyStart[0] && !dummyEnd[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 24;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0];
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = nNeighbors[neighbor][0] - ghostOffsetStart[0];
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = 0;
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = ghostOffsetStart[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* UP FRONT */
  if (!star && !dummyEnd[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 25;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  /* RIGHT UP FRONT */
  if (!star && !dummyEnd[0] && !dummyEnd[1] && !dummyEnd[2]) {
    const PetscInt  neighbor     = 26;
    const PetscInt  eprNeighbor  = stag->entriesPerElement * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    const PetscInt  eplNeighbor  = eprNeighbor * nNeighbors[neighbor][1] + (nextToDummyEnd[1] ? nNeighbors[neighbor][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
    const PetscInt  epFaceRow    = -1;
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  startGhost1  = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  startGhost2  = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscInt  endGhost0    = stag->nGhost[0];
    const PetscInt  endGhost1    = stag->nGhost[1];
    const PetscInt  endGhost2    = stag->nGhost[2];
    const PetscBool extra0       = PETSC_FALSE;
    const PetscBool extra1       = PETSC_FALSE;
    const PetscBool extra2       = PETSC_FALSE;
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,eprNeighbor,eplNeighbor,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }

  if (count != entriesToTransferTotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of entries computed in gtol (%d) is not as expected (%d)",count,entriesToTransferTotal);

  /* Create stag->gtol. The order is computed as PETSc ordering, and doesn't include dummy entries */
  {
    Vec local,global;
    IS isLocal,isGlobal;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
    ierr = VecScatterCreate(global,isGlobal,local,isLocal,&stag->gtol);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
    ierr = VecDestroy(&local);CHKERRQ(ierr);
    ierr = ISDestroy(&isLocal);CHKERRQ(ierr);  /* frees idxLocal */
    ierr = ISDestroy(&isGlobal);CHKERRQ(ierr); /* free idxGlobal */
  }

  PetscFunctionReturn(0);
}

/* Note: this function assumes that DMBoundary types of none, ghosted, and periodic are the only ones of interest.
Adding support for others should be done very carefully.  */
static PetscErrorCode DMStagSetUpBuildL2G_3d(DM dm,const PetscInt *globalOffsets)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              *idxGlobalAll;
  PetscInt              d,count,ighost,jghost,kghost,ghostOffsetStart[3],ghostOffsetEnd[3],entriesPerFace,entriesPerEdge;
  PetscInt              nNeighbors[27][3],eprNeighbor[27],eplNeighbor[27],globalOffsetNeighbor[27];
  PetscBool             nextToDummyEnd[3],dummyStart[3],dummyEnd[3],star;

  PetscFunctionBegin;

  /* Check stencil type */
  if (stag->stencilType != DMSTAG_STENCIL_NONE && stag->stencilType != DMSTAG_STENCIL_BOX && stag->stencilType != DMSTAG_STENCIL_STAR) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported stencil type %s",DMStagStencilTypes[stag->stencilType]);
  if (stag->stencilType == DMSTAG_STENCIL_NONE && stag->stencilWidth != 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMStag 3d setup requires stencil width 0 with stencil type none");
  star = (PetscBool)(stag->stencilType == DMSTAG_STENCIL_STAR);

  /* Convenience variables */
  entriesPerFace                      = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
  entriesPerEdge                      = stag->dof[0] + stag->dof[1];
  for (d=0;d<3;++d) nextToDummyEnd[d] = (PetscBool)(stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->rank[d] == stag->nRanks[d]-2);

  /* Ghost offsets (may not be the ghost width, since we always have at least one ghost element on the right/top/front) */
  for (d=0; d<3; ++d) ghostOffsetStart[d] = stag->start[d] - stag->startGhost[d];
  for (d=0; d<3; ++d) ghostOffsetEnd[d]   = stag->startGhost[d]+stag->nGhost[d] - (stag->start[d]+stag->n[d]);

  /* Whether the ghost region includes dummies. Currently equivalent to being a non-periodic boundary. */
  for (d=0; d<3; ++d) dummyStart[d] = (PetscBool)(stag->firstRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
  for (d=0; d<3; ++d) dummyEnd[d]   = (PetscBool)(stag->lastRank[d]  && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);

  /* Compute numbers of elements on each neighbor  and offset*/
  {
    PetscInt i;
    for (i=0; i<27; ++i) {
      const PetscInt neighborRank = stag->neighbors[i];
      if (neighborRank >= 0) { /* note we copy the values for our own rank (neighbor 13) */
        nNeighbors[i][0] =  stag->l[0][neighborRank % stag->nRanks[0]];
        nNeighbors[i][1] =  stag->l[1][neighborRank % (stag->nRanks[0] * stag->nRanks[1]) / stag->nRanks[0]];
        nNeighbors[i][2] =  stag->l[2][neighborRank / (stag->nRanks[0] * stag->nRanks[1])];
        globalOffsetNeighbor[i] = globalOffsets[stag->neighbors[i]];
      } /* else leave uninitialized - error if accessed */
    }
  }

  /* Precompute elements per row and layer on neighbor (zero unused) */
  ierr = PetscMemzero(eprNeighbor,sizeof(eprNeighbor));CHKERRQ(ierr);
  ierr = PetscMemzero(eplNeighbor,sizeof(eplNeighbor));CHKERRQ(ierr);
  if (stag->neighbors[0] >= 0) {
    eprNeighbor[0] = stag->entriesPerElement * nNeighbors[0][0];
    eplNeighbor[0] = eprNeighbor[0] * nNeighbors[0][1];
  }
  if (stag->neighbors[1] >= 0) {
    eprNeighbor[1] = stag->entriesPerElement * nNeighbors[1][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[1] = eprNeighbor[1] * nNeighbors[1][1];
  }
  if (stag->neighbors[2] >= 0) {
    eprNeighbor[2] = stag->entriesPerElement * nNeighbors[2][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[2] = eprNeighbor[2] * nNeighbors[2][1];
  }
  if (stag->neighbors[3] >= 0) {
    eprNeighbor[3] = stag->entriesPerElement * nNeighbors[3][0];
    eplNeighbor[3] = eprNeighbor[3] * nNeighbors[3][1] + (dummyEnd[1] ? nNeighbors[3][0] * entriesPerFace : 0);  /* May be a top boundary */
  }
  if (stag->neighbors[4] >= 0) {
    eprNeighbor[4] = stag->entriesPerElement * nNeighbors[4][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may  be a right boundary */
    eplNeighbor[4] = eprNeighbor[4] * nNeighbors[4][1] + (dummyEnd[1] ? nNeighbors[4][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* We+neighbor may be a top boundary */
  }
  if (stag->neighbors[5] >= 0) {
    eprNeighbor[5] = stag->entriesPerElement * nNeighbors[5][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[5] = eprNeighbor[5]  * nNeighbors[5][1] + (dummyEnd[1] ? nNeighbors[5][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
  }
  if (stag->neighbors[6] >= 0) {
    eprNeighbor[6] = stag->entriesPerElement * nNeighbors[6][0];
    eplNeighbor[6] = eprNeighbor[6] * nNeighbors[6][1] + (nextToDummyEnd[1] ? nNeighbors[6][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[7] >= 0) {
    eprNeighbor[7] = stag->entriesPerElement * nNeighbors[7][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[7] = eprNeighbor[7] * nNeighbors[7][1] + (nextToDummyEnd[1] ? nNeighbors[7][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[8] >= 0) {
    eprNeighbor[8] = stag->entriesPerElement * nNeighbors[8][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[8] = eprNeighbor[8] * nNeighbors[8][1] + (nextToDummyEnd[1] ? nNeighbors[8][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[9] >= 0) {
    eprNeighbor[9] = stag->entriesPerElement * nNeighbors[9][0];
    eplNeighbor[9] = eprNeighbor[9] * nNeighbors[9][1];
  }
  if (stag->neighbors[10] >= 0) {
    eprNeighbor[10] = stag->entriesPerElement * nNeighbors[10][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[10] = eprNeighbor[10] * nNeighbors[10][1];
  }
  if (stag->neighbors[11] >= 0) {
    eprNeighbor[11] = stag->entriesPerElement * nNeighbors[11][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[11] = eprNeighbor[11] * nNeighbors[11][1];
  }
  if (stag->neighbors[12] >= 0) {
    eprNeighbor[12] = stag->entriesPerElement * nNeighbors[12][0];
    eplNeighbor[12] = eprNeighbor[12] * nNeighbors[12][1] + (dummyEnd[1] ? nNeighbors[12][0] * entriesPerFace : 0);  /* We+neighbor may be a top boundary */
  }
  if (stag->neighbors[13] >= 0) {
    eprNeighbor[13] = stag->entriesPerElement * nNeighbors[13][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We may be a right boundary */
    eplNeighbor[13] = eprNeighbor[13] * nNeighbors[13][1] + (dummyEnd[1] ? nNeighbors[13][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* We may be a top boundary */
  }
  if (stag->neighbors[14] >= 0) {
    eprNeighbor[14] = stag->entriesPerElement * nNeighbors[14][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[14] = eprNeighbor[14] * nNeighbors[14][1] + (dummyEnd[1] ? nNeighbors[14][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
  }
  if (stag->neighbors[15] >= 0) {
    eprNeighbor[15] = stag->entriesPerElement * nNeighbors[15][0];
    eplNeighbor[15] = eprNeighbor[15] * nNeighbors[15][1] + (nextToDummyEnd[1] ? nNeighbors[15][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[16] >= 0) {
    eprNeighbor[16] = stag->entriesPerElement * nNeighbors[16][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[16] = eprNeighbor[16] * nNeighbors[16][1] + (nextToDummyEnd[1] ? nNeighbors[16][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[17] >= 0) {
    eprNeighbor[17] = stag->entriesPerElement * nNeighbors[17][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[17] = eprNeighbor[17] * nNeighbors[17][1] + (nextToDummyEnd[1] ? nNeighbors[17][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[18] >= 0) {
    eprNeighbor[18] = stag->entriesPerElement * nNeighbors[18][0];
    eplNeighbor[18] = eprNeighbor[18] * nNeighbors[18][1];
  }
  if (stag->neighbors[19] >= 0) {
    eprNeighbor[19] = stag->entriesPerElement * nNeighbors[19][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[19] = eprNeighbor[19] * nNeighbors[19][1];
  }
  if (stag->neighbors[20] >= 0) {
    eprNeighbor[20] = stag->entriesPerElement * nNeighbors[20][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[20] = eprNeighbor[20] * nNeighbors[20][1];
  }
  if (stag->neighbors[21] >= 0) {
    eprNeighbor[21] = stag->entriesPerElement * nNeighbors[21][0];
    eplNeighbor[21] = eprNeighbor[21] * nNeighbors[21][1] + (dummyEnd[1] ? nNeighbors[21][0] * entriesPerFace : 0);  /* We+neighbor may be a top boundary */
  }
  if (stag->neighbors[22] >= 0) {
    eprNeighbor[22] = stag->entriesPerElement * nNeighbors[22][0] + (dummyEnd[0] ? entriesPerFace : 0); /* neighbor is a right boundary if we are*/
    eplNeighbor[22] = eprNeighbor[22] * nNeighbors[22][1] + (dummyEnd[1] ? nNeighbors[22][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* May be a top boundary */
  }
  if (stag->neighbors[23] >= 0) {
    eprNeighbor[23] = stag->entriesPerElement * nNeighbors[23][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[23] = eprNeighbor[23] * nNeighbors[23][1] + (dummyEnd[1] ? nNeighbors[23][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* We and neighbor may be a top boundary */
  }
  if (stag->neighbors[24] >= 0) {
    eprNeighbor[24] = stag->entriesPerElement * nNeighbors[24][0];
    eplNeighbor[24] = eprNeighbor[24] * nNeighbors[24][1] + (nextToDummyEnd[1] ? nNeighbors[24][0] * entriesPerFace : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[25] >= 0) {
    eprNeighbor[25] = stag->entriesPerElement * nNeighbors[25][0] + (dummyEnd[0] ? entriesPerFace : 0); /* We+neighbor may be a right boundary */
    eplNeighbor[25] = eprNeighbor[25] * nNeighbors[25][1] + (nextToDummyEnd[1] ? nNeighbors[25][0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }
  if (stag->neighbors[26] >= 0) {
    eprNeighbor[26] = stag->entriesPerElement * nNeighbors[26][0] + (nextToDummyEnd[0] ? entriesPerFace : 0); /* Neighbor may be a right boundary */
    eplNeighbor[26] = eprNeighbor[26] * nNeighbors[26][1] + (nextToDummyEnd[1] ? nNeighbors[26][0] * entriesPerFace + (nextToDummyEnd[0] ? entriesPerEdge : 0) : 0); /* Neighbor may be a top boundary */
  }

  /* Populate idxGlobalAll */
  ierr = PetscMalloc1(stag->entriesGhost,&idxGlobalAll);CHKERRQ(ierr);
  count = 0;

  /* Loop over layers 1/3 : Back */
  if (!dummyStart[2]) {
    for (kghost = 0; kghost<ghostOffsetStart[2]; ++kghost) {
      const PetscInt k = nNeighbors[4][2] - ghostOffsetStart[2] + kghost; /* Note: this is the same value for all neighbors in this layer (use neighbor 4 which will always exist if we're lookng at this layer) */

      /* Loop over rows 1/3: Down Back*/
      if (!star && !dummyStart[1]) {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          const PetscInt j = nNeighbors[1][1] - ghostOffsetStart[1] + jghost; /* Note: this is the same value for all neighbors in this row (use neighbor 1, down back)*/

          /* Loop over columns 1/3: Left Back Down*/
          if (!dummyStart[0]) {
            const PetscInt neighbor = 0;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Down Back */
          {
            const PetscInt neighbor = 1;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Down Back */
          if (!dummyEnd[0]) {
            const PetscInt neighbor = 2;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            /* Partial dummy entries on (Middle) Down Back neighbor */
            const PetscInt neighbor          = 1;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }

      /* Loop over rows 2/3: (Middle) Back */
      {
        for (jghost = ghostOffsetStart[1]; jghost<stag->nGhost[1]-ghostOffsetEnd[1]; ++jghost) {
          const PetscInt j = jghost - ghostOffsetStart[1];

          /* Loop over columns 1/3: Left (Middle) Back */
          if (!star && !dummyStart[0]) {
            const PetscInt neighbor = 3;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) (Middle) Back */
          {
            const PetscInt neighbor = 4;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right (Middle) Back */
          if (!star && !dummyEnd[0]) {
            const PetscInt neighbor = 5;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else if (dummyEnd[0]) {
            /* Partial dummy entries on (Middle) (Middle) Back rank */
            const PetscInt neighbor = 4;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor]+ j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          } else {
            /* Right (Middle) Back dummies */
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      }

      /* Loop over rows 3/3: Up Back */
      if (!star && !dummyEnd[1]) {
        PetscInt j;
        for (j=0; j<ghostOffsetEnd[1]; ++j) {
          /* Loop over columns 1/3: Left Up Back*/
          if (!dummyStart[0]) {
            const PetscInt neighbor = 6;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Up Back*/
          {
            const PetscInt neighbor = 7;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Up Back*/
          if (!dummyEnd[0]) {
            const PetscInt neighbor = 8;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            /* Partial dummies on (Middle) Up Back */
            const PetscInt neighbor = 7;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else if (dummyEnd[1]) {
        /* Up Back partial dummy row */
        PetscInt j = stag->n[1];

        if (!star && !dummyStart[0]) {
          /* Up Back partial dummy row: Loop over columns 1/3: Left Up Back, on Left (Middle) Back rank */
          const PetscInt neighbor = 3;
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }

            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Up Back partial dummy row: Loop over columns 2/3: (Middle) Up Back, on (Middle) (Middle) Back rank */
        {
          const PetscInt neighbor = 4;
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Up Back partial dummy row: Loop over columns 3/3: Right Up Back, on Right (Middle) Back rank */
        if (!star && !dummyEnd[0]) {
          const PetscInt neighbor = 5;
          PetscInt       i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        } else if (dummyEnd[0]) {
          /* Up Right Back partial dummy element, on (Middle) (Middle) Back rank */
          const PetscInt neighbor = 4;
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back down edge, back face and back left edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies down face, left face and element */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          /* Up Right Back dummies */
          PetscInt i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
        ++j;
        /* Up Back additional dummy rows */
        for (; j<stag->n[1] + ghostOffsetEnd[1]; ++j) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      } else {
        /* Up Back dummy rows */
        PetscInt j;
        for (j=0; j<ghostOffsetEnd[1]; ++j) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    }
  } else {
    for (kghost = 0; kghost<ghostOffsetStart[2]; ++kghost) {
      for (jghost = 0; jghost<stag->nGhost[1]; ++jghost) {
        for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }
    }
  }

  /* Loop over layers 2/3 : (Middle)  */
  {
    for (kghost = ghostOffsetStart[2]; kghost<stag->nGhost[2]-ghostOffsetEnd[2]; ++kghost) {
      const PetscInt k = kghost - ghostOffsetStart[2];

      /* Loop over rows 1/3: Down (Middle) */
      if (!dummyStart[1]) {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          const PetscInt j = nNeighbors[10][1] - ghostOffsetStart[1] + jghost; /* down neighbor (10) always exists */

          /* Loop over columns 1/3: Left Down (Middle) */
          if (!star && !dummyStart[0]) {
            const PetscInt neighbor = 9;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Down (Middle) */
          {
            const PetscInt neighbor = 10;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Down (Middle) */
          if (!star && !dummyEnd[0]) {
            const PetscInt neighbor = 11;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else if (dummyEnd[0]) {
            /* Partial dummies on (Middle) Down (Middle) neighbor */
            const PetscInt neighbor = 10;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor]+ j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          } else {
            /* Right Down (Middle) dummies */
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }

      /* Loop over rows 2/3: (Middle) (Middle) */
      {
        for (jghost = ghostOffsetStart[1]; jghost<stag->nGhost[1]-ghostOffsetEnd[1]; ++jghost) {
          const PetscInt j = jghost - ghostOffsetStart[1];

          /* Loop over columns 1/3: Left (Middle) (Middle) */
          if (!dummyStart[0]) {
            const PetscInt neighbor = 12;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) (Middle) (Middle) aka (Here) */
          {
            const PetscInt neighbor = 13;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right (Middle) (Middle) */
          if (!dummyEnd[0]) {
            const PetscInt neighbor = 14;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            /* Partial dummies on this rank */
            const PetscInt neighbor = 13;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor]+ j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (;i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      }

      /* Loop over rows 3/3: Up (Middle) */
      if (!dummyEnd[1]) {
        PetscInt j;
        for (j=0; j<ghostOffsetEnd[1]; ++j) {

          /* Loop over columns 1/3: Left Up (Middle) */
          if (!star && !dummyStart[0]) {
            const PetscInt neighbor = 15;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Up (Middle) */
          {
            const PetscInt neighbor = 16;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Up (Middle) */
          if (!star && !dummyEnd[0]) {
            const PetscInt neighbor = 17;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else if (dummyEnd[0]) {
            /* Partial dummies on (Middle) Up (Middle) rank */
            const PetscInt neighbor = 16;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          } else {
            /* Right Up (Middle) dumies */
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else {
        /* Up (Middle) partial dummy row */
        PetscInt j = stag->n[1];

        /* Up (Middle) partial dummy row: colums 1/3: Left Up (Middle), on Left (Middle) (Middle) rank */
        if (!dummyStart[0]) {
          const PetscInt neighbor = 12;
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Up (Middle) partial dummy row: columns 2/3: (Middle) Up (Middle), on (Middle) (Middle) (Middle) = here rank */
        {
          const PetscInt neighbor = 13;
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        }

        if (!dummyEnd[0]) {
          /* Up (Middle) partial dummy row: columns 3/3: Right Up (Middle), on Right (Middle) (Middle) rank */
          const PetscInt neighbor = 14;
          PetscInt       i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          /* Up (Middle) Right partial dummy element, on (Middle) (Middle) (Middle) = here rank */
          const PetscInt neighbor = 13;
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back down edge, back face and back left edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies down face, left face and element */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
        ++j;
        /* Up (Middle) additional dummy rows */
        for (; j<stag->n[1] + ghostOffsetEnd[1]; ++j) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    }
  }

  /* Loop over layers 3/3 : Front */
  if (!dummyEnd[2]) {
    PetscInt k;
    for (k=0; k<ghostOffsetEnd[2]; ++k) {

      /* Loop over rows 1/3: Down Front */
      if (!star && !dummyStart[1]) {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          const PetscInt j = nNeighbors[19][1] - ghostOffsetStart[1] + jghost; /* Constant for whole row (use down front neighbor) */

          /* Loop over columns 1/3: Left Down Front */
          if (!dummyStart[0]) {
            const PetscInt neighbor = 18;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Down Front */
          {
            const PetscInt neighbor = 19;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) { /* vertex, 2 edges, and face associated with back face */
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Down Front */
          if (!dummyEnd[0]) {
            const PetscInt neighbor = 20;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            /* Partial dummy element on (Middle) Down Front rank */
            const PetscInt neighbor = 19;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else {
        for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }

      /* Loop over rows 2/3: (Middle) Front */
      {
        for (jghost = ghostOffsetStart[1]; jghost<stag->nGhost[1]-ghostOffsetEnd[1]; ++jghost) {
          const PetscInt j = jghost - ghostOffsetStart[1];

          /* Loop over columns 1/3: Left (Middle) Front*/
          if (!star && !dummyStart[0]) {
            const PetscInt neighbor = 21;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) (Middle) Front*/
          {
            const PetscInt neighbor = 22;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right (Middle) Front*/
          if (!star && !dummyEnd[0]) {
            const PetscInt neighbor = 23;
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else if (dummyEnd[0]) {
            /* Partial dummy element on (Middle) (Middle) Front element */
            const PetscInt neighbor = 22;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          } else {
            /* Right (Middle) Front dummies */
            PetscInt       i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      }

      /* Loop over rows 3/3: Up Front */
      if (!star && !dummyEnd[1]) {
        PetscInt j;
        for (j=0; j<ghostOffsetEnd[1]; ++j) {

          /* Loop over columns 1/3: Left Up Front */
          if (!dummyStart[0]) {
            const PetscInt neighbor = 24;
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }

          /* Loop over columns 2/3: (Middle) Up Front */
          {
            const PetscInt neighbor = 25;
            for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
              const PetscInt i = ighost - ghostOffsetStart[0];
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          }

          /* Loop over columns 3/3: Right Up Front */
          if (!dummyEnd[0]) {
            const PetscInt neighbor = 26;
            PetscInt     i;
            for (i=0; i<ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
              }
            }
          } else {
            /* Partial dummy element on (Middle) Up Front rank */
            const PetscInt neighbor = 25;
            PetscInt       i,dLocal;
            i = stag->n[0];
            for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* Back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++dLocal, ++count) { /* Dummies on down face */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+3*stag->dof[2]; ++d, ++dLocal, ++count) { /* Left face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*stag->entriesPerElement + d;
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on element */
              idxGlobalAll[count] = -1;
            }
            ++i;
            for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
              for (d=0; d<stag->entriesPerElement; ++d,++count) {
                idxGlobalAll[count] = -1;
              }
            }
          }
        }
      } else if (dummyEnd[1]) {
        /* Up Front partial dummy row */
        PetscInt j = stag->n[1];

        /* Up Front partial dummy row: columns 1/3: Left Up Front, on Left (Middle) Front rank */
        if (!star && !dummyStart[0]) {
          const PetscInt neighbor = 21;
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Up Front partial dummy row: columns 2/3: (Middle) Up Front, on (Middle) (Middle) Front rank */
        {
          const PetscInt neighbor = 22;
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }
        }

        if (!star && !dummyEnd[0]) {
          /* Up Front partial dummy row: columns 3/3: Right Up Front, on Right (Middle) Front rank */
          const PetscInt neighbor = 23;
          PetscInt       i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt       dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex and back down edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back face and back left edge */
              idxGlobalAll[count] = -1;
            }
            for (; dLocal<stag->dof[0]+3*stag->dof[1]+2*stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge and down face */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies left face and element */
              idxGlobalAll[count] = -1;
            }
          }

        } else if (dummyEnd[0]) {
          /* Partial Right Up Front dummy element, on (Middle) (Middle) Front rank */
          const PetscInt neighbor = 22;
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++dLocal,++count) { /* Dummies on back down edge, back face and back left edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+3*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Down left edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*eprNeighbor[neighbor] + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies down face, left face and element */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          /* Right Up Front dummies */
          PetscInt i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
        ++j;
        /* Up Front additional dummy rows */
        for (; j<stag->n[1] + ghostOffsetEnd[1]; ++j) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      } else {
        /* Up Front dummies */
        PetscInt j;
        for (j=0; j<ghostOffsetEnd[1]; ++j) {
          for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    }
  } else {
    PetscInt k = stag->n[2];

    /* Front partial ghost layer: rows 1/3: Down Front, on Down (Middle) */
    if (!dummyStart[1]) {
      for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
        const PetscInt j = nNeighbors[10][1] - ghostOffsetStart[1] + jghost; /* wrt down neighbor (10) */

        /* Down Front partial ghost row: colums 1/3: Left Down Front, on  Left Down (Middle) */
        if (!star && !dummyStart[0]) {
          const PetscInt neighbor = 9;
          const PetscInt epFaceRow         = entriesPerFace * nNeighbors[neighbor][0]; /* Note that we can't be a right boundary */
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Down Front partial ghost row: columns 2/3: (Middle) Down Front, on (Middle) Down (Middle) */
        {
          const PetscInt neighbor = 10;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        }

        if (!star && !dummyEnd[0]) {
          /* Down Front partial dummy row: columns 3/3: Right Down Front, on Right Down (Middle) */
          const PetscInt neighbor = 11;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
          PetscInt       i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else if (dummyEnd[0]) {
          /* Right Down Front partial dummy element, living on (Middle) Down (Middle) rank */
          const PetscInt neighbor = 10;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* left back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          PetscInt i;
          /* Right Down Front dummies */
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    } else {
      for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
        for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }
    }

    /* Front partial ghost layer: rows 2/3: (Middle) Front, on (Middle) (Middle) */
    {
      for (jghost = ghostOffsetStart[1]; jghost<stag->nGhost[1]-ghostOffsetEnd[1]; ++jghost) {
        const PetscInt j = jghost - ghostOffsetStart[1];

        /* (Middle) Front partial dummy row: columns 1/3: Left (Middle) Front, on Left (Middle) (Middle) */
        if (!dummyStart[0]) {
          const PetscInt neighbor = 12;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0];
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* (Middle) Front partial dummy row: columns 2/3: (Middle) (Middle) Front, on (Middle) (Middle) (Middle) */
        {
          const PetscInt neighbor = 13;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        }

        if (!dummyEnd[0]) {
          /* (Middle) Front partial dummy row: columns 3/3: Right (Middle) Front, on Right (Middle) (Middle) */
          const PetscInt neighbor = 14;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
          PetscInt i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          /* Right (Middle) Front partial dummy element, on (Middle) (Middle) (Middle) rank */
          const PetscInt neighbor = 13;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* left back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    }

    /* Front partial ghost layer: rows 3/3: Up Front, on Up (Middle) */
    if (!dummyEnd[1]) {
      PetscInt j;
      for (j=0; j<ghostOffsetEnd[1]; ++j) {

        /* Up Front partial dummy row: columns 1/3: Left Up Front, on Left Up (Middle) */
        if (!star && !dummyStart[0]) {
          const PetscInt neighbor = 15;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0];
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }

        /* Up Front partial dummy row: columns 2/3: (Middle) Up Front, on (Middle) Up (Middle) */
        {
          const PetscInt neighbor = 16;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
          for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
            const PetscInt i = ighost - ghostOffsetStart[0];
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        }

        if (!star && !dummyEnd[0]) {
          /* Up Front partial dummy row: columsn 3/3: Right Up Front, on Right Up (Middle) */
          const PetscInt neighbor = 17;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
          PetscInt       i;
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            PetscInt  dLocal;
            for (d=0,dLocal=0; dLocal<stag->dof[0]+2*stag->dof[1]+stag->dof[2]; ++d, ++dLocal, ++count) { /* Vertex, back down edge, back face, back left edge */
              idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note j increment by face */
            }
            for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining points */
              idxGlobalAll[count] = -1;
            }
          }
        } else if (dummyEnd[0]) {
          /* Right Up Front partial dummy element, on (Middle) Up (Middle) */
          const PetscInt neighbor = 16;
          const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We+neighbor may be a right boundary */
          PetscInt       i,dLocal;
          i = stag->n[0];
          for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->dof[0]+stag->dof[1]; ++dLocal,++count) { /* Dummies on back down edge */
            idxGlobalAll[count] = -1;
          }
          for (; dLocal<stag->dof[0]+2*stag->dof[1]; ++d, ++dLocal, ++count) { /* left back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerFace + d; /* Note i increment by face */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
          ++i;
          for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        } else {
          /* Right Up Front dummies */
          PetscInt i;
          /* Right Down Front dummies */
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d,++count) {
              idxGlobalAll[count] = -1;
            }
          }
        }
      }
    } else {
      /* Up Front partial dummy row */
      PetscInt j = stag->n[1];

      /* Up Front partial dummy row: columns 1/3: Left Up Front, on Left (Middle) (Middle) */
      if (!dummyStart[0]) {
        const PetscInt neighbor = 12;
        const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0];
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          const PetscInt i = nNeighbors[neighbor][0] - ghostOffsetStart[0] + ighost;
          PetscInt       dLocal;
          for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex  and down back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerEdge + d; /* Note increments */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
        }
      } else {
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }

      /* Up Front partial dummy row: columns 2/3: (Middle) Up Front, on (Middle) (Middle) (Middle) */
      {
        const PetscInt neighbor = 13;
        const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
        for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
          const PetscInt i = ighost - ghostOffsetStart[0];
          PetscInt       dLocal;
          for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex  and down back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerEdge + d; /* Note increments */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
        }
      }

      if (!dummyEnd[0]) {
        /* Up Front partial dummy row: columns 3/3: Right Up Front, on Right (Middle) (Middle) */
        const PetscInt neighbor = 14;
        const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (nextToDummyEnd[0] ? entriesPerEdge : 0); /* Neighbor may be a right boundary */
        PetscInt       i;
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          PetscInt       dLocal;
          for (d=0,dLocal=0; dLocal<stag->dof[0]+stag->dof[1]; ++d, ++dLocal, ++count) { /* Vertex  and down back edge */
            idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerEdge + d; /* Note increments */
          }
          for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
            idxGlobalAll[count] = -1;
          }
        }
      } else {
        /* Right Up Front partial dummy element, on this rank */
        const PetscInt neighbor = 13;
        const PetscInt epFaceRow = entriesPerFace * nNeighbors[neighbor][0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
        PetscInt       i,dLocal;
        i = stag->n[0];
        for (d=0,dLocal=0; dLocal<stag->dof[0]; ++d, ++dLocal, ++count) { /* Vertex */
          idxGlobalAll[count] = globalOffsetNeighbor[neighbor] + k*eplNeighbor[neighbor] + j*epFaceRow + i*entriesPerEdge + d; /* Note increments */
        }
        for (; dLocal<stag->entriesPerElement; ++dLocal, ++count) { /* Dummies on remaining */
          idxGlobalAll[count] = -1;
        }
        ++i;
        for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }
      ++j;
      /* Up Front additional dummy rows */
      for (; j<stag->n[1] + ghostOffsetEnd[1]; ++j) {
        for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }
    }
    /* Front additional dummy layers */
    ++k;
    for (; k<stag->n[2] + ghostOffsetEnd[2]; ++k) {
      for (jghost = 0; jghost<stag->nGhost[1]; ++jghost) {
        for (ighost = 0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobalAll[count] = -1;
          }
        }
      }
    }
  }

  /* Create local-to-global map (in local ordering, includes maps to -1 for dummy points) */
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm),1,stag->entriesGhost,idxGlobalAll,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)dm->ltogmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagComputeLocationOffsets_3d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt epe = stag->entriesPerElement;
  const PetscInt epr = stag->nGhost[0]*epe;
  const PetscInt epl = stag->nGhost[1]*epr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(DMSTAG_NUMBER_LOCATIONS,&stag->locationOffsets);CHKERRQ(ierr);
  stag->locationOffsets[DMSTAG_BACK_DOWN_LEFT]   = 0;
  stag->locationOffsets[DMSTAG_BACK_DOWN]        = stag->locationOffsets[DMSTAG_BACK_DOWN_LEFT]  + stag->dof[0];
  stag->locationOffsets[DMSTAG_BACK_DOWN_RIGHT]  = stag->locationOffsets[DMSTAG_BACK_DOWN_LEFT]  + epe;
  stag->locationOffsets[DMSTAG_BACK_LEFT]        = stag->locationOffsets[DMSTAG_BACK_DOWN]       + stag->dof[1];
  stag->locationOffsets[DMSTAG_BACK]             = stag->locationOffsets[DMSTAG_BACK_LEFT]       + stag->dof[1];
  stag->locationOffsets[DMSTAG_BACK_RIGHT]       = stag->locationOffsets[DMSTAG_BACK_LEFT]       + epe;
  stag->locationOffsets[DMSTAG_BACK_UP_LEFT]     = stag->locationOffsets[DMSTAG_BACK_DOWN_LEFT]  + epr;
  stag->locationOffsets[DMSTAG_BACK_UP]          = stag->locationOffsets[DMSTAG_BACK_DOWN]       + epr;
  stag->locationOffsets[DMSTAG_BACK_UP_RIGHT]    = stag->locationOffsets[DMSTAG_BACK_UP_LEFT]    + epe;
  stag->locationOffsets[DMSTAG_DOWN_LEFT]        = stag->locationOffsets[DMSTAG_BACK]            + stag->dof[2];
  stag->locationOffsets[DMSTAG_DOWN]             = stag->locationOffsets[DMSTAG_DOWN_LEFT]       + stag->dof[1];
  stag->locationOffsets[DMSTAG_DOWN_RIGHT]       = stag->locationOffsets[DMSTAG_DOWN_LEFT]       + epe;
  stag->locationOffsets[DMSTAG_LEFT]             = stag->locationOffsets[DMSTAG_DOWN]            + stag->dof[2];
  stag->locationOffsets[DMSTAG_ELEMENT]          = stag->locationOffsets[DMSTAG_LEFT]            + stag->dof[2];
  stag->locationOffsets[DMSTAG_RIGHT]            = stag->locationOffsets[DMSTAG_LEFT]            + epe;
  stag->locationOffsets[DMSTAG_UP_LEFT]          = stag->locationOffsets[DMSTAG_DOWN_LEFT]       + epr;
  stag->locationOffsets[DMSTAG_UP]               = stag->locationOffsets[DMSTAG_DOWN]            + epr;
  stag->locationOffsets[DMSTAG_UP_RIGHT]         = stag->locationOffsets[DMSTAG_UP_LEFT]         + epe;
  stag->locationOffsets[DMSTAG_FRONT_DOWN_LEFT]  = stag->locationOffsets[DMSTAG_BACK_DOWN_LEFT]  + epl;
  stag->locationOffsets[DMSTAG_FRONT_DOWN]       = stag->locationOffsets[DMSTAG_BACK_DOWN]       + epl;
  stag->locationOffsets[DMSTAG_FRONT_DOWN_RIGHT] = stag->locationOffsets[DMSTAG_FRONT_DOWN_LEFT] + epe;
  stag->locationOffsets[DMSTAG_FRONT_LEFT]       = stag->locationOffsets[DMSTAG_BACK_LEFT]       + epl;
  stag->locationOffsets[DMSTAG_FRONT]            = stag->locationOffsets[DMSTAG_BACK]            + epl;
  stag->locationOffsets[DMSTAG_FRONT_RIGHT]      = stag->locationOffsets[DMSTAG_FRONT_LEFT]      + epe;
  stag->locationOffsets[DMSTAG_FRONT_UP_LEFT]    = stag->locationOffsets[DMSTAG_FRONT_DOWN_LEFT] + epr;
  stag->locationOffsets[DMSTAG_FRONT_UP]         = stag->locationOffsets[DMSTAG_FRONT_DOWN]      + epr;
  stag->locationOffsets[DMSTAG_FRONT_UP_RIGHT]   = stag->locationOffsets[DMSTAG_FRONT_UP_LEFT]   + epe;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_3d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        *idxLocal,*idxGlobal,*globalOffsetsRecomputed;
  const PetscInt  *globalOffsets;
  PetscInt        count,d,entriesPerEdge,entriesPerFace,eprGhost,eplGhost,ghostOffsetStart[3],ghostOffsetEnd[3];
  IS              isLocal,isGlobal;
  PetscBool       dummyEnd[3];

  PetscFunctionBegin;
  ierr = DMStagSetUpBuildGlobalOffsets_3d(dm,&globalOffsetsRecomputed);CHKERRQ(ierr); /* note that we don't actually use all of these. An available optimization is to pass them, when available */
  globalOffsets = globalOffsetsRecomputed;
  ierr = PetscMalloc1(stag->entries,&idxLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(stag->entries,&idxGlobal);CHKERRQ(ierr);
  for (d=0; d<3; ++d) dummyEnd[d]   = (PetscBool)(stag->lastRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
  entriesPerFace                      = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
  entriesPerEdge                      = stag->dof[0] + stag->dof[1];
  eprGhost                            = stag->nGhost[0]*stag->entriesPerElement; /* epr = entries per (element) row   */
  eplGhost                            = stag->nGhost[1]*eprGhost;                /* epl = entries per (element) layer */
  count = 0;
  for (d=0; d<3; ++d) ghostOffsetStart[d] = stag->start[d] - stag->startGhost[d];
  for (d=0; d<3; ++d) ghostOffsetEnd[d]   = stag->startGhost[d]+stag->nGhost[d] - (stag->start[d]+stag->n[d]);
  {
    const PetscInt  neighbor     = 13;
    const PetscInt  epr          = stag->entriesPerElement * stag->n[0] + (dummyEnd[0] ? entriesPerFace : 0); /* We may be a right boundary */
    const PetscInt  epl          = epr                     * stag->n[1] + (dummyEnd[1] ? stag->n[0] * entriesPerFace + (dummyEnd[0] ? entriesPerEdge : 0) : 0); /* We may be a top boundary */
    const PetscInt  epFaceRow    = entriesPerFace          * stag->n[0] + (dummyEnd[0] ? entriesPerEdge : 0); /* We may be a right boundary */
    const PetscInt  start0       = 0;
    const PetscInt  start1       = 0;
    const PetscInt  start2       = 0;
    const PetscInt  startGhost0  = ghostOffsetStart[0];
    const PetscInt  startGhost1  = ghostOffsetStart[1];
    const PetscInt  startGhost2  = ghostOffsetStart[2];
    const PetscInt  endGhost0    = stag->nGhost[0] - ghostOffsetEnd[0];
    const PetscInt  endGhost1    = stag->nGhost[1] - ghostOffsetEnd[1];
    const PetscInt  endGhost2    = stag->nGhost[2] - ghostOffsetEnd[2];
    const PetscBool extra0       = dummyEnd[0];
    const PetscBool extra1       = dummyEnd[1];
    const PetscBool extra2       = dummyEnd[2];
    ierr = DMStagSetUpBuildScatterPopulateIdx_3d(stag,&count,idxLocal,idxGlobal,entriesPerEdge,entriesPerFace,epr,epl,eprGhost,eplGhost,epFaceRow,globalOffsets[stag->neighbors[neighbor]],start0,start1,start2,startGhost0,startGhost1,startGhost2,endGhost0,endGhost1,endGhost2,extra0,extra1,extra2);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);
  {
    Vec local,global;
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
    ierr = VecScatterCreate(local,isLocal,global,isGlobal,&stag->ltog_injective);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
    ierr = VecDestroy(&local);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);
  if (globalOffsetsRecomputed) {
    ierr = PetscFree(globalOffsetsRecomputed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
