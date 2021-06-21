/* Functions specific to the 2-dimensional implementation of DMStag */
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagCreate2d - Create an object to manage data living on the faces, edges, and vertices of a parallelized regular 2D grid.

  Collective

  Input Parameters:
+ comm - MPI communicator
. bndx,bndy - boundary type: DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, or DM_BOUNDARY_GHOSTED
. M,N - global number of grid points in x,y directions
. m,n - number of ranks in the x,y directions (may be PETSC_DECIDE)
. dof0 - number of degrees of freedom per vertex/point/node/0-cell
. dof1 - number of degrees of freedom per edge/1-cell
. dof2 - number of degrees of freedom per element/2-cell
. stencilType - ghost/halo region type: DMSTAG_STENCIL_NONE, DMSTAG_STENCIL_BOX, or DMSTAG_STENCIL_STAR
. stencilWidth - width, in elements, of halo/ghost region
- lx,ly - arrays of local x,y element counts, of length equal to m,n, summing to M,N

  Output Parameter:
. dm - the new DMStag object

  Options Database Keys:
+ -dm_view - calls DMViewFromOptions() a the conclusion of DMSetUp()
. -stag_grid_x <nx> - number of elements in the x direction
. -stag_grid_y <ny> - number of elements in the y direction
. -stag_ranks_x <rx> - number of ranks in the x direction
. -stag_ranks_y <ry> - number of ranks in the y direction
. -stag_ghost_stencil_width - width of ghost region, in elements
. -stag_boundary_type_x <none,ghosted,periodic> - DMBoundaryType value
- -stag_boundary_type_y <none,ghosted,periodic> - DMBoundaryType value

  Notes:
  You must call DMSetUp() after this call, before using the DM.
  If you wish to use the options database (see the keys above) to change values in the DMStag, you must call
  DMSetFromOptions() after this function but before DMSetUp().

  Level: beginner

.seealso: DMSTAG, DMStagCreate1d(), DMStagCreate3d(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateLocalVector(), DMLocalToGlobalBegin(), DMDACreate2d()
@*/
PETSC_EXTERN PetscErrorCode DMStagCreate2d(MPI_Comm comm, DMBoundaryType bndx,DMBoundaryType bndy, PetscInt M,PetscInt N, PetscInt m,PetscInt n, PetscInt dof0, PetscInt dof1, PetscInt dof2, DMStagStencilType stencilType, PetscInt stencilWidth, const PetscInt lx[], const PetscInt ly[],DM* dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm,2);CHKERRQ(ierr);
  ierr = DMStagInitialize(bndx,bndy,DM_BOUNDARY_NONE,M,N,0,m,n,0,dof0,dof1,dof2,0,stencilType,stencilWidth,lx,ly,NULL,*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_2d(DM dm,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax)
{
  PetscErrorCode ierr;
  DM_Stag        *stagCoord;
  DM             dmCoord;
  Vec            coordLocal;
  PetscReal      h[2],min[2];
  PetscScalar    ***arr;
  PetscInt       ind[2],start_ghost[2],n_ghost[2],s,c;
  PetscInt       idownleft,idown,ileft,ielement;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  stagCoord = (DM_Stag*) dmCoord->data;
  for (s=0; s<3; ++s) {
    if (stagCoord->dof[s] !=0 && stagCoord->dof[s] != 2) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Coordinate DM in 2 dimensions must have 0 or 2 dof on each stratum, but stratum %d has %d dof",s,stagCoord->dof[s]);
  }
  ierr = DMCreateLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  if (stagCoord->dof[0]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_DOWN_LEFT,0,&idownleft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[1]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_DOWN     ,0,&idown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_LEFT     ,0,&ileft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[2]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_ELEMENT  ,0,&ielement);CHKERRQ(ierr);
  }
  ierr = DMStagGetGhostCorners(dmCoord,&start_ghost[0],&start_ghost[1],NULL,&n_ghost[0],&n_ghost[1],NULL);CHKERRQ(ierr);

  min[0] = xmin; min[1]= ymin;
  h[0] = (xmax-xmin)/stagCoord->N[0];
  h[1] = (ymax-ymin)/stagCoord->N[1];

  for (ind[1]=start_ghost[1]; ind[1]<start_ghost[1] + n_ghost[1]; ++ind[1]) {
    for (ind[0]=start_ghost[0]; ind[0]<start_ghost[0] + n_ghost[0]; ++ind[0]) {
      if (stagCoord->dof[0]) {
        const PetscReal offs[2] = {0.0,0.0};
        for (c=0; c<2; ++c) {
          arr[ind[1]][ind[0]][idownleft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
        }
      }
      if (stagCoord->dof[1]) {
        const PetscReal offs[2] = {0.5,0.0};
        for (c=0; c<2; ++c) {
          arr[ind[1]][ind[0]][idown + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
        }
      }
      if (stagCoord->dof[1]) {
        const PetscReal offs[2] = {0.0,0.5};
        for (c=0; c<2; ++c) {
          arr[ind[1]][ind[0]][ileft + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
        }
      }
      if (stagCoord->dof[2]) {
        const PetscReal offs[2] = {0.5,0.5};
        for (c=0; c<2; ++c) {
          arr[ind[1]][ind[0]][ielement + c] = min[c] + ((PetscReal)ind[c] + offs[c]) * h[c];
        }
      }
    }
  }
  ierr = DMStagVecRestoreArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm,coordLocal);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)coordLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&coordLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Helper functions used in DMSetUp_Stag() */
static PetscErrorCode DMStagSetUpBuildRankGrid_2d(DM);
static PetscErrorCode DMStagSetUpBuildNeighbors_2d(DM);
static PetscErrorCode DMStagSetUpBuildGlobalOffsets_2d(DM,PetscInt**);
static PetscErrorCode DMStagComputeLocationOffsets_2d(DM);

PETSC_INTERN PetscErrorCode DMSetUp_Stag_2d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscMPIInt     size,rank;
  PetscInt        i,j,d,entriesPerElementRowGhost,entriesPerCorner,entriesPerEdge,entriesPerElementRow;
  MPI_Comm        comm;
  PetscInt        *globalOffsets;
  PetscBool       star,dummyStart[2],dummyEnd[2];
  const PetscInt  dim = 2;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* Rank grid sizes (populates stag->nRanks) */
  ierr = DMStagSetUpBuildRankGrid_2d(dm);CHKERRQ(ierr);

  /* Determine location of rank in grid (these get extra boundary points on the last element)
     Order is x-fast, as usual */
    stag->rank[0] = rank % stag->nRanks[0];
    stag->rank[1] = rank / stag->nRanks[0];
    for (i=0; i<dim; ++i) {
      stag->firstRank[i] = PetscNot(stag->rank[i]);
      stag->lastRank[i]  = (PetscBool)(stag->rank[i] == stag->nRanks[i]-1);
    }

  /* Determine Locally owned region

   Divide equally, giving lower ranks in each dimension and extra element if needbe.

   Note that this uses O(P) storage. If this ever becomes an issue, this could
   be refactored to not keep this data around.  */
  for (i=0; i<dim; ++i) {
    if (!stag->l[i]) {
      const PetscInt Ni = stag->N[i], nRanksi = stag->nRanks[i];
      ierr = PetscMalloc1(stag->nRanks[i],&stag->l[i]);CHKERRQ(ierr);
      for (j=0; j<stag->nRanks[i]; ++j) {
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
    for (j=0;j<stag->rank[i];++j) {
      stag->start[i] += stag->l[i][j];
    }
  }

  /* Determine ranks of neighbors, using DMDA's convention

     n6 n7 n8
     n3    n5
     n0 n1 n2                                               */
  ierr = DMStagSetUpBuildNeighbors_2d(dm);CHKERRQ(ierr);

    /* Determine whether the ghost region includes dummies or not. This is currently
       equivalent to having a non-periodic boundary. If not, then
       ghostOffset{Start,End}[d] elements correspond to elements on the neighbor.
       If true, then
       - at the start, there are ghostOffsetStart[d] ghost elements
       - at the end, there is a layer of extra "physical" points inside a layer of
         ghostOffsetEnd[d] ghost elements
       Note that this computation should be updated if any boundary types besides
       NONE, GHOSTED, and PERIODIC are supported.  */
    for (d=0; d<2; ++d) dummyStart[d] = (PetscBool)(stag->firstRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
    for (d=0; d<2; ++d) dummyEnd[d]   = (PetscBool)(stag->lastRank[d]  && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);

  /* Define useful sizes */
  stag->entriesPerElement = stag->dof[0] + 2*stag->dof[1] + stag->dof[2];
  entriesPerEdge          = stag->dof[0] + stag->dof[1];
  entriesPerCorner        = stag->dof[0];
  entriesPerElementRow    = stag->n[0]*stag->entriesPerElement + (dummyEnd[0] ? entriesPerEdge : 0);
  stag->entries           = stag->n[1]*entriesPerElementRow +  (dummyEnd[1] ? stag->n[0]*entriesPerEdge : 0) + (dummyEnd[0] && dummyEnd[1] ? entriesPerCorner: 0);

  /* Compute offsets for each rank into global vectors
     This again requires O(P) storage, which could be replaced with some global
     communication.  */
  ierr = DMStagSetUpBuildGlobalOffsets_2d(dm,&globalOffsets);CHKERRQ(ierr);

  for (d=0; d<dim; ++d) if (stag->boundaryType[d] != DM_BOUNDARY_NONE && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->boundaryType[d] != DM_BOUNDARY_GHOSTED) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported boundary type");

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
            SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
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
            SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
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
            SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
        }
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported boundary type in dimension %D",d);
    }
  }
  stag->entriesGhost = stag->nGhost[0]*stag->nGhost[1]*stag->entriesPerElement;
  entriesPerElementRowGhost = stag->nGhost[0]*stag->entriesPerElement;

  /* Create global-->local VecScatter and local->global ISLocalToGlobalMapping

     We iterate over all local points twice. First, we iterate over each neighbor, populating
     1. idxLocal[] : the subset of points, in local numbering ("S" from 0 on all points including ghosts), which correspond to global points. That is, the set of all non-dummy points in the ghosted representation
     2. idxGlobal[]: the corresponding global points, in global numbering (Nested "S"s - ranks then non-ghost points in each rank)

     Next, we iterate over all points in the local ordering, populating
     3. idxGlobalAll[] : entry i is the global point corresponding to local point i, or -1 if local point i is a dummy.

     Note further here that the local/ghosted vectors:
     - Are always an integral number of elements-worth of points, in all directions.
     - Contain three flavors of points:
     1. Points which "live here" in the global representation
     2. Ghost points which correspond to points on other ranks in the global representation
     3. Ghost points, which we call "dummy points," which do not correspond to any point in the global representation

     Dummy ghost points arise in at least three ways:
     1. As padding for the right, top, and front physical boundaries, to complete partial elements
     2. As unused space in the "corners" on interior ranks when using a star stencil
     3. As additional work space on all physical boundaries, when DM_BOUNDARY_GHOSTED is used

     Note that, because of the boundary dummies,
     with a stencil width of zero, on 1 rank, local and global vectors
     are still different!

     We assume that the size on each rank is greater than or equal to the
     stencil width.
     */

  if (stag->n[0] < stag->stencilWidth || stag->n[1] < stag->stencilWidth) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStag 2d setup does not support local sizes (%D x %D) smaller than the elementwise stencil width (%D)",stag->n[0],stag->n[1],stag->stencilWidth);
  }

  /* Check stencil type */
  if (stag->stencilType != DMSTAG_STENCIL_NONE && stag->stencilType != DMSTAG_STENCIL_BOX && stag->stencilType != DMSTAG_STENCIL_STAR) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported stencil type %s",DMStagStencilTypes[stag->stencilType]);
  if (stag->stencilType == DMSTAG_STENCIL_NONE && stag->stencilWidth != 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMStag 2d setup requires stencil width 0 with stencil type none");
  star = (PetscBool)(stag->stencilType == DMSTAG_STENCIL_STAR);

  {
    PetscInt *idxLocal,*idxGlobal,*idxGlobalAll;
    PetscInt  count,countAll,entriesToTransferTotal,i,j,d,ghostOffsetStart[2],ghostOffsetEnd[2];
    IS        isLocal,isGlobal;
    PetscInt  jghost,ighost;
    PetscInt  nNeighbors[9][2];
    PetscBool nextToDummyEnd[2];

    /* Compute numbers of elements on each neighbor */
    for (i=0; i<9; ++i) {
      const PetscInt neighborRank = stag->neighbors[i];
      if (neighborRank >= 0) { /* note we copy the values for our own rank (neighbor 4) */
        nNeighbors[i][0] =  stag->l[0][neighborRank % stag->nRanks[0]];
        nNeighbors[i][1] =  stag->l[1][neighborRank / stag->nRanks[0]];
      } /* else leave uninitialized - error if accessed */
    }

    /* These offsets should always be non-negative, and describe how many
       ghost elements exist at each boundary. These are not always equal to the stencil width,
       because we may have different numbers of ghost elements at the boundaries. In particular,
       we always have at least one ghost (dummy) element at the right/top/front. */
    for (d=0; d<2; ++d) ghostOffsetStart[d] = stag->start[d] - stag->startGhost[d];
    for (d=0; d<2; ++d) ghostOffsetEnd[d]   = stag->startGhost[d]+stag->nGhost[d] - (stag->start[d]+stag->n[d]);

    /* Compute whether the next rank has an extra point (only used in x direction) */
    for (d=0;d<2;++d) nextToDummyEnd[d] = (PetscBool)(stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->rank[d] == stag->nRanks[d]-2);

    /* Compute the number of local entries which correspond to any global entry */
    {
      PetscInt nNonDummyGhost[2];
      for (d=0; d<2; ++d) nNonDummyGhost[d] = stag->nGhost[d] - (dummyStart[d] ? ghostOffsetStart[d] : 0) - (dummyEnd[d] ? ghostOffsetEnd[d] : 0);
      if (star) {
        entriesToTransferTotal = (nNonDummyGhost[0] * stag->n[1] + stag->n[0] * nNonDummyGhost[1] - stag->n[0] * stag->n[1]) * stag->entriesPerElement
          + (dummyEnd[0]                ? nNonDummyGhost[1] * entriesPerEdge   : 0)
          + (dummyEnd[1]                ? nNonDummyGhost[0] * entriesPerEdge   : 0)
          + (dummyEnd[0] && dummyEnd[1] ?                     entriesPerCorner : 0);
      } else {
        entriesToTransferTotal = nNonDummyGhost[0] * nNonDummyGhost[1] * stag->entriesPerElement
          + (dummyEnd[0]                ? nNonDummyGhost[1] * entriesPerEdge   : 0)
          + (dummyEnd[1]                ? nNonDummyGhost[0] * entriesPerEdge   : 0)
          + (dummyEnd[0] && dummyEnd[1] ?                     entriesPerCorner : 0);
      }
    }

    /* Allocate arrays to populate */
    ierr = PetscMalloc1(entriesToTransferTotal,&idxLocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(entriesToTransferTotal,&idxGlobal);CHKERRQ(ierr);

    /* Counts into idxLocal/idxGlobal */
    count = 0;

    /* Here and below, we work with (i,j) describing element numbers within a neighboring rank's global ordering,
       to be offset by that rank's global offset,
       and (ighost,jghost) referring to element numbers within this ranks local (ghosted) ordering */

    /* Neighbor 0 (down left) */
    if (!star && !dummyStart[0] && !dummyStart[1]) {
      const PetscInt         neighbor             = 0;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = stag->entriesPerElement * nNeighbor[0];
      for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
        const PetscInt j = nNeighbor[1] - ghostOffsetStart[1] + jghost;
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          const PetscInt i = nNeighbor[0] - ghostOffsetStart[0] + ighost;
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Neighbor 1 (down) */
    if (!dummyStart[1]) {
      /* We may be a ghosted boundary in x, in which case the neighbor is also */
      const PetscInt         neighbor                     = 1;
      const PetscInt         globalOffset                 = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor                    = nNeighbors[neighbor];
      const PetscInt         entriesPerElementRowNeighbor = entriesPerElementRow; /* same as here */
      for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
        const PetscInt j = nNeighbor[1] - ghostOffsetStart[1] + jghost;
        for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
          const PetscInt i = ighost - ghostOffsetStart[0];
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =               jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
        if (dummyEnd[0]) {
          const PetscInt ighost = stag->nGhost[0]-ghostOffsetEnd[0];
          const PetscInt i = stag->n[0];
          for (d=0; d<stag->dof[0]; ++d, ++count) { /* Vertex */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
          for (d=0; d<stag->dof[1]; ++d,++count) { /* Edge */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + stag->dof[0]                + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + stag->dof[0] + stag->dof[1] + d;
          }
        }
      }
    }

    /* Neighbor 2 (down right) */
    if (!star && !dummyEnd[0] && !dummyStart[1]) {
      const PetscInt         neighbor             = 2;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
      for (jghost = 0; jghost<ghostOffsetStart[1]; ++jghost) {
        const PetscInt j = nNeighbor[1] - ghostOffsetStart[1] + jghost;
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          const PetscInt ighost = stag->nGhost[0] - ghostOffsetEnd[0] + i;
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Neighbor 3 (left) */
    if (!dummyStart[0]) {
      /* Our neighbor is never a ghosted boundary in x, but we may be
         Here, we may be a ghosted boundary in y and thus so will our neighbor be */
      const PetscInt         neighbor             = 3;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = nNeighbor[0]*stag->entriesPerElement;
      for (jghost = ghostOffsetStart[1]; jghost<stag->nGhost[1] - ghostOffsetEnd[1]; ++jghost) {
        const PetscInt j = jghost-ghostOffsetStart[1];
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          const PetscInt i = nNeighbor[0] - ghostOffsetStart[0] + ighost;
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
      if (dummyEnd[1]) {
        const PetscInt jghost = stag->nGhost[1]-ghostOffsetEnd[1];
        const PetscInt j = stag->n[1];
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          const PetscInt i = nNeighbor[0] - ghostOffsetStart[0] + ighost;
          for (d=0; d<entriesPerEdge; ++d, ++count) { /* only vertices and horizontal edge (which are the first dof) */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *entriesPerEdge          + d; /* i moves by edge here */
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Interior/Resident-here-in-global elements ("Neighbor 4" - same rank)
       *including* entries from boundary dummy elements */
    {
      const PetscInt neighbor     = 4;
      const PetscInt globalOffset = globalOffsets[stag->neighbors[neighbor]];
      for (j=0; j<stag->n[1]; ++j) {
        const PetscInt jghost = j + ghostOffsetStart[1];
        for (i=0; i<stag->n[0]; ++i) {
          const PetscInt ighost = i + ghostOffsetStart[0];
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + d;
          }
        }
        if (dummyEnd[0]) {
          const PetscInt ighost = i + ghostOffsetStart[0];
          i = stag->n[0];
          for (d=0; d<stag->dof[0]; ++d, ++count) { /* vertex first */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRow      + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost + ighost*stag->entriesPerElement + d;
          }
          for (d=0; d<stag->dof[1]; ++d, ++count) { /* then left ege (skipping bottom edge) */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *stag->entriesPerElement + stag->dof[0]                + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + stag->dof[0] + stag->dof[1] + d;
          }
        }
      }
      if (dummyEnd[1]) {
        const PetscInt jghost = j + ghostOffsetStart[1];
        j = stag->n[1];
        for (i=0; i<stag->n[0]; ++i) {
          const PetscInt ighost = i + ghostOffsetStart[0];
          for (d=0; d<entriesPerEdge; ++d, ++count) { /* vertex and bottom edge (which are the first entries) */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRow +      i*entriesPerEdge               + d; /* note i increment by entriesPerEdge */
            idxLocal[count]  =                jghost*entriesPerElementRowGhost + ighost*stag->entriesPerElement + d;
          }
        }
        if (dummyEnd[0]) {
          const PetscInt ighost = i + ghostOffsetStart[0];
          i = stag->n[0];
          for (d=0; d<entriesPerCorner; ++d, ++count) { /* vertex only */
            idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *entriesPerEdge          + d; /* note i increment by entriesPerEdge */
            idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Neighbor 5 (right) */
    if (!dummyEnd[0]) {
      /* We can never be right boundary, but we may be a top boundary, along with the right neighbor */
      const PetscInt         neighbor             = 5;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
      for (jghost = ghostOffsetStart[1];jghost<stag->nGhost[1]-ghostOffsetEnd[1]; ++jghost) {
        const PetscInt j = jghost-ghostOffsetStart[1];
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          const PetscInt ighost = stag->nGhost[0] - ghostOffsetEnd[0] + i;
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =               jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
      if (dummyEnd[1]) {
        const PetscInt jghost = stag->nGhost[1]-ghostOffsetEnd[1];
        const PetscInt j = nNeighbor[1];
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          const PetscInt ighost = stag->nGhost[0] - ghostOffsetEnd[0] + i;
          for (d=0; d<entriesPerEdge; ++d, ++count) { /* only vertices and horizontal edge (which are the first dof) */
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor    + i     *entriesPerEdge    + d; /* Note i increment by entriesPerEdge */
            idxLocal[count]  =               jghost*entriesPerElementRowGhost + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Neighbor 6 (up left) */
    if (!star && !dummyStart[0] && !dummyEnd[1]) {
      /* We can never be a top boundary, but our neighbor may be
       We may be a right boundary, but our neighbor cannot be */
      const PetscInt         neighbor             = 6;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = nNeighbor[0]*stag->entriesPerElement;
      for (j=0; j<ghostOffsetEnd[1]; ++j) {
        const PetscInt jghost = stag->nGhost[1] - ghostOffsetEnd[1] + j;
        for (ighost = 0; ighost<ghostOffsetStart[0]; ++ighost) {
          const PetscInt i = nNeighbor[0] - ghostOffsetStart[0] + ighost;
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =               jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    /* Neighbor 7 (up) */
    if (!dummyEnd[1]) {
      /* We cannot be the last rank in y, though our neighbor may be
       We may be the last rank in x, in which case our neighbor is also */
      const PetscInt         neighbor             = 7;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = entriesPerElementRow; /* same as here */
      for (j=0; j<ghostOffsetEnd[1]; ++j) {
        const PetscInt jghost = stag->nGhost[1] - ghostOffsetEnd[1] + j;
        for (ighost = ghostOffsetStart[0]; ighost<stag->nGhost[0]-ghostOffsetEnd[0]; ++ighost) {
          const PetscInt i = ighost - ghostOffsetStart[0];
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
        if (dummyEnd[0]) {
          const PetscInt ighost = stag->nGhost[0]-ghostOffsetEnd[0];
          const PetscInt i = nNeighbor[0];
          for (d=0; d<stag->dof[0]; ++d, ++count) { /* Vertex */
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =               jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
          for (d=0; d<stag->dof[1]; ++d, ++count) { /* Edge */
            idxGlobal[count] = globalOffset+ j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + stag->dof[0]                + d;
            idxLocal[count]  =               jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + stag->dof[0] + stag->dof[1] + d;
          }
        }
      }
    }

    /* Neighbor 8 (up right) */
    if (!star && !dummyEnd[0] && !dummyEnd[1]) {
      /* We can never be a ghosted boundary
         Our neighbor may be a top boundary, a right boundary, or both */
      const PetscInt         neighbor             = 8;
      const PetscInt         globalOffset         = globalOffsets[stag->neighbors[neighbor]];
      const PetscInt * const nNeighbor            = nNeighbors[neighbor];
      const PetscInt entriesPerElementRowNeighbor = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
      for (j=0; j<ghostOffsetEnd[1]; ++j) {
        const PetscInt jghost = stag->nGhost[1] - ghostOffsetEnd[1] + j;
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          const PetscInt ighost = stag->nGhost[0] - ghostOffsetEnd[0] + i;
          for (d=0; d<stag->entriesPerElement; ++d, ++count) {
            idxGlobal[count] = globalOffset + j     *entriesPerElementRowNeighbor + i     *stag->entriesPerElement + d;
            idxLocal[count]  =                jghost*entriesPerElementRowGhost    + ighost*stag->entriesPerElement + d;
          }
        }
      }
    }

    if (count != entriesToTransferTotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of entries computed in gtol (%d) is not as expected (%d)",count,entriesToTransferTotal);

    /* Create Local and Global ISs (transferring pointer ownership) */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);

    /* Create stag->gtol. The order is computed as PETSc ordering, and doesn't include dummy entries */
    {
      Vec local,global;
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
      ierr = VecScatterCreate(global,isGlobal,local,isLocal,&stag->gtol);CHKERRQ(ierr);
      ierr = VecDestroy(&global);CHKERRQ(ierr);
      ierr = VecDestroy(&local);CHKERRQ(ierr);
    }

    /* Destroy ISs */
    ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
    ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);

    /* Next, we iterate over the local entries  again, in local order, recording the global entry to which each maps,
       or -1 if there is none */
    ierr = PetscMalloc1(stag->entriesGhost,&idxGlobalAll);CHKERRQ(ierr);

    countAll = 0;

    /* Loop over rows 1/3 : down */
    if (!dummyStart[1]) {
      for (jghost=0; jghost<ghostOffsetStart[1]; ++jghost) {

        /* Loop over columns 1/3 : down left */
        if (!star && !dummyStart[0]) {
          const PetscInt         neighbor     = 0;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         j            = nNeighbor[1] - ghostOffsetStart[1] + jghost; /* Note: this is actually the same value for the whole row of ranks below, so recomputing it for the next two ranks is redundant, and one could even get rid of jghost entirely if desired */
          const PetscInt         eprNeighbor  = nNeighbor[0] * stag->entriesPerElement;
          for (i=nNeighbor[0]-ghostOffsetStart[0]; i<nNeighbor[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        } else {
          /* Down Left dummies */
          for (ighost=0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        }

        /* Loop over columns 2/3 : down middle */
        {
          const PetscInt         neighbor     = 1;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         j            = nNeighbor[1] - ghostOffsetStart[1] + jghost;
          const PetscInt         eprNeighbor  = entriesPerElementRow; /* same as here */
          for (i=0; i<nNeighbor[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        }

        /* Loop over columns 3/3 : down right */
        if (!star && !dummyEnd[0]) {
          const PetscInt         neighbor     = 2;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         j            = nNeighbor[1] - ghostOffsetStart[1] + jghost;
          const PetscInt         eprNeighbor  = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        } else if (dummyEnd[0]) {
          /* Down right partial dummy elements, living on the *down* rank */
          const PetscInt         neighbor     = 1;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         j            = nNeighbor[1] - ghostOffsetStart[1] + jghost;
          const PetscInt         eprNeighbor  = entriesPerElementRow; /* same as here */
          PetscInt dGlobal;
          i = nNeighbor[0];
          for (d=0,dGlobal=0; d<stag->dof[0]; ++d, ++dGlobal, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
          }
          for (; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy down edge point */
          }
          for (; d<stag->dof[0] + 2*stag->dof[1]; ++d, ++dGlobal, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
          }
          for (; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy element point */
          }
          ++i;
          for (; i<nNeighbor[0]+ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        } else {
          /* Down Right dummies */
          for (ighost=0; ighost<ghostOffsetEnd[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        }
      }
    } else {
      /* Down dummies row */
      for (jghost=0; jghost<ghostOffsetStart[1]; ++jghost) {
        for (ighost=0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
    }

    /* Loop over rows 2/3 : center */
    for (j=0; j<stag->n[1]; ++j) {
      /* Loop over columns 1/3 : left */
      if (!dummyStart[0]) {
        const PetscInt         neighbor     = 3;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt * const nNeighbor    = nNeighbors[neighbor];
        const PetscInt         eprNeighbor  = nNeighbor[0] * stag->entriesPerElement;
        for (i=nNeighbor[0]-ghostOffsetStart[0]; i<nNeighbor[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
          }
        }
      } else {
        /* (Middle) Left dummies */
        for (ighost=0; ighost < ghostOffsetStart[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }

      /* Loop over columns 2/3 : here (the "neighbor" is ourselves, here) */
      {
        const PetscInt neighbor     = 4;
        const PetscInt globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt eprNeighbor  = entriesPerElementRow; /* same as here (obviously) */
        for (i=0; i<stag->n[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
          }
        }
      }

      /* Loop over columns 3/3 : right */
      if (!dummyEnd[0]) {
          const PetscInt         neighbor     = 5;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         eprNeighbor  = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
        for (i=0; i<ghostOffsetEnd[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
          }
        }
      } else {
        /* -1's for right layer of partial dummies, living on *this* rank */
        const PetscInt         neighbor     = 4;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt * const nNeighbor    = nNeighbors[neighbor];
        const PetscInt         eprNeighbor  = entriesPerElementRow; /* same as here (obviously) */
        PetscInt dGlobal;
        i = nNeighbor[0];
        for (d=0,dGlobal=0; d<stag->dof[0]; ++d, ++dGlobal, ++countAll) {
          idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
        }
        for (; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
          idxGlobalAll[countAll] = -1; /* dummy down edge point */
        }
        for (; d<stag->dof[0] + 2*stag->dof[1]; ++d, ++dGlobal, ++countAll) {
          idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
        }
        for (; d<stag->entriesPerElement; ++d, ++countAll) {
          idxGlobalAll[countAll] = -1; /* dummy element point */
        }
        ++i;
        for (; i<nNeighbor[0]+ghostOffsetEnd[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
    }

    /* Loop over rows 3/3 : up */
    if (!dummyEnd[1]) {
      for (j=0; j<ghostOffsetEnd[1]; ++j) {

        /* Loop over columns 1/3 : up left */
        if (!star && !dummyStart[0]) {
          const PetscInt         neighbor     = 6;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         eprNeighbor  = nNeighbor[0] * stag->entriesPerElement;
          for (i=nNeighbor[0]-ghostOffsetStart[0]; i<nNeighbor[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        } else {
          /* Up Left dummies */
          for (ighost=0; ighost<ghostOffsetStart[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        }

        /* Loop over columns 2/3 : up */
        {
          const PetscInt         neighbor     = 7;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         eprNeighbor  = entriesPerElementRow; /* Same as here */
          for (i=0; i<nNeighbor[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        }

        /* Loop over columns 3/3 : up right */
        if (!star && !dummyEnd[0]) {
          const PetscInt         neighbor     = 8;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         eprNeighbor  = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
          for (i=0; i<ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + d;
            }
          }
        } else if (dummyEnd[0]) {
          /* -1's for right layer of partial dummies, living on rank above */
          const PetscInt         neighbor     = 7;
          const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
          const PetscInt * const nNeighbor    = nNeighbors[neighbor];
          const PetscInt         eprNeighbor  = entriesPerElementRow; /* Same as here */
          PetscInt dGlobal;
          i = nNeighbor[0];
          for (d=0,dGlobal=0; d<stag->dof[0]; ++d, ++dGlobal, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
          }
          for (; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy down edge point */
          }
          for (; d<stag->dof[0] + 2*stag->dof[1]; ++d, ++dGlobal, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*stag->entriesPerElement + dGlobal;
          }
          for (; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy element point */
          }
          ++i;
          for (; i<nNeighbor[0]+ghostOffsetEnd[0]; ++i) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        } else {
          /* Up Right dummies */
          for (ighost=0; ighost<ghostOffsetEnd[0]; ++ighost) {
            for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
              idxGlobalAll[countAll] = -1;
            }
          }
        }
      }
    } else {
      j = stag->n[1];
      /* Top layer of partial dummies */

      /* up left partial dummies layer : Loop over columns 1/3 : living on *left* neighbor */
      if (!dummyStart[0]) {
        const PetscInt         neighbor     = 3;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt * const nNeighbor    = nNeighbors[neighbor];
        const PetscInt         eprNeighbor  = nNeighbor[0] * stag->entriesPerElement;
        for (i=nNeighbor[0]-ghostOffsetStart[0]; i<nNeighbor[0]; ++i) {
          for (d=0; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*entriesPerEdge + d; /* Note entriesPerEdge here */
          }
          for (; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy left edge and element points */
          }
        }
      } else {
        for (ighost=0; ighost<ghostOffsetStart[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }

      /* up partial dummies layer : Loop over columns 2/3 : living on *this* rank */
      {
        const PetscInt         neighbor     = 4;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt         eprNeighbor  = entriesPerElementRow; /* same as here (obviously) */
        for (i=0; i<stag->n[0]; ++i) {
          for (d=0; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*entriesPerEdge + d; /* Note entriesPerEdge here */
          }
          for (; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy left edge and element points */
          }
        }
      }

      if (!dummyEnd[0]) {
        /* up right partial dummies layer : Loop over columns 3/3 :  living on *right* neighbor */
        const PetscInt         neighbor     = 5;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt * const nNeighbor    = nNeighbors[neighbor];
        const PetscInt         eprNeighbor  = nNeighbor[0]*stag->entriesPerElement + (nextToDummyEnd[0] ? entriesPerEdge : 0);
        for (i = 0; i<ghostOffsetEnd[0]; ++i) {
          for (d=0; d<stag->dof[0] + stag->dof[1]; ++d, ++countAll) {
            idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*entriesPerEdge + d; /* Note entriesPerEdge here */
          }
          for (; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1; /* dummy left edge and element points */
          }
        }
      } else {
        /* Top partial dummies layer : Loop over columns 3/3 : right, living *here* */
        const PetscInt         neighbor     = 4;
        const PetscInt         globalOffset = globalOffsets[stag->neighbors[neighbor]];
        const PetscInt         eprNeighbor  = entriesPerElementRow; /* same as here (obviously) */
        i = stag->n[0];
        for (d=0; d<stag->dof[0]; ++d, ++countAll) { /* Note just the vertex here */
          idxGlobalAll[countAll] = globalOffset + j*eprNeighbor + i*entriesPerEdge + d; /* Note entriesPerEdge here */
        }
        for (; d<stag->entriesPerElement; ++d, ++countAll) {
          idxGlobalAll[countAll] = -1; /* dummy bottom edge, left edge and element points */
        }
        ++i;
        for (; i<stag->n[0] + ghostOffsetEnd[0]; ++i) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
      ++j;
      /* Additional top dummy layers */
      for (; j<stag->n[1]+ghostOffsetEnd[1]; ++j) {
        for (ighost=0; ighost<stag->nGhost[0]; ++ighost) {
          for (d=0; d<stag->entriesPerElement; ++d, ++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
    }

    /* Create local-to-global map (in local ordering, includes maps to -1 for dummy points) */
    ierr = ISLocalToGlobalMappingCreate(comm,1,stag->entriesGhost,idxGlobalAll,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)dm->ltogmap);CHKERRQ(ierr);
  }

  /* In special cases, create a dedicated injective local-to-global map */
  if ((stag->boundaryType[0] == DM_BOUNDARY_PERIODIC && stag->nRanks[0] == 1) ||
      (stag->boundaryType[1] == DM_BOUNDARY_PERIODIC && stag->nRanks[1] == 1)) {
    ierr = DMStagPopulateLocalToGlobalInjective(dm);CHKERRQ(ierr);
  }

  /* Free global offsets */
  ierr = PetscFree(globalOffsets);CHKERRQ(ierr);

  /* Precompute location offsets */
  ierr = DMStagComputeLocationOffsets_2d(dm);CHKERRQ(ierr);

  /* View from Options */
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* adapted from da2.c */
static PetscErrorCode DMStagSetUpBuildRankGrid_2d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        m,n;
  PetscMPIInt     rank,size;
  const PetscInt  M = stag->N[0];
  const PetscInt  N = stag->N[1];

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  m = stag->nRanks[0];
  n = stag->nRanks[1];
  if (m != PETSC_DECIDE) {
    if (m < 1) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of ranks in X direction: %D",m);
    else if (m > size) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Too many ranks in X direction: %D %d",m,size);
  }
  if (n != PETSC_DECIDE) {
    if (n < 1) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Non-positive number of ranks in Y direction: %D",n);
    else if (n > size) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Too many ranks in Y direction: %D %d",n,size);
  }
  if (m == PETSC_DECIDE || n == PETSC_DECIDE) {
    if (n != PETSC_DECIDE) {
      m = size/n;
    } else if (m != PETSC_DECIDE) {
      n = size/m;
    } else {
      /* try for squarish distribution */
      m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)N)));
      if (!m) m = 1;
      while (m > 0) {
        n = size/m;
        if (m*n == size) break;
        m--;
      }
      if (M > N && m < n) {PetscInt _m = m; m = n; n = _m;}
    }
    if (m*n != size) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Unable to create partition, check the size of the communicator and input m and n ");
  } else if (m*n != size) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Given Bad partition. Product of sizes (%D) does not equal communicator size (%d)",m*n,size);
  if (M < m) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Partition in x direction is too fine! %D %D",M,m);
  if (N < n) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Partition in y direction is too fine! %D %D",N,n);
  stag->nRanks[0] = m;
  stag->nRanks[1] = n;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagSetUpBuildNeighbors_2d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        d,i;
  PetscBool       per[2],first[2],last[2];
  PetscInt        neighborRank[9][2],r[2],n[2];
  const PetscInt  dim = 2;

  PetscFunctionBegin;
  for (d=0; d<dim; ++d) if (stag->boundaryType[d] != DM_BOUNDARY_NONE && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC && stag->boundaryType[d] != DM_BOUNDARY_GHOSTED) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Neighbor determination not implemented for %s",DMBoundaryTypes[stag->boundaryType[d]]);

  /* Assemble some convenience variables */
  for (d=0; d<dim; ++d) {
    per[d]   = (PetscBool)(stag->boundaryType[d] == DM_BOUNDARY_PERIODIC);
    first[d] = stag->firstRank[d];
    last[d]  = stag->lastRank[d];
    r[d]     = stag->rank[d];
    n[d]     = stag->nRanks[d];
  }

  /* First, compute the position in the rank grid for all neighbors */
  neighborRank[0][0]  = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  down */
  neighborRank[0][1]  = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;

  neighborRank[1][0] =                                      r[0]    ; /*       down */
  neighborRank[1][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;

  neighborRank[2][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right down */
  neighborRank[2][1] = first[1] ? (per[1] ?  n[1]-1 : -1) : r[1] - 1;

  neighborRank[3][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left       */
  neighborRank[3][1] =                                      r[1]    ;

  neighborRank[4][0] =                                      r[0]    ; /*            */
  neighborRank[4][1] =                                      r[1]    ;

  neighborRank[5][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right      */
  neighborRank[5][1] =                                      r[1]    ;

  neighborRank[6][0] = first[0] ? (per[0] ?  n[0]-1 : -1) : r[0] - 1; /* left  up   */
  neighborRank[6][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;

  neighborRank[7][0] =                                      r[0]    ; /*       up   */
  neighborRank[7][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;

  neighborRank[8][0] = last[0]  ? (per[0] ?  0      : -1) : r[0] + 1; /* right up   */
  neighborRank[8][1] = last[1]  ? (per[1] ?  0      : -1) : r[1] + 1;

  /* Then, compute the rank of each in the linear ordering */
  ierr = PetscMalloc1(9,&stag->neighbors);CHKERRQ(ierr);
  for (i=0; i<9; ++i) {
    if  (neighborRank[i][0] >= 0 && neighborRank[i][1] >=0) {
      stag->neighbors[i] = neighborRank[i][0] + n[0]*neighborRank[i][1];
    } else {
      stag->neighbors[i] = -1;
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagSetUpBuildGlobalOffsets_2d(DM dm,PetscInt **pGlobalOffsets)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              *globalOffsets;
  PetscInt              i,j,d,entriesPerEdge,count;
  PetscMPIInt           size;
  PetscBool             extra[2];

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  for (d=0; d<2; ++d) extra[d] = (PetscBool)(stag->boundaryType[d] != DM_BOUNDARY_PERIODIC); /* Extra points in global rep */
  entriesPerEdge = stag->dof[0] + stag->dof[1];
  ierr = PetscMalloc1(size,pGlobalOffsets);CHKERRQ(ierr);
  globalOffsets = *pGlobalOffsets;
  globalOffsets[0] = 0;
  count = 1; /* note the count is offset by 1 here. We add the size of the previous rank */
  for (j=0; j<stag->nRanks[1]-1; ++j) {
    const PetscInt nnj = stag->l[1][j];
    for (i=0; i<stag->nRanks[0]-1; ++i) {
      const PetscInt nni = stag->l[0][i];
      globalOffsets[count] = globalOffsets[count-1] + nnj*nni*stag->entriesPerElement; /* No right/top/front boundaries */
      ++count;
    }
    {
      /* i = stag->nRanks[0]-1; */
      const PetscInt nni = stag->l[0][i];
      globalOffsets[count] = globalOffsets[count-1] + nnj*nni*stag->entriesPerElement
                             + (extra[0] ? nnj*entriesPerEdge : 0); /* Extra edges on the right */
      ++count;
    }
  }
  {
    /* j = stag->nRanks[1]-1; */
    const PetscInt nnj = stag->l[1][j];
    for (i=0; i<stag->nRanks[0]-1; ++i) {
      const PetscInt nni = stag->l[0][i];
      globalOffsets[count] = globalOffsets[count-1] + nni*nnj*stag->entriesPerElement
                             + (extra[1] ? nni*entriesPerEdge : 0); /* Extra edges on the top */
      ++count;
    }
    /* Don't need to compute entries in last element */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagComputeLocationOffsets_2d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt  epe = stag->entriesPerElement;
  const PetscInt  epr = stag->nGhost[0]*epe;

  PetscFunctionBegin;
  ierr = PetscMalloc1(DMSTAG_NUMBER_LOCATIONS,&stag->locationOffsets);CHKERRQ(ierr);
  stag->locationOffsets[DMSTAG_DOWN_LEFT]  = 0;
  stag->locationOffsets[DMSTAG_DOWN]       = stag->locationOffsets[DMSTAG_DOWN_LEFT]  + stag->dof[0];
  stag->locationOffsets[DMSTAG_DOWN_RIGHT] = stag->locationOffsets[DMSTAG_DOWN_LEFT]  + epe;
  stag->locationOffsets[DMSTAG_LEFT]       = stag->locationOffsets[DMSTAG_DOWN]       + stag->dof[1];
  stag->locationOffsets[DMSTAG_ELEMENT]    = stag->locationOffsets[DMSTAG_LEFT]       + stag->dof[1];
  stag->locationOffsets[DMSTAG_RIGHT]      = stag->locationOffsets[DMSTAG_LEFT]       + epe;
  stag->locationOffsets[DMSTAG_UP_LEFT]    = stag->locationOffsets[DMSTAG_DOWN_LEFT]  + epr;
  stag->locationOffsets[DMSTAG_UP]         = stag->locationOffsets[DMSTAG_DOWN]       + epr;
  stag->locationOffsets[DMSTAG_UP_RIGHT]   = stag->locationOffsets[DMSTAG_UP_LEFT]    + epe;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_2d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        *idxLocal,*idxGlobal,*globalOffsetsRecomputed;
  const PetscInt  *globalOffsets;
  PetscInt        i,j,d,count,entriesPerCorner,entriesPerEdge,entriesPerElementRowGhost,entriesPerElementRow,ghostOffsetStart[2];
  IS              isLocal,isGlobal;
  PetscBool       dummyEnd[2];

  PetscFunctionBegin;
  ierr = DMStagSetUpBuildGlobalOffsets_2d(dm,&globalOffsetsRecomputed);CHKERRQ(ierr); /* note that we don't actually use all of these. An available optimization is to pass them, when available */
  globalOffsets = globalOffsetsRecomputed;
  ierr = PetscMalloc1(stag->entries,&idxLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(stag->entries,&idxGlobal);CHKERRQ(ierr);
  for (d=0; d<2; ++d) dummyEnd[d]   = (PetscBool)(stag->lastRank[d] && stag->boundaryType[d] != DM_BOUNDARY_PERIODIC);
  entriesPerCorner          = stag->dof[0];
  entriesPerEdge            = stag->dof[0] + stag->dof[1];
  entriesPerElementRow      = stag->n[0]     *stag->entriesPerElement + (dummyEnd[0] ? entriesPerEdge : 0);
  entriesPerElementRowGhost = stag->nGhost[0]*stag->entriesPerElement;
  count = 0;
  for (d=0; d<2; ++d) ghostOffsetStart[d] = stag->start[d] - stag->startGhost[d];
  {
    const PetscInt neighbor     = 4;
    const PetscInt globalOffset = globalOffsets[stag->neighbors[neighbor]];
    for (j=0; j<stag->n[1]; ++j) {
      const PetscInt jghost = j + ghostOffsetStart[1];
      for (i=0; i<stag->n[0]; ++i) {
        const PetscInt ighost = i + ghostOffsetStart[0];
        for (d=0; d<stag->entriesPerElement; ++d, ++count) {
          idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *stag->entriesPerElement + d;
          idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + d;
        }
      }
      if (dummyEnd[0]) {
        const PetscInt ighost = i + ghostOffsetStart[0];
        i = stag->n[0];
        for (d=0; d<stag->dof[0]; ++d, ++count) { /* vertex first */
          idxGlobal[count] = globalOffset + j     *entriesPerElementRow      + i     *stag->entriesPerElement + d;
          idxLocal[count]  =                jghost*entriesPerElementRowGhost + ighost*stag->entriesPerElement + d;
        }
        for (d=0; d<stag->dof[1]; ++d, ++count) { /* then left ege (skipping bottom edge) */
          idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *stag->entriesPerElement + stag->dof[0]                + d;
          idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + stag->dof[0] + stag->dof[1] + d;
        }
      }
    }
    if (dummyEnd[1]) {
      const PetscInt jghost = j + ghostOffsetStart[1];
      j = stag->n[1];
      for (i=0; i<stag->n[0]; ++i) {
        const PetscInt ighost = i + ghostOffsetStart[0];
        for (d=0; d<entriesPerEdge; ++d, ++count) { /* vertex and bottom edge (which are the first entries) */
          idxGlobal[count] = globalOffset + j     *entriesPerElementRow +      i*entriesPerEdge               + d; /* note i increment by entriesPerEdge */
          idxLocal[count]  =                jghost*entriesPerElementRowGhost + ighost*stag->entriesPerElement + d;
        }
      }
      if (dummyEnd[0]) {
        const PetscInt ighost = i + ghostOffsetStart[0];
        i = stag->n[0];
        for (d=0; d<entriesPerCorner; ++d, ++count) { /* vertex only */
          idxGlobal[count] = globalOffset + j     *entriesPerElementRow       + i     *entriesPerEdge          + d; /* note i increment by entriesPerEdge */
          idxLocal[count]  =                jghost*entriesPerElementRowGhost  + ighost*stag->entriesPerElement + d;
        }
      }
    }
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
