/*
   Functions specific to the 1-dimensional implementation of DMStag
*/
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagCreate1d - Create an object to manage data living on the faces, edges, and vertices of a parallelized regular 1D grid.

  Collective

  Input Parameters:
+ comm - MPI communicator
. bndx - boundary type: DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, or DM_BOUNDARY_GHOSTED
. M - global number of grid points
. dof0 - number of degrees of freedom per vertex/point/node/0-cell
. dof1 - number of degrees of freedom per element/edge/1-cell
. stencilType - ghost/halo region type: DMSTAG_STENCIL_BOX or DMSTAG_STENCIL_NONE
. stencilWidth - width, in elements, of halo/ghost region
- lx - array of local sizes, of length equal to the comm size, summing to M

  Output Parameter:
. dm - the new DMStag object

  Options Database Keys:
+ -dm_view - calls DMViewFromOptions() a the conclusion of DMSetUp()
. -stag_grid_x <nx> - number of elements in the x direction
. -stag_ghost_stencil_width - width of ghost region, in elements
- -stag_boundary_type_x <none,ghosted,periodic> - DMBoundaryType value

  Notes:
  You must call DMSetUp() after this call before using the DM.
  If you wish to use the options database (see the keys above) to change values in the DMStag, you must call
  DMSetFromOptions() after this function but before DMSetUp().

  Level: beginner

.seealso: DMSTAG, DMStagCreate2d(), DMStagCreate3d(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateLocalVector(), DMLocalToGlobalBegin(), DMDACreate1d()
@*/
PETSC_EXTERN PetscErrorCode DMStagCreate1d(MPI_Comm comm,DMBoundaryType bndx,PetscInt M,PetscInt dof0,PetscInt dof1,DMStagStencilType stencilType,PetscInt stencilWidth,const PetscInt lx[],DM* dm)
{
  PetscErrorCode ierr;
  DM_Stag        *stag;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm,1);CHKERRQ(ierr); /* Must precede DMSetType */
  ierr = DMSetType(*dm,DMSTAG);CHKERRQ(ierr);
  stag = (DM_Stag*)(*dm)->data;

  /* Global sizes and flags (derived quantities set in DMSetUp_Stag) */
  stag->boundaryType[0] = bndx;
  stag->N[0]            = M;
  stag->nRanks[0]       = size;
  stag->stencilType     = stencilType;
  stag->stencilWidth    = stencilWidth;
  ierr = DMStagSetDOF(*dm,dof0,dof1,0,0);CHKERRQ(ierr);
  ierr = DMStagSetOwnershipRanges(*dm,lx,NULL,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_1d(DM dm,PetscReal xmin,PetscReal xmax)
{
  PetscErrorCode ierr;
  DM_Stag        *stagCoord;
  DM             dmCoord;
  Vec            coordLocal,coord;
  PetscReal      h,min;
  PetscScalar    **arr;
  PetscInt       ind,start,n,nExtra,s;
  PetscInt       ileft,ielement;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  stagCoord = (DM_Stag*) dmCoord->data;
  for (s=0; s<2; ++s) {
    if (stagCoord->dof[s] !=0 && stagCoord->dof[s] != 1) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Coordinate DM in 1 dimensions must have 0 or 1 dof on each stratum, but stratum %d has %d dof",s,stagCoord->dof[s]);
  }
  ierr = DMGetLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  if (stagCoord->dof[0]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_LEFT,0,&ileft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[1]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_ELEMENT,0,&ielement);CHKERRQ(ierr);
  }
  ierr = DMStagGetCorners(dmCoord,&start,NULL,NULL,&n,NULL,NULL,&nExtra,NULL,NULL);CHKERRQ(ierr);

  min = xmin;
  h = (xmax-xmin)/stagCoord->N[0];

  for(ind=start; ind<start + n + nExtra; ++ind) {
    if (stagCoord->dof[0]) {
      const PetscReal off = 0.0;
        arr[ind][ileft] = min + ((PetscReal)ind + off) * h;
    }
    if (stagCoord->dof[1]) {
      const PetscReal off = 0.5;
        arr[ind][ielement] = min + ((PetscReal)ind + off) * h;
    }
  }
  ierr = DMStagVecRestoreArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmCoord,&coord);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmCoord,coordLocal,INSERT_VALUES,coord);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmCoord,coordLocal,INSERT_VALUES,coord);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm,coord);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)coord);CHKERRQ(ierr);
  ierr = VecDestroy(&coord);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Helper functions used in DMSetUp_Stag() */
static PetscErrorCode DMStagComputeLocationOffsets_1d(DM);

PETSC_INTERN PetscErrorCode DMSetUp_Stag_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscMPIInt     size,rank;
  MPI_Comm        comm;
  PetscInt        j;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Check Global size */
  if (stag->N[0] < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Global grid size of %D < 1 specified",stag->N[0]);

  /* Local sizes */
  if (stag->N[0] < size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"More ranks (%d) than elements (%D) specified",size,stag->N[0]);
  if (!stag->l[0]) {
    /* Divide equally, giving an extra elements to higher ranks */
    ierr = PetscMalloc1(stag->nRanks[0],&stag->l[0]);CHKERRQ(ierr);
    for (j=0; j<stag->nRanks[0]; ++j) stag->l[0][j] = stag->N[0]/stag->nRanks[0] + (stag->N[0] % stag->nRanks[0] > j ? 1 : 0);
  }
  {
    PetscInt Nchk = 0;
    for (j=0; j<size; ++j) Nchk += stag->l[0][j];
    if (Nchk != stag->N[0]) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Sum of specified local sizes (%D) is not equal to global size (%D)",Nchk,stag->N[0]);
  }
  stag->n[0] = stag->l[0][rank];

  /* Rank (trivial in 1d) */
  stag->rank[0]      = rank;
  stag->firstRank[0] = (PetscBool)(rank == 0);
  stag->lastRank[0]  = (PetscBool)(rank == size-1);

  /* Local (unghosted) numbers of entries */
  stag->entriesPerElement = stag->dof[0] + stag->dof[1];
  switch (stag->boundaryType[0]) {
    case DM_BOUNDARY_NONE:
    case DM_BOUNDARY_GHOSTED:  stag->entries = stag->n[0] * stag->entriesPerElement + (stag->lastRank[0] ?  stag->dof[0] : 0); break;
    case DM_BOUNDARY_PERIODIC: stag->entries = stag->n[0] * stag->entriesPerElement;                                           break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Starting element */
  stag->start[0] = 0;
  for(j=0; j<stag->rank[0]; ++j) stag->start[0] += stag->l[0][j];

  /* Local/ghosted size and starting element */
  switch (stag->boundaryType[0]) {
    case DM_BOUNDARY_NONE :
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE : /* Only dummy cells on the right */
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0] + (stag->lastRank[0] ? 1 : 0);
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->firstRank[0] ? stag->start[0]: stag->start[0] - stag->stencilWidth;
          stag->nGhost[0] = stag->n[0];
          stag->nGhost[0] += stag->firstRank[0] ? 0 : stag->stencilWidth;
          stag->nGhost[0] += stag->lastRank[0]  ? 1 : stag->stencilWidth;
          break;
        default :
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    case DM_BOUNDARY_GHOSTED:
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE :
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0] + (stag->lastRank[0] ? 1 : 0);
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->start[0] - stag->stencilWidth; /* This value may be negative */
          stag->nGhost[0]     = stag->n[0] + 2*stag->stencilWidth + (stag->lastRank[0] && stag->stencilWidth == 0 ? 1 : 0);
          break;
        default :
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    case DM_BOUNDARY_PERIODIC:
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE :
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0];
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->start[0] - stag->stencilWidth; /* This value may be negative */
          stag->nGhost[0]     = stag->n[0] + 2*stag->stencilWidth;
          break;
        default :
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    default :
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Total size of ghosted/local represention */
  stag->entriesGhost = stag->nGhost[0]*stag->entriesPerElement;

  /* Define neighbors */
  ierr = PetscMalloc1(3,&stag->neighbors);CHKERRQ(ierr);
  if (stag->firstRank[0]) {
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:     stag->neighbors[0] = -1;                break;
      case DM_BOUNDARY_PERIODIC: stag->neighbors[0] = stag->nRanks[0]-1; break;
      default : SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }
  } else {
    stag->neighbors[0] = stag->rank[0]-1;
  }
  stag->neighbors[1] = stag->rank[0];
  if (stag->lastRank[0]) {
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:     stag->neighbors[2] = -1;                break;
      case DM_BOUNDARY_PERIODIC: stag->neighbors[2] = 0;                 break;
      default : SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }
  } else {
    stag->neighbors[2] = stag->rank[0]+1;
  }

  if (stag->n[0] < stag->stencilWidth) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStag 1d setup does not support local sizes (%d) smaller than the elementwise stencil width (%d)",stag->n[0],stag->stencilWidth);
  }

  /* Create global->local VecScatter and ISLocalToGlobalMapping */
  {
    PetscInt *idxLocal,*idxGlobal,*idxGlobalAll;
    PetscInt i,iLocal,d,entriesToTransferTotal,ghostOffsetStart,ghostOffsetEnd,nNonDummyGhost;
    IS       isLocal,isGlobal;

    /* The offset on the right (may not be equal to the stencil width, as we
       always have at least one ghost element, to account for the boundary
       point, and may with ghosted boundaries), and the number of non-dummy ghost elements */
    ghostOffsetStart = stag->start[0] - stag->startGhost[0];
    ghostOffsetEnd   = stag->startGhost[0]+stag->nGhost[0] - (stag->start[0]+stag->n[0]);
    nNonDummyGhost   = stag->nGhost[0] - (stag->lastRank[0] ? ghostOffsetEnd : 0) - (stag->firstRank[0] ? ghostOffsetStart : 0);

    /* Compute the number of non-dummy entries in the local representation
       This is equal to the number of non-dummy elements in the local (ghosted) representation,
       plus some extra entries on the right boundary on the last rank*/
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:
        entriesToTransferTotal = nNonDummyGhost * stag->entriesPerElement + (stag->lastRank[0] ? stag->dof[0] : 0);
        break;
      case DM_BOUNDARY_PERIODIC:
        entriesToTransferTotal = stag->entriesGhost; /* No dummy points */
        break;
      default :
        SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }

    ierr = PetscMalloc1(entriesToTransferTotal,&idxLocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(entriesToTransferTotal,&idxGlobal);CHKERRQ(ierr);
    ierr = PetscMalloc1(stag->entriesGhost,&idxGlobalAll);CHKERRQ(ierr);
    if (stag->boundaryType[0] == DM_BOUNDARY_NONE) {
      PetscInt count = 0,countAll = 0;
      /* Left ghost points and native points */
      for (i=stag->startGhost[0], iLocal=0; iLocal<nNonDummyGhost; ++i,++iLocal) {
        for (d=0; d<stag->entriesPerElement; ++d,++count,++countAll) {
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
      }
      /* Ghost points on the right
         Special case for last (partial dummy) element on the last rank */
      if (stag->lastRank[0] ) {
        i      = stag->N[0];
        iLocal = (stag->nGhost[0]-ghostOffsetEnd);
        /* Only vertex (0-cell) dofs in global representation */
        for (d=0; d<stag->dof[0]; ++d,++count,++countAll) {
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
        for (d=stag->dof[0]; d<stag->entriesPerElement; ++d,++countAll) { /* Additional dummy entries */
          idxGlobalAll[countAll] = -1;
        }
      }
    } else if (stag->boundaryType[0] == DM_BOUNDARY_PERIODIC) {
      PetscInt count = 0,iLocal = 0; /* No dummy points, so idxGlobal and idxGlobalAll are identical */
      const PetscInt iMin = stag->firstRank[0] ? stag->start[0] : stag->startGhost[0];
      const PetscInt iMax = stag->lastRank[0] ? stag->startGhost[0] + stag->nGhost[0] - stag->stencilWidth : stag->startGhost[0] + stag->nGhost[0];
      /* Ghost points on the left */
      if (stag->firstRank[0]) {
        for (i=stag->N[0]-stag->stencilWidth; iLocal<stag->stencilWidth; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobal[count] = i      * stag->entriesPerElement + d;
            idxLocal [count] = iLocal * stag->entriesPerElement + d;
            idxGlobalAll[count] = idxGlobal[count];
          }
        }
      }
      /* Native points */
      for (i=iMin; i<iMax; ++i,++iLocal) {
        for (d=0; d<stag->entriesPerElement; ++d,++count) {
          idxGlobal[count] = i      * stag->entriesPerElement + d;
          idxLocal [count] = iLocal * stag->entriesPerElement + d;
          idxGlobalAll[count] = idxGlobal[count];
        }
      }
      /* Ghost points on the right */
      if (stag->lastRank[0]) {
        for (i=0; iLocal<stag->nGhost[0]; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobal[count] = i      * stag->entriesPerElement + d;
            idxLocal [count] = iLocal * stag->entriesPerElement + d;
            idxGlobalAll[count] = idxGlobal[count];
          }
        }
      }
    } else if (stag->boundaryType[0] == DM_BOUNDARY_GHOSTED) {
      PetscInt count = 0,countAll = 0;
      /* Dummy elements on the left, on the first rank */
      if (stag->firstRank[0]) {
        for(iLocal=0; iLocal<ghostOffsetStart; ++iLocal) {
          /* Complete elements full of dummy entries */
          for (d=0; d<stag->entriesPerElement; ++d,++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
        i = 0; /* nonDummy entries start with global entry 0 */
      } else {
        /* nonDummy entries start as usual */
        i = stag->startGhost[0];
        iLocal = 0;
      }

      /* non-Dummy entries */
      {
        PetscInt iLocalNonDummyMax = stag->firstRank[0] ? nNonDummyGhost + ghostOffsetStart : nNonDummyGhost;
        for (; iLocal<iLocalNonDummyMax; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count,++countAll) {
            idxLocal [count]       = iLocal * stag->entriesPerElement + d;
            idxGlobal[count]       = i      * stag->entriesPerElement + d;
            idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
          }
        }
      }

      /* (partial) dummy elements on the right, on the last rank */
      if (stag->lastRank[0]) {
        /* First one is partial dummy */
        i      = stag->N[0];
        iLocal = (stag->nGhost[0]-ghostOffsetEnd);
        for (d=0; d<stag->dof[0]; ++d,++count,++countAll) { /* Only vertex (0-cell) dofs in global representation */
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
        for (d=stag->dof[0]; d<stag->entriesPerElement; ++d,++countAll) { /* Additional dummy entries */
          idxGlobalAll[countAll] = -1;
        }
        for (iLocal = stag->nGhost[0] - ghostOffsetEnd + 1; iLocal < stag->nGhost[0]; ++iLocal) {
          /* Additional dummy elements */
          for (d=0; d<stag->entriesPerElement; ++d,++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
    } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);

    /* Create Local IS (transferring pointer ownership) */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);

    /* Create Global IS (transferring pointer ownership) */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);

    /* Create stag->gtol, which doesn't include dummy entries */
    {
      Vec local,global;
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
      ierr = VecScatterCreate(global,isGlobal,local,isLocal,&stag->gtol);CHKERRQ(ierr);
      ierr = VecDestroy(&global);CHKERRQ(ierr);
      ierr = VecDestroy(&local);CHKERRQ(ierr);
    }

    /* In special cases, create a dedicated injective local-to-global map */
    if (stag->boundaryType[0] == DM_BOUNDARY_PERIODIC && stag->nRanks[0] == 1) {
      ierr = DMStagPopulateLocalToGlobalInjective(dm);CHKERRQ(ierr);
    }

    /* Destroy ISs */
    ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
    ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);

    /* Create local-to-global map (transferring pointer ownership) */
    ierr = ISLocalToGlobalMappingCreate(comm,1,stag->entriesGhost,idxGlobalAll,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)dm->ltogmap);CHKERRQ(ierr);
  }

  /* Precompute location offsets */
  ierr = DMStagComputeLocationOffsets_1d(dm);CHKERRQ(ierr);

  /* View from Options */
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

static PetscErrorCode DMStagComputeLocationOffsets_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt  epe = stag->entriesPerElement;

  PetscFunctionBegin;
  ierr = PetscMalloc1(DMSTAG_NUMBER_LOCATIONS,&stag->locationOffsets);CHKERRQ(ierr);
  stag->locationOffsets[DMSTAG_LEFT]    = 0;
  stag->locationOffsets[DMSTAG_ELEMENT] = stag->locationOffsets[DMSTAG_LEFT] + stag->dof[0];
  stag->locationOffsets[DMSTAG_RIGHT]   = stag->locationOffsets[DMSTAG_LEFT] + epe;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        *idxLocal,*idxGlobal;
  PetscInt        i,iLocal,d,count;
  IS              isLocal,isGlobal;

  PetscFunctionBegin;
  ierr = PetscMalloc1(stag->entries,&idxLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(stag->entries,&idxGlobal);CHKERRQ(ierr);
  count = 0;
  iLocal = stag->start[0]-stag->startGhost[0];
  for (i=stag->start[0]; i<stag->start[0]+stag->n[0]; ++i,++iLocal) {
    for (d=0; d<stag->entriesPerElement; ++d,++count) {
      idxGlobal[count] = i      * stag->entriesPerElement + d;
      idxLocal [count] = iLocal * stag->entriesPerElement + d;
    }
  }
  if (stag->lastRank[0] && stag->boundaryType[0] != DM_BOUNDARY_PERIODIC) {
    i = stag->start[0]+stag->n[0];
    iLocal = stag->start[0]-stag->startGhost[0] + stag->n[0];
    for (d=0; d<stag->dof[0]; ++d,++count) {
      idxGlobal[count] = i      * stag->entriesPerElement + d;
      idxLocal [count] = iLocal * stag->entriesPerElement + d;
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
  PetscFunctionReturn(0);
}
