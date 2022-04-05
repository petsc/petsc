/*
   Functions specific to the 1-dimensional implementation of DMStag
*/
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagCreate1d - Create an object to manage data living on the elements and vertices of a parallelized regular 1D grid.

  Collective

  Input Parameters:
+ comm - MPI communicator
. bndx - boundary type: DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, or DM_BOUNDARY_GHOSTED
. M - global number of elements
. dof0 - number of degrees of freedom per vertex/0-cell
. dof1 - number of degrees of freedom per element/1-cell
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
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(DMCreate(comm,dm));
  PetscCall(DMSetDimension(*dm,1));
  PetscCall(DMStagInitialize(bndx,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,M,0,0,size,0,0,dof0,dof1,0,0,stencilType,stencilWidth,lx,NULL,NULL,*dm));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_1d(DM dm,PetscReal xmin,PetscReal xmax)
{
  DM_Stag        *stagCoord;
  DM             dmCoord;
  Vec            coordLocal;
  PetscReal      h,min;
  PetscScalar    **arr;
  PetscInt       start_ghost,n_ghost,s;
  PetscInt       ileft,ielement;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &dmCoord));
  stagCoord = (DM_Stag*) dmCoord->data;
  for (s=0; s<2; ++s) {
    PetscCheckFalse(stagCoord->dof[s] !=0 && stagCoord->dof[s] != 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Coordinate DM in 1 dimensions must have 0 or 1 dof on each stratum, but stratum %d has %d dof",s,stagCoord->dof[s]);
  }
  PetscCall(DMCreateLocalVector(dmCoord,&coordLocal));

  PetscCall(DMStagVecGetArray(dmCoord,coordLocal,&arr));
  if (stagCoord->dof[0]) {
    PetscCall(DMStagGetLocationSlot(dmCoord,DMSTAG_LEFT,0,&ileft));
  }
  if (stagCoord->dof[1]) {
    PetscCall(DMStagGetLocationSlot(dmCoord,DMSTAG_ELEMENT,0,&ielement));
  }
  PetscCall(DMStagGetGhostCorners(dmCoord,&start_ghost,NULL,NULL,&n_ghost,NULL,NULL));

  min = xmin;
  h = (xmax-xmin)/stagCoord->N[0];

  for (PetscInt ind=start_ghost; ind<start_ghost + n_ghost; ++ind) {
    if (stagCoord->dof[0]) {
      const PetscReal off = 0.0;
        arr[ind][ileft] = min + ((PetscReal)ind + off) * h;
    }
    if (stagCoord->dof[1]) {
      const PetscReal off = 0.5;
        arr[ind][ielement] = min + ((PetscReal)ind + off) * h;
    }
  }
  PetscCall(DMStagVecRestoreArray(dmCoord,coordLocal,&arr));
  PetscCall(DMSetCoordinatesLocal(dm,coordLocal));
  PetscCall(PetscLogObjectParent((PetscObject)dm,(PetscObject)coordLocal));
  PetscCall(VecDestroy(&coordLocal));
  PetscFunctionReturn(0);
}

/* Helper functions used in DMSetUp_Stag() */
static PetscErrorCode DMStagComputeLocationOffsets_1d(DM);

PETSC_INTERN PetscErrorCode DMSetUp_Stag_1d(DM dm)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscMPIInt     size,rank;
  MPI_Comm        comm;
  PetscInt        j;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  /* Check Global size */
  PetscCheckFalse(stag->N[0] < 1,comm,PETSC_ERR_ARG_OUTOFRANGE,"Global grid size of %D < 1 specified",stag->N[0]);

  /* Local sizes */
  PetscCheckFalse(stag->N[0] < size,comm,PETSC_ERR_ARG_OUTOFRANGE,"More ranks (%d) than elements (%D) specified",size,stag->N[0]);
  if (!stag->l[0]) {
    /* Divide equally, giving an extra elements to higher ranks */
    PetscCall(PetscMalloc1(stag->nRanks[0],&stag->l[0]));
    for (j=0; j<stag->nRanks[0]; ++j) stag->l[0][j] = stag->N[0]/stag->nRanks[0] + (stag->N[0] % stag->nRanks[0] > j ? 1 : 0);
  }
  {
    PetscInt Nchk = 0;
    for (j=0; j<size; ++j) Nchk += stag->l[0][j];
    PetscCheckFalse(Nchk != stag->N[0],comm,PETSC_ERR_ARG_OUTOFRANGE,"Sum of specified local sizes (%D) is not equal to global size (%D)",Nchk,stag->N[0]);
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
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Starting element */
  stag->start[0] = 0;
  for (j=0; j<stag->rank[0]; ++j) stag->start[0] += stag->l[0][j];

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
          SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
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
          SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
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
          SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    default :
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Total size of ghosted/local representation */
  stag->entriesGhost = stag->nGhost[0]*stag->entriesPerElement;

  /* Define neighbors */
  PetscCall(PetscMalloc1(3,&stag->neighbors));
  if (stag->firstRank[0]) {
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:     stag->neighbors[0] = -1;                break;
      case DM_BOUNDARY_PERIODIC: stag->neighbors[0] = stag->nRanks[0]-1; break;
      default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
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
      default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }
  } else {
    stag->neighbors[2] = stag->rank[0]+1;
  }

  if (stag->n[0] < stag->stencilWidth) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStag 1d setup does not support local sizes (%d) smaller than the elementwise stencil width (%d)",stag->n[0],stag->stencilWidth);
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
        SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }

    PetscCall(PetscMalloc1(entriesToTransferTotal,&idxLocal));
    PetscCall(PetscMalloc1(entriesToTransferTotal,&idxGlobal));
    PetscCall(PetscMalloc1(stag->entriesGhost,&idxGlobalAll));
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
      if (stag->lastRank[0]) {
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
        for (iLocal=0; iLocal<ghostOffsetStart; ++iLocal) {
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
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);

    /* Create Local IS (transferring pointer ownership) */
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxLocal,PETSC_OWN_POINTER,&isLocal));

    /* Create Global IS (transferring pointer ownership) */
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxGlobal,PETSC_OWN_POINTER,&isGlobal));

    /* Create stag->gtol, which doesn't include dummy entries */
    {
      Vec local,global;
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global));
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local));
      PetscCall(VecScatterCreate(global,isGlobal,local,isLocal,&stag->gtol));
      PetscCall(VecDestroy(&global));
      PetscCall(VecDestroy(&local));
    }

    /* In special cases, create a dedicated injective local-to-global map */
    if (stag->boundaryType[0] == DM_BOUNDARY_PERIODIC && stag->nRanks[0] == 1) {
      PetscCall(DMStagPopulateLocalToGlobalInjective(dm));
    }

    /* Destroy ISs */
    PetscCall(ISDestroy(&isLocal));
    PetscCall(ISDestroy(&isGlobal));

    /* Create local-to-global map (transferring pointer ownership) */
    PetscCall(ISLocalToGlobalMappingCreate(comm,1,stag->entriesGhost,idxGlobalAll,PETSC_OWN_POINTER,&dm->ltogmap));
    PetscCall(PetscLogObjectParent((PetscObject)dm,(PetscObject)dm->ltogmap));
  }

  /* Precompute location offsets */
  PetscCall(DMStagComputeLocationOffsets_1d(dm));

  /* View from Options */
  PetscCall(DMViewFromOptions(dm,NULL,"-dm_view"));

 PetscFunctionReturn(0);
}

static PetscErrorCode DMStagComputeLocationOffsets_1d(DM dm)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt  epe = stag->entriesPerElement;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(DMSTAG_NUMBER_LOCATIONS,&stag->locationOffsets));
  stag->locationOffsets[DMSTAG_LEFT]    = 0;
  stag->locationOffsets[DMSTAG_ELEMENT] = stag->locationOffsets[DMSTAG_LEFT] + stag->dof[0];
  stag->locationOffsets[DMSTAG_RIGHT]   = stag->locationOffsets[DMSTAG_LEFT] + epe;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_1d(DM dm)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        *idxLocal,*idxGlobal;
  PetscInt        i,iLocal,d,count;
  IS              isLocal,isGlobal;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(stag->entries,&idxLocal));
  PetscCall(PetscMalloc1(stag->entries,&idxGlobal));
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
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxLocal,PETSC_OWN_POINTER,&isLocal));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxGlobal,PETSC_OWN_POINTER,&isGlobal));
  {
    Vec local,global;
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local));
    PetscCall(VecScatterCreate(local,isLocal,global,isGlobal,&stag->ltog_injective));
    PetscCall(VecDestroy(&global));
    PetscCall(VecDestroy(&local));
  }
  PetscCall(ISDestroy(&isLocal));
  PetscCall(ISDestroy(&isGlobal));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMCreateMatrix_Stag_1D_AIJ_Assemble(DM dm,Mat A)
{
  DMStagStencilType stencil_type;
  PetscInt          dof[2],start,n,n_extra,stencil_width,N,epe;
  DMBoundaryType    boundary_type_x;

  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL));
  PetscCall(DMStagGetStencilType(dm,&stencil_type));
  PetscCall(DMStagGetStencilWidth(dm,&stencil_width));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  PetscCall(DMStagGetEntriesPerElement(dm,&epe));
  PetscCall(DMStagGetBoundaryTypes(dm,&boundary_type_x,NULL,NULL));
  if (stencil_type == DMSTAG_STENCIL_NONE) {
    /* Couple all DOF at each location to each other */
    DMStagStencil *row_vertex,*row_element;

    PetscCall(PetscMalloc1(dof[0],&row_vertex));
    for (PetscInt c=0; c<dof[0]; ++c) {
      row_vertex[c].loc = DMSTAG_LEFT;
      row_vertex[c].c = c;
    }

    PetscCall(PetscMalloc1(dof[1],&row_element));
    for (PetscInt c=0; c<dof[1]; ++c) {
      row_element[c].loc = DMSTAG_ELEMENT;
      row_element[c].c = c;
    }

    for (PetscInt e=start; e<start+n+n_extra; ++e) {
      {
        for (PetscInt c=0; c<dof[0]; ++c){
          row_vertex[c].i = e;
        }
        PetscCall(DMStagMatSetValuesStencil(dm,A,dof[0],row_vertex,dof[0],row_vertex,NULL,INSERT_VALUES));
      }
      if (e < N) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          row_element[c].i = e;
        }
        PetscCall(DMStagMatSetValuesStencil(dm,A,dof[1],row_element,dof[1],row_element,NULL,INSERT_VALUES));
      }
    }
    PetscCall(PetscFree(row_vertex));
    PetscCall(PetscFree(row_element));
  } else if (stencil_type == DMSTAG_STENCIL_STAR || stencil_type == DMSTAG_STENCIL_BOX) {
    DMStagStencil *col,*row;

    PetscCall(PetscMalloc1(epe,&row));
    {
      PetscInt nrows = 0;
      for (PetscInt c=0; c<dof[0]; ++c) {
        row[nrows].c = c;
        row[nrows].loc = DMSTAG_LEFT;
        ++nrows;
      }
      for (PetscInt c=0; c<dof[1]; ++c) {
        row[nrows].c = c;
        row[nrows].loc = DMSTAG_ELEMENT;
        ++nrows;
      }
    }
    PetscCall(PetscMalloc1(epe,&col));
    {
      PetscInt ncols = 0;
      for (PetscInt c=0; c<dof[0]; ++c) {
        col[ncols].c = c;
        col[ncols].loc = DMSTAG_LEFT;
        ++ncols;
      }
      for (PetscInt c=0; c<dof[1]; ++c) {
        col[ncols].c = c;
        col[ncols].loc = DMSTAG_ELEMENT;
        ++ncols;
      }
    }
    for (PetscInt e=start; e<start+n+n_extra; ++e) {
      for (PetscInt i=0; i<epe; ++i) {
        row[i].i = e;
      }
      for (PetscInt offset = -stencil_width; offset<=stencil_width; ++offset) {
        const PetscInt e_offset = e + offset;

        /* Only set values corresponding to elements which can have non-dummy entries,
           meaning those that map to unknowns in the global representation. In the periodic
           case, this is the entire stencil, but in all other cases, only includes a single
           "extra" element which is partially outside the physical domain (those points in the
           global representation */
        if (boundary_type_x == DM_BOUNDARY_PERIODIC || (e_offset < N+1 && e_offset >= 0)) {
          for (PetscInt i=0; i<epe; ++i) {
            col[i].i = e_offset;
          }
          PetscCall(DMStagMatSetValuesStencil(dm,A,epe,row,epe,col,NULL,INSERT_VALUES));
        }
      }
    }
    PetscCall(PetscFree(row));
    PetscCall(PetscFree(col));
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported stencil type %s",DMStagStencilTypes[stencil_type]);
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
