#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/* TODO PetscArrayExchangeBegin/End */
/* TODO blocksize */
/* TODO move to API ? */
static PetscErrorCode ExchangeArrayByRank_Private(PetscObject obj, MPI_Datatype dt, PetscInt nsranks, const PetscMPIInt sranks[], PetscInt ssize[], const void *sarr[], PetscInt nrranks, const PetscMPIInt rranks[], PetscInt *rsize_out[], void **rarr_out[])
{
  PetscInt r;
  PetscInt *rsize;
  void **rarr;
  MPI_Request *sreq, *rreq;
  PetscMPIInt tag, unitsize;
  MPI_Comm comm;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Type_size(dt, &unitsize));
  CHKERRQ(PetscObjectGetComm(obj, &comm));
  CHKERRQ(PetscMalloc2(nrranks, &rsize, nrranks, &rarr));
  CHKERRQ(PetscMalloc2(nrranks, &rreq, nsranks, &sreq));
  /* exchange array size */
  CHKERRQ(PetscObjectGetNewTag(obj,&tag));
  for (r=0; r<nrranks; r++) {
    CHKERRMPI(MPI_Irecv(&rsize[r], 1, MPIU_INT, rranks[r], tag, comm, &rreq[r]));
  }
  for (r=0; r<nsranks; r++) {
    CHKERRMPI(MPI_Isend(&ssize[r], 1, MPIU_INT, sranks[r], tag, comm, &sreq[r]));
  }
  CHKERRMPI(MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE));
  CHKERRMPI(MPI_Waitall(nsranks, sreq, MPI_STATUSES_IGNORE));
  /* exchange array */
  CHKERRQ(PetscObjectGetNewTag(obj,&tag));
  for (r=0; r<nrranks; r++) {
    CHKERRQ(PetscMalloc(rsize[r]*unitsize, &rarr[r]));
    CHKERRMPI(MPI_Irecv(rarr[r], rsize[r], dt, rranks[r], tag, comm, &rreq[r]));
  }
  for (r=0; r<nsranks; r++) {
    CHKERRMPI(MPI_Isend(sarr[r], ssize[r], dt, sranks[r], tag, comm, &sreq[r]));
  }
  CHKERRMPI(MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE));
  CHKERRMPI(MPI_Waitall(nsranks, sreq, MPI_STATUSES_IGNORE));
  CHKERRQ(PetscFree2(rreq, sreq));
  *rsize_out = rsize;
  *rarr_out = rarr;
  PetscFunctionReturn(0);
}

/* TODO VecExchangeBegin/End */
/* TODO move to API ? */
static PetscErrorCode ExchangeVecByRank_Private(PetscObject obj, PetscInt nsranks, const PetscMPIInt sranks[], Vec svecs[], PetscInt nrranks, const PetscMPIInt rranks[], Vec *rvecs[])
{
  PetscInt r;
  PetscInt *ssize, *rsize;
  PetscScalar **rarr;
  const PetscScalar **sarr;
  Vec *rvecs_;
  MPI_Request *sreq, *rreq;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc4(nsranks, &ssize, nsranks, &sarr, nrranks, &rreq, nsranks, &sreq));
  for (r=0; r<nsranks; r++) {
    CHKERRQ(VecGetLocalSize(svecs[r], &ssize[r]));
    CHKERRQ(VecGetArrayRead(svecs[r], &sarr[r]));
  }
  CHKERRQ(ExchangeArrayByRank_Private(obj, MPIU_SCALAR, nsranks, sranks, ssize, (const void**)sarr, nrranks, rranks, &rsize, (void***)&rarr));
  CHKERRQ(PetscMalloc1(nrranks, &rvecs_));
  for (r=0; r<nrranks; r++) {
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, rsize[r], NULL, &rvecs_[r]));
    CHKERRQ(VecReplaceArray(rvecs_[r], rarr[r]));
  }
  for (r=0; r<nsranks; r++) {
    CHKERRQ(VecRestoreArrayRead(svecs[r], &sarr[r]));
  }
  CHKERRQ(PetscFree2(rsize, rarr));
  CHKERRQ(PetscFree4(ssize, sarr, rreq, sreq));
  *rvecs = rvecs_;
  PetscFunctionReturn(0);
}

static PetscErrorCode SortByRemote_Private(PetscSF sf, PetscInt *rmine1[], PetscInt *rremote1[])
{
  PetscInt            nleaves;
  PetscInt            nranks;
  const PetscMPIInt   *ranks;
  const PetscInt      *roffset, *rmine, *rremote;
  PetscInt            n, o, r;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote));
  nleaves = roffset[nranks];
  CHKERRQ(PetscMalloc2(nleaves, rmine1, nleaves, rremote1));
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    CHKERRQ(PetscArraycpy(&(*rmine1)[o], &rmine[o], n));
    CHKERRQ(PetscArraycpy(&(*rremote1)[o], &rremote[o], n));
    CHKERRQ(PetscSortIntWithArray(n, &(*rremote1)[o], &(*rmine1)[o]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GetRecursiveConeCoordinatesPerRank_Private(DM dm, PetscSF sf, PetscInt rmine[], Vec *coordinatesPerRank[])
{
  IS                  pointsPerRank, conesPerRank;
  PetscInt            nranks;
  const PetscMPIInt   *ranks;
  const PetscInt      *roffset;
  PetscInt            n, o, r;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinatesLocalSetUp(dm));
  CHKERRQ(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, NULL, NULL));
  CHKERRQ(PetscMalloc1(nranks, coordinatesPerRank));
  for (r=0; r<nranks; r++) {
    o = roffset[r];
    n = roffset[r+1] - o;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, n, &rmine[o], PETSC_USE_POINTER, &pointsPerRank));
    CHKERRQ(DMPlexGetConeRecursiveVertices(dm, pointsPerRank, &conesPerRank));
    CHKERRQ(DMGetCoordinatesLocalTuple(dm, conesPerRank, NULL, &(*coordinatesPerRank)[r]));
    CHKERRQ(ISDestroy(&pointsPerRank));
    CHKERRQ(ISDestroy(&conesPerRank));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFComputeMultiRootOriginalNumberingByRank_Private(PetscSF sf, PetscSF imsf, PetscInt *irmine1[])
{
  PetscInt            *mRootsOrigNumbering;
  PetscInt            nileaves, niranks;
  const PetscInt      *iroffset, *irmine, *degree;
  PetscInt            i, n, o, r;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL));
  CHKERRQ(PetscSFGetRootRanks(imsf, &niranks, NULL, &iroffset, &irmine, NULL));
  PetscCheckFalse(nileaves != iroffset[niranks],PETSC_COMM_SELF,PETSC_ERR_PLIB,"nileaves != iroffset[niranks])");
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &degree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &degree));
  CHKERRQ(PetscSFComputeMultiRootOriginalNumbering(sf, degree, NULL, &mRootsOrigNumbering));
  CHKERRQ(PetscMalloc1(nileaves, irmine1));
  for (r=0; r<niranks; r++) {
    o = iroffset[r];
    n = iroffset[r+1] - o;
    for (i=0; i<n; i++) (*irmine1)[o+i] = mRootsOrigNumbering[irmine[o+i]];
  }
  CHKERRQ(PetscFree(mRootsOrigNumbering));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckInterfaceCones - Check that points on inter-partition interfaces have conforming order of cone points.

  Input Parameters:
. dm - The DMPlex object

  Notes:
  For example, if there is an edge (rank,index)=(0,2) connecting points cone(0,2)=[(0,0),(0,1)] in this order, and the point SF containts connections 0 <- (1,0), 1 <- (1,1) and 2 <- (1,2),
  then this check would pass if the edge (1,2) has cone(1,2)=[(1,0),(1,1)]. By contrast, if cone(1,2)=[(1,1),(1,0)], then this check would fail.

  This is mainly intended for debugging/testing purposes. Does not check cone orientation, for this purpose use DMPlexCheckFaces().

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Developer Note:
  Interface cones are expanded into vertices and then their coordinates are compared.

  Level: developer

.seealso: DMPlexGetCone(), DMPlexGetConeSize(), DMGetPointSF(), DMGetCoordinates(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckInterfaceCones(DM dm)
{
  PetscSF             sf;
  PetscInt            nleaves, nranks, nroots;
  const PetscInt      *mine, *roffset, *rmine, *rremote;
  const PetscSFNode   *remote;
  const PetscMPIInt   *ranks;
  PetscSF             msf, imsf;
  PetscInt            nileaves, niranks;
  const PetscMPIInt   *iranks;
  const PetscInt      *iroffset, *irmine, *irremote;
  PetscInt            *rmine1, *rremote1; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  PetscInt            *mine_orig_numbering;
  Vec                 *sntCoordinatesPerRank;
  Vec                 *refCoordinatesPerRank;
  Vec                 *recCoordinatesPerRank=NULL;
  PetscInt            r;
  PetscMPIInt         commsize, myrank;
  PetscBool           same;
  PetscBool           verbose=PETSC_FALSE;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &myrank));
  CHKERRMPI(MPI_Comm_size(comm, &commsize));
  if (commsize < 2) PetscFunctionReturn(0);
  CHKERRQ(DMGetPointSF(dm, &sf));
  if (!sf) PetscFunctionReturn(0);
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleaves, &mine, &remote));
  if (nroots < 0) PetscFunctionReturn(0);
  PetscCheckFalse(!dm->coordinates && !dm->coordinatesLocal,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM coordinates must be set");
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote));

  /* Expand sent cones per rank */
  CHKERRQ(SortByRemote_Private(sf, &rmine1, &rremote1));
  CHKERRQ(GetRecursiveConeCoordinatesPerRank_Private(dm, sf, rmine1, &sntCoordinatesPerRank));

  /* Create inverse SF */
  CHKERRQ(PetscSFGetMultiSF(sf,&msf));
  CHKERRQ(PetscSFCreateInverseSF(msf,&imsf));
  CHKERRQ(PetscSFSetUp(imsf));
  CHKERRQ(PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL));
  CHKERRQ(PetscSFGetRootRanks(imsf, &niranks, &iranks, &iroffset, &irmine, &irremote));

  /* Compute original numbering of multi-roots (referenced points) */
  CHKERRQ(PetscSFComputeMultiRootOriginalNumberingByRank_Private(sf, imsf, &mine_orig_numbering));

  /* Expand coordinates of the referred cones per rank */
  CHKERRQ(GetRecursiveConeCoordinatesPerRank_Private(dm, imsf, mine_orig_numbering, &refCoordinatesPerRank));

  /* Send the coordinates */
  CHKERRQ(ExchangeVecByRank_Private((PetscObject)sf, nranks, ranks, sntCoordinatesPerRank, niranks, iranks, &recCoordinatesPerRank));

  /* verbose output */
  CHKERRQ(PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_check_cones_conform_on_interfaces_verbose", &verbose, NULL));
  if (verbose) {
    PetscViewer sv, v = PETSC_VIEWER_STDOUT_WORLD;
    CHKERRQ(PetscViewerASCIIPrintf(v, "============\nDMPlexCheckInterfaceCones output\n============\n"));
    CHKERRQ(PetscViewerASCIIPushSynchronized(v));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "[%d] --------\n", myrank));
    for (r=0; r<nranks; r++) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "  r=%D ranks[r]=%d sntCoordinatesPerRank[r]:\n", r, ranks[r]));
      CHKERRQ(PetscViewerASCIIPushTab(v));
      CHKERRQ(PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(VecView(sntCoordinatesPerRank[r], sv));
      CHKERRQ(PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(PetscViewerASCIIPopTab(v));
    }
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "  ----------\n"));
    for (r=0; r<niranks; r++) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "  r=%D iranks[r]=%d refCoordinatesPerRank[r]:\n", r, iranks[r]));
      CHKERRQ(PetscViewerASCIIPushTab(v));
      CHKERRQ(PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(VecView(refCoordinatesPerRank[r], sv));
      CHKERRQ(PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(PetscViewerASCIIPopTab(v));
    }
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "  ----------\n"));
    for (r=0; r<niranks; r++) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "  r=%D iranks[r]=%d recCoordinatesPerRank[r]:\n", r, iranks[r]));
      CHKERRQ(PetscViewerASCIIPushTab(v));
      CHKERRQ(PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(VecView(recCoordinatesPerRank[r], sv));
      CHKERRQ(PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv));
      CHKERRQ(PetscViewerASCIIPopTab(v));
    }
    CHKERRQ(PetscViewerFlush(v));
    CHKERRQ(PetscViewerASCIIPopSynchronized(v));
  }

  /* Compare recCoordinatesPerRank with refCoordinatesPerRank */
  for (r=0; r<niranks; r++) {
    CHKERRQ(VecEqual(refCoordinatesPerRank[r], recCoordinatesPerRank[r], &same));
    PetscCheck(same,PETSC_COMM_SELF, PETSC_ERR_PLIB, "interface cones do not conform for remote rank %d", iranks[r]);
  }

  /* destroy sent stuff */
  for (r=0; r<nranks; r++) {
    CHKERRQ(VecDestroy(&sntCoordinatesPerRank[r]));
  }
  CHKERRQ(PetscFree(sntCoordinatesPerRank));
  CHKERRQ(PetscFree2(rmine1, rremote1));
  CHKERRQ(PetscSFDestroy(&imsf));

  /* destroy referenced stuff */
  for (r=0; r<niranks; r++) {
    CHKERRQ(VecDestroy(&refCoordinatesPerRank[r]));
  }
  CHKERRQ(PetscFree(refCoordinatesPerRank));
  CHKERRQ(PetscFree(mine_orig_numbering));

  /* destroy received stuff */
  for (r=0; r<niranks; r++) {
    CHKERRQ(VecDestroy(&recCoordinatesPerRank[r]));
  }
  CHKERRQ(PetscFree(recCoordinatesPerRank));
  PetscFunctionReturn(0);
}
