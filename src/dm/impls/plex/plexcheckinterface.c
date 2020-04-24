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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Type_size(dt, &unitsize);CHKERRQ(ierr);
  ierr = PetscObjectGetComm(obj, &comm);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrranks, &rsize, nrranks, &rarr);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrranks, &rreq, nsranks, &sreq);CHKERRQ(ierr);
  /* exchange array size */
  ierr = PetscObjectGetNewTag(obj,&tag);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    ierr = MPI_Irecv(&rsize[r], 1, MPIU_INT, rranks[r], tag, comm, &rreq[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = MPI_Isend(&ssize[r], 1, MPIU_INT, sranks[r], tag, comm, &sreq[r]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(nsranks, sreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  /* exchange array */
  ierr = PetscObjectGetNewTag(obj,&tag);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    ierr = PetscMalloc(rsize[r]*unitsize, &rarr[r]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rarr[r], rsize[r], dt, rranks[r], tag, comm, &rreq[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = MPI_Isend(sarr[r], ssize[r], dt, sranks[r], tag, comm, &sreq[r]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nrranks, rreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(nsranks, sreq, MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree2(rreq, sreq);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc4(nsranks, &ssize, nsranks, &sarr, nrranks, &rreq, nsranks, &sreq);CHKERRQ(ierr);
  for (r=0; r<nsranks; r++) {
    ierr = VecGetLocalSize(svecs[r], &ssize[r]);CHKERRQ(ierr);
    ierr = VecGetArrayRead(svecs[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = ExchangeArrayByRank_Private(obj, MPIU_SCALAR, nsranks, sranks, ssize, (const void**)sarr, nrranks, rranks, &rsize, (void***)&rarr);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrranks, &rvecs_);CHKERRQ(ierr);
  for (r=0; r<nrranks; r++) {
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, rsize[r], NULL, &rvecs_[r]);CHKERRQ(ierr);
    ierr = VecReplaceArray(rvecs_[r], rarr[r]);CHKERRQ(ierr);
  }
  for (r=0; r<nsranks; r++) {
    ierr = VecRestoreArrayRead(svecs[r], &sarr[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rsize, rarr);CHKERRQ(ierr);
  ierr = PetscFree4(ssize, sarr, rreq, sreq);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);
  nleaves = roffset[nranks];
  ierr = PetscMalloc2(nleaves, rmine1, nleaves, rremote1);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    ierr = PetscArraycpy(&(*rmine1)[o], &rmine[o], n);CHKERRQ(ierr);
    ierr = PetscArraycpy(&(*rremote1)[o], &rremote[o], n);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n, &(*rremote1)[o], &(*rmine1)[o]);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocalSetUp(dm);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks, coordinatesPerRank);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    o = roffset[r];
    n = roffset[r+1] - o;
    ierr = ISCreateGeneral(PETSC_COMM_SELF, n, &rmine[o], PETSC_USE_POINTER, &pointsPerRank);CHKERRQ(ierr);
    ierr = DMPlexGetConeRecursiveVertices(dm, pointsPerRank, &conesPerRank);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocalTuple(dm, conesPerRank, NULL, &(*coordinatesPerRank)[r]);CHKERRQ(ierr);
    ierr = ISDestroy(&pointsPerRank);CHKERRQ(ierr);
    ierr = ISDestroy(&conesPerRank);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFComputeMultiRootOriginalNumberingByRank_Private(PetscSF sf, PetscSF imsf, PetscInt *irmine1[])
{
  PetscInt            *mRootsOrigNumbering;
  PetscInt            nileaves, niranks;
  const PetscInt      *iroffset, *irmine, *degree;
  PetscInt            i, n, o, r;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(imsf, &niranks, NULL, &iroffset, &irmine, NULL);CHKERRQ(ierr);
  if (PetscUnlikely(nileaves != iroffset[niranks])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"nileaves != iroffset[niranks])");
  ierr = PetscSFComputeDegreeBegin(sf, &degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &degree);CHKERRQ(ierr);
  ierr = PetscSFComputeMultiRootOriginalNumbering(sf, degree, NULL, &mRootsOrigNumbering);CHKERRQ(ierr);
  ierr = PetscMalloc1(nileaves, irmine1);CHKERRQ(ierr);
  for (r=0; r<niranks; r++) {
    o = iroffset[r];
    n = iroffset[r+1] - o;
    for (i=0; i<n; i++) (*irmine1)[o+i] = mRootsOrigNumbering[irmine[o+i]];
  }
  ierr = PetscFree(mRootsOrigNumbering);CHKERRQ(ierr);
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
  Vec                 *recCoordinatesPerRank=0;
  PetscInt            r;
  PetscMPIInt         commsize, myrank;
  PetscBool           same;
  PetscBool           verbose=PETSC_FALSE;
  MPI_Comm            comm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &myrank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &commsize);CHKERRQ(ierr);
  if (commsize < 2) PetscFunctionReturn(0);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  if (!sf) PetscFunctionReturn(0);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &mine, &remote);CHKERRQ(ierr);
  if (nroots < 0) PetscFunctionReturn(0);
  if (!dm->coordinates && !dm->coordinatesLocal) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM coordinates must be set");
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);

  /* Expand sent cones per rank */
  ierr = SortByRemote_Private(sf, &rmine1, &rremote1);CHKERRQ(ierr);
  ierr = GetRecursiveConeCoordinatesPerRank_Private(dm, sf, rmine1, &sntCoordinatesPerRank);CHKERRQ(ierr);

  /* Create inverse SF */
  ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
  ierr = PetscSFCreateInverseSF(msf,&imsf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(imsf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(imsf, NULL, &nileaves, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(imsf, &niranks, &iranks, &iroffset, &irmine, &irremote);CHKERRQ(ierr);

  /* Compute original numbering of multi-roots (referenced points) */
  ierr = PetscSFComputeMultiRootOriginalNumberingByRank_Private(sf, imsf, &mine_orig_numbering);CHKERRQ(ierr);

  /* Expand coordinates of the referred cones per rank */
  ierr = GetRecursiveConeCoordinatesPerRank_Private(dm, imsf, mine_orig_numbering, &refCoordinatesPerRank);CHKERRQ(ierr);

  /* Send the coordinates */
  ierr = ExchangeVecByRank_Private((PetscObject)sf, nranks, ranks, sntCoordinatesPerRank, niranks, iranks, &recCoordinatesPerRank);CHKERRQ(ierr);

  /* verbose output */
  ierr = PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_check_cones_conform_on_interfaces_verbose", &verbose, NULL);CHKERRQ(ierr);
  if (verbose) {
    PetscViewer sv, v = PETSC_VIEWER_STDOUT_WORLD;
    ierr = PetscViewerASCIIPrintf(v, "============\nDMPlexCheckInterfaceCones output\n============\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d] --------\n", myrank);CHKERRQ(ierr);
    for (r=0; r<nranks; r++) {
      ierr = PetscViewerASCIISynchronizedPrintf(v, "  r=%D ranks[r]=%d sntCoordinatesPerRank[r]:\n", r, ranks[r]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = VecView(sntCoordinatesPerRank[r], sv);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(v, "  ----------\n");CHKERRQ(ierr);
    for (r=0; r<niranks; r++) {
      ierr = PetscViewerASCIISynchronizedPrintf(v, "  r=%D iranks[r]=%d refCoordinatesPerRank[r]:\n", r, iranks[r]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = VecView(refCoordinatesPerRank[r], sv);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(v, "  ----------\n");CHKERRQ(ierr);
    for (r=0; r<niranks; r++) {
      ierr = PetscViewerASCIISynchronizedPrintf(v, "  r=%D iranks[r]=%d recCoordinatesPerRank[r]:\n", r, iranks[r]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = VecView(recCoordinatesPerRank[r], sv);CHKERRQ(ierr);
      ierr = PetscViewerRestoreSubViewer(v,PETSC_COMM_SELF,&sv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
    }
  }

  /* Compare recCoordinatesPerRank with refCoordinatesPerRank */
  for (r=0; r<niranks; r++) {
    ierr = VecEqual(refCoordinatesPerRank[r], recCoordinatesPerRank[r], &same);CHKERRQ(ierr);
    if (!same) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "interface cones do not conform for remote rank %d", iranks[r]);
  }

  /* destroy sent stuff */
  for (r=0; r<nranks; r++) {
    ierr = VecDestroy(&sntCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sntCoordinatesPerRank);CHKERRQ(ierr);
  ierr = PetscFree2(rmine1, rremote1);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&imsf);CHKERRQ(ierr);

  /* destroy referenced stuff */
  for (r=0; r<niranks; r++) {
    ierr = VecDestroy(&refCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(refCoordinatesPerRank);CHKERRQ(ierr);
  ierr = PetscFree(mine_orig_numbering);CHKERRQ(ierr);

  /* destroy received stuff */
  for (r=0; r<niranks; r++) {
    ierr = VecDestroy(&recCoordinatesPerRank[r]);CHKERRQ(ierr);
  }
  ierr = PetscFree(recCoordinatesPerRank);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
