#include <petscsys.h>
#include <private/vecimpl.h>
#include <sieve/BasicCommunication.hh>

typedef struct _n_PetscOverlap *PetscOverlap;
struct _n_PetscOverlap
{
  MPI_Comm  comm;
  PetscInt  numRanks;     // Number of partner processes
  PetscInt *ranks;        // MPI Rank of each partner process
  PetscInt *pointsOffset; // Offset into points array for each partner process
  PetscInt *points;       // Points array for each partner process, in sorted order
  PetscInt *remotePoints; // Remote points array for each partner process
};

PetscErrorCode PetscOverlapCreate(MPI_Comm comm, PetscOverlap *o)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscOverlap, o);CHKERRQ(ierr);
  (*o)->comm         = comm;
  (*o)->numRanks     = 0;
  (*o)->ranks        = PETSC_NULL;
  (*o)->pointsOffset = PETSC_NULL;
  (*o)->points       = PETSC_NULL;
  (*o)->remotePoints = PETSC_NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOverlapDestroy(PetscOverlap *o)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*o) PetscFunctionReturn(0);
  ierr = PetscFree((*o)->ranks);CHKERRQ(ierr);
  ierr = PetscFree((*o)->pointsOffset);CHKERRQ(ierr);
  ierr = PetscFree((*o)->points);CHKERRQ(ierr);
  ierr = PetscFree((*o)->remotePoints);CHKERRQ(ierr);
  ierr = PetscFree((*o));CHKERRQ(ierr);
  *o = PETSC_NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOverlapGetNumRanks(PetscOverlap overlap, PetscInt *numRanks)
{
  PetscFunctionBegin;
  PetscValidIntPointer(numRanks,2);
  *numRanks = overlap->numRanks;
  PetscFunctionReturn(0);
};

PetscErrorCode PetscOverlapGetRank(PetscOverlap overlap, PetscInt r, PetscInt *rank)
{
  PetscFunctionBegin;
  PetscValidIntPointer(rank,3);
  if (r < 0 || r >= overlap->numRanks) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rank index %d should be in [%d, %d)", r, 0, overlap->numRanks);}
  *rank = overlap->ranks[r];
  PetscFunctionReturn(0);
};

// TODO: Replace this with binary search
PetscErrorCode PetscOverlapGetRankIndex(PetscOverlap overlap, PetscInt rank, PetscInt *r)
{
  PetscInt p;

  PetscFunctionBegin;
  PetscValidIntPointer(r,3);
  for(p = 0; p < overlap->numRanks; ++p) {
    if (overlap->ranks[p] == rank) {
      *r = p;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid rank %d was not contained in this overlap", rank);
};

PetscErrorCode PetscOverlapGetNumPoints(PetscOverlap overlap, PetscInt r, PetscInt *numPoints)
{
  PetscFunctionBegin;
  PetscValidIntPointer(numPoints,3);
  if (r < 0 || r >= overlap->numRanks) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rank index %d should be in [%d, %d)", r, 0, overlap->numRanks);}
  *numPoints = overlap->pointsOffset[r+1] - overlap->pointsOffset[r];
  PetscFunctionReturn(0);
};

// These should be sorted
PetscErrorCode PetscOverlapGetPoints(PetscOverlap overlap, PetscInt r, const PetscInt **points)
{
  PetscFunctionBegin;
  PetscValidIntPointer(points,3);
  if (r < 0 || r >= overlap->numRanks) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rank index %d should be in [%d, %d)", r, 0, overlap->numRanks);}
  *points = &overlap->points[overlap->pointsOffset[r]];
  PetscFunctionReturn(0);
};

// These cannot be sorted
PetscErrorCode PetscOverlapGetRemotePoints(PetscOverlap overlap, PetscInt r, const PetscInt **remotePoints)
{
  PetscFunctionBegin;
  PetscValidIntPointer(remotePoints,3);
  if (r < 0 || r >= overlap->numRanks) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rank index %d should be in [%d, %d)", r, 0, overlap->numRanks);}
  *remotePoints = &overlap->remotePoints[overlap->pointsOffset[r]];
  PetscFunctionReturn(0);
};

PetscErrorCode PetscOverlapGetNumPointsByRank(PetscOverlap overlap, PetscInt rank, PetscInt *numPoints)
{
  PetscInt       r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(numPoints,3);
  ierr = PetscOverlapGetRankIndex(overlap, rank, &r);CHKERRQ(ierr);
  *numPoints = overlap->pointsOffset[r+1] - overlap->pointsOffset[r];
  PetscFunctionReturn(0);
};

// These should be sorted
PetscErrorCode PetscOverlapGetPointsByRank(PetscOverlap overlap, PetscInt rank, const PetscInt **points)
{
  PetscInt       r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(points,3);
  ierr = PetscOverlapGetRankIndex(overlap, rank, &r);CHKERRQ(ierr);
  *points = &overlap->points[overlap->pointsOffset[r]];
  PetscFunctionReturn(0);
};

typedef PetscInt point_type;

// Copies the PetscUniformSection from one process to another, specified by the overlap
//   Moves the chart interval [pStart, pEnd) and the numDof
PetscErrorCode PetscCopySection(PetscOverlap sendOverlap, PetscOverlap recvOverlap, const PetscUniformSection sendSection, const PetscUniformSection recvSection) {
  PetscInt       numSendRanks; // This is sendOverlap->base()
  PetscInt       numRecvRanks; // This is recvOverlap->cap()
  const int      debug = 0;
  point_type   **sendPoints;
  point_type   **recvPoints;
  PetscErrorCode ierr;
  ALE::MPIMover<point_type> pMover(sendSection->comm, debug);

  PetscFunctionBegin;
  ierr = PetscOverlapGetNumRanks(sendOverlap, &numSendRanks);CHKERRQ(ierr);
  ierr = PetscOverlapGetNumRanks(recvOverlap, &numRecvRanks);CHKERRQ(ierr);
  ierr = PetscMalloc(numSendRanks * sizeof(point_type *), &sendPoints);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    PetscInt    rank;
    point_type *v;

    ierr = PetscOverlapGetRank(sendOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscMalloc(3 * sizeof(point_type), &v);CHKERRQ(ierr);
    v[0] = sendSection->pStart;
    v[1] = sendSection->pEnd;
    v[2] = sendSection->numDof;
    sendPoints[r] = v;
    pMover.send(rank, 3, sendPoints[r]);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Sending chart (%d, %d, %d) to process %d\n", pMover.commRank(), v[0], v[1], v[2], rank);CHKERRQ(ierr);}
  }

  ierr = PetscMalloc(numRecvRanks * sizeof(point_type *), &recvPoints);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    PetscInt    rank;
    point_type *v;

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscMalloc(3 * sizeof(point_type), &v);CHKERRQ(ierr);
    recvPoints[r] = v;
    pMover.recv(rank, 3, recvPoints[r]);
  }
  pMover.start();
  pMover.end();
  point_type min = -1;
  point_type max = -1;
  point_type dof = -1;

  if (!numRecvRanks) {min = max = 0;}
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    const point_type *v      = recvPoints[r];
    point_type        newMin = v[0];
    point_type        newMax = v[1];
    PetscInt          rank;

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Received chart (%d, %d, %d) from process %d\n", pMover.commRank(), v[0],v[1], v[2], rank);CHKERRQ(ierr);}
    // Take the union of all charts
    if (min < 0) {
      min = newMin;
      max = newMax;
    } else {
      min = std::min(min, newMin);
      max = std::max(max, newMax);
    }
    if (dof < 0) {
      dof = v[2];
    } else {
      if (dof != v[2]) {SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of dof %d from rank %d should be %d", v[2], rank, dof);}
    }
  }
  recvSection->pStart = min;
  recvSection->pEnd   = max;
  recvSection->numDof = dof;
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    ierr = PetscFree(sendPoints[r]);CHKERRQ(ierr);
  }
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    ierr = PetscFree(recvPoints[r]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
};

// Full Sequence:
//   PetscCopySection(sOv, rOv, sSec, sStor, rSec, rStor)
//   --> PetscCopySection(sOv, rOv, sSec->atlasLayout, sSec->atlasDof, rSec->atlasLayout, rSec->atlasDof)
//       --> PetscCopyChart(sOv, rOv, sSec->atlasLayout, rSec->atlasLayout)
//       --> Copy sSec->atlasDof to rSec->atlasDof
//   --> Copy sStor to rStor
//
// RecvSection
//   1) Usually has scattered values, so using an interval chart is wrong
//   2) Must differentiate between data from different ranks (wait for fuse() process to merge)
//      Should we combine the fuse step here?

// Copies the PetscSection from one process to another, specified by the overlap
//   Use MPI_DATATYPE_NULL for a default
template<typename section_type, typename send_value_type, typename recv_value_type>
PetscErrorCode PetscCopySection(PetscOverlap sendOverlap, PetscOverlap recvOverlap, section_type sendSection, section_type recvSection, send_value_type *sendStorage, recv_value_type **recvStorage) {
  PetscInt         numSendRanks; // This is sendOverlap->base()
  PetscInt         numRecvRanks; // This is recvOverlap->cap()
  PetscInt         recvSize;
  send_value_type *sendValues;
  send_value_type *recvValues;
  const int        debug = 1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!sendStorage) {
    ierr = PetscCopySection(sendOverlap, recvOverlap, sendSection, recvSection);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    ierr = PetscCopySection(sendOverlap, recvOverlap, sendSection->atlasLayout, recvSection->atlasLayout, sendSection->atlasDof, &recvSection->atlasDof);CHKERRQ(ierr);
  }
  ALE::MPIMover<send_value_type> vMover(sendSection->comm, debug);

  ierr = PetscOverlapGetNumRanks(sendOverlap, &numSendRanks);CHKERRQ(ierr);
  ierr = PetscOverlapGetNumRanks(recvOverlap, &numRecvRanks);CHKERRQ(ierr);
  ierr = PetscMalloc(numSendRanks * sizeof(send_value_type *), &sendValues);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    PetscInt        rank;
    PetscInt        numPoints;
    const PetscInt *points; // This is sendOverlap->cone(rank)
    PetscInt        numVals   = 0;

    ierr = PetscOverlapGetRank(sendOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(sendOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(sendOverlap, r, &points);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim;

      ierr = PetscSectionGetDof(sendSection, points[p], &fDim);CHKERRQ(ierr);
      numVals += fDim;
    }
    send_value_type *v;
    PetscInt         k = 0;

    ierr = PetscMalloc(numVals * sizeof(send_value_type), &v);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim, off;

      ierr = PetscSectionGetDof(sendSection, points[p], &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sendSection, points[p], &off);
      for(PetscInt i = 0; i < fDim; ++i, ++k) {
        v[k] = sendStorage[off+i];
      }
    }
    sendValues[r] = v;
    vMover.send(r, numVals, v);
  }
  ierr = PetscSectionSetUp(recvSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(recvSection, &recvSize);CHKERRQ(ierr);
  ierr = PetscMalloc(recvSize * sizeof(recv_value_type), &recvStorage);CHKERRQ(ierr);

  ierr = PetscMalloc(numRecvRanks * sizeof(send_value_type *), &recvValues);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    PetscInt        rank;
    PetscInt        numPoints;
    const PetscInt *points;       // This is recvOverlap->support(rank)
    const PetscInt *remotePoints; // This is recvOverlap->support(rank).color()
    PetscInt        numVals      = 0;

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(recvOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(recvOverlap, r, &points);CHKERRQ(ierr);
    ierr = PetscOverlapGetRemotePoints(recvOverlap, r, &remotePoints);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim;

      ierr = PetscSectionGetDof(recvSection, recv_point_type(rank, remotePoints[p]), &fDim);CHKERRQ(ierr);
      numVals += fDim;
    }
    send_value_type *v;

    ierr = PetscMalloc(numVals * sizeof(send_value_type), &v);CHKERRQ(ierr);
    recvValues[r] = v;
    vMover.recv(rank, numVals, v);
  }
  vMover.start();
  vMover.end();

  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    PetscInt         rank;
    PetscInt         numPoints;
    const PetscInt  *points;       // This is recvOverlap->support(rank)
    const PetscInt  *remotePoints; // This is recvOverlap->support(rank).color()
    send_value_type *v = recvValues[r];
    PetscInt         p, k = 0;
    point_type      *sortedPoints;

    ierr = PetscMalloc(numPoints * sizeof(point_type), &sortedPoints);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      sortedPoints[p] = remotePoints[p];
    }
    ierr = PetscSortInt(numPoints, sortedPoints);CHKERRQ(ierr);

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(recvOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(recvOverlap, r, &points);CHKERRQ(ierr);
    ierr = PetscOverlapGetRemotePoints(recvOverlap, r, &remotePoints);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      const int fDim, off;

      ierr = PetscSectionGetDof(recvSection, recv_point_type(rank, sortedPoints[p]), &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(recvSection, recv_point_type(rank, sortedPoints[p]), &off);CHKERRQ(ierr);
      for(PetscInt i = 0; i < fDim; ++i, ++k) {
        recvStorage[off+i] = (recv_value_type) v[k];
      }
    }
    ierr = PetscFree(sortedPoints);CHKERRQ(ierr);
  }
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    ierr = PetscFree(sendValues[r]);CHKERRQ(ierr);
  }
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    ierr = PetscFree(recvValues[r]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
};

#if 0
// Copies the PetscSection from one process to another, specified by the overlap
//   Use MPI_DATATYPE_NULL for a default
template<typename send_value_type, typename recv_value_type>
PetscErrorCode PetscCopySection(PetscOverlap sendOverlap, PetscOverlap recvOverlap, PetscSection sendSection, PetscSection recvSection, const MPI_Datatype datatype, send_value_type *sendStorage, recv_value_type **recvStorage) {
  PetscInt         numSendRanks; // This is sendOverlap->base()
  PetscInt         numRecvRanks; // This is recvOverlap->cap()
  PetscInt         recvSize;
  send_value_type *sendValues;
  send_value_type *recvValues;
  const int        debug = 1;
  PetscErrorCode   ierr;
  ALE::MPIMover<send_value_type> vMover(sendSection->atlasLayout.comm, datatype, MPI_UNDEFINED, debug);

  ierr = PetscCopyChart(sendOverlap, recvOverlap, &sendSection->atlasLayout, &recvSection->atlasLayout);CHKERRQ(ierr);

  ierr = PetscOverlapGetNumRanks(sendOverlap, &numSendRanks);CHKERRQ(ierr);
  ierr = PetscOverlapGetNumRanks(recvOverlap, &numRecvRanks);CHKERRQ(ierr);
  ierr = PetscMalloc(numSendRanks * sizeof(send_value_type *), &sendValues);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    PetscInt        rank;
    PetscInt        numPoints;
    const PetscInt *points; // This is sendOverlap->cone(rank)
    PetscInt        numVals   = 0;

    ierr = PetscOverlapGetRank(sendOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(sendOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(sendOverlap, r, &points);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim;

      ierr = PetscSectionGetDof(sendSection, points[p], &fDim);CHKERRQ(ierr);
      numVals += fDim;
    }
    send_value_type *v;
    PetscInt         k = 0;

    ierr = PetscMalloc(numVals * sizeof(send_value_type), &v);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim, off;

      ierr = PetscSectionGetDof(sendSection, points[p], &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sendSection, points[p], &off);
      for(PetscInt i = 0; i < fDim; ++i, ++k) {
        v[k] = sendStorage[off+i];
      }
    }
    sendValues[r] = v;
    vMover.send(r, numVals, v);
  }
  ierr = PetscSectionSetUp(recvSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(recvSection, &recvSize);CHKERRQ(ierr);
  ierr = PetscMalloc(recvSize * sizeof(recv_value_type), &recvStorage);CHKERRQ(ierr);

  ierr = PetscMalloc(numRecvRanks * sizeof(send_value_type *), &recvValues);CHKERRQ(ierr);
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    PetscInt        rank;
    PetscInt        numPoints;
    const PetscInt *points;       // This is recvOverlap->support(rank)
    const PetscInt *remotePoints; // This is recvOverlap->support(rank).color()
    PetscInt        numVals      = 0;

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(recvOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(recvOverlap, r, &points);CHKERRQ(ierr);
    ierr = PetscOverlapGetRemotePoints(recvOverlap, r, &remotePoints);CHKERRQ(ierr);
    for(PetscInt p = 0; p < numPoints; ++p) {
      PetscInt fDim;

      ierr = PetscSectionGetDof(recvSection, recv_point_type(rank, remotePoints[p]), &fDim);CHKERRQ(ierr);
      numVals += fDim;
    }
    send_value_type *v;

    ierr = PetscMalloc(numVals * sizeof(send_value_type), &v);CHKERRQ(ierr);
    recvValues[r] = v;
    vMover.recv(rank, numVals, v);
  }
  vMover.start();
  vMover.end();

  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    PetscInt         rank;
    PetscInt         numPoints;
    const PetscInt  *points;       // This is recvOverlap->support(rank)
    const PetscInt  *remotePoints; // This is recvOverlap->support(rank).color()
    send_value_type *v = recvValues[r];
    PetscInt         p, k = 0;
    point_type      *sortedPoints;

    ierr = PetscMalloc(numPoints * sizeof(point_type), &sortedPoints);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      sortedPoints[p] = remotePoints[p];
    }
    ierr = PetscSortInt(numPoints, sortedPoints);CHKERRQ(ierr);

    ierr = PetscOverlapGetRank(recvOverlap, r, &rank);CHKERRQ(ierr);
    ierr = PetscOverlapGetNumPoints(recvOverlap, r, &numPoints);CHKERRQ(ierr);
    ierr = PetscOverlapGetPoints(recvOverlap, r, &points);CHKERRQ(ierr);
    ierr = PetscOverlapGetRemotePoints(recvOverlap, r, &remotePoints);CHKERRQ(ierr);
    for(p = 0; p < numPoints; ++p) {
      const int fDim, off;

      ierr = PetscSectionGetDof(recvSection, recv_point_type(rank, sortedPoints[p]), &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetOff(recvSection, recv_point_type(rank, sortedPoints[p]), &off);CHKERRQ(ierr);
      for(PetscInt i = 0; i < fDim; ++i, ++k) {
        recvStorage[off+i] = (recv_value_type) v[k];
      }
    }
    ierr = PetscFree(sortedPoints);CHKERRQ(ierr);
  }
  for(PetscInt r = 0; r < numSendRanks; ++r) {
    ierr = PetscFree(sendValues[r]);CHKERRQ(ierr);
  }
  for(PetscInt r = 0; r < numRecvRanks; ++r) {
    ierr = PetscFree(recvValues[r]);CHKERRQ(ierr);
  }
};
#endif
