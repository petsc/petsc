#include <petscsys.h>
#include <private/vecimpl.h>

typedef struct _n_PetscOverlap *PetscOverlap;
struct _n_PetscOverlap
{
  MPI_Comm  comm;
  PetscInt  numRanks;     // Number of partner processes
  PetscInt *ranks;        // MPI Rank of each partner process
  PetscInt *pointsOffset; // Offset into points array for each partner process
  PetscInt *points;       // Points array for each partner process, in sorted order
};

PetscErrorCode PetscOverlapCreate(MPI_Comm comm, PetscOverlap *o);
PetscErrorCode PetscOverlapDestroy(PetscOverlap *o);
PetscErrorCode PetscOverlapGetNumRanks(PetscOverlap overlap, PetscInt *numRanks);
PetscErrorCode PetscOverlapGetRank(PetscOverlap overlap, PetscInt r, PetscInt *rank);
PetscErrorCode PetscOverlapGetRankIndex(PetscOverlap overlap, PetscInt rank, PetscInt *r);
PetscErrorCode PetscOverlapGetNumPoints(PetscOverlap overlap, PetscInt r, PetscInt *numPoints);
PetscErrorCode PetscOverlapGetPoints(PetscOverlap overlap, PetscInt r, const PetscInt **points);
PetscErrorCode PetscOverlapGetNumPointsByRank(PetscOverlap overlap, PetscInt rank, PetscInt *numPoints);
PetscErrorCode PetscOverlapGetPointsByRank(PetscOverlap overlap, PetscInt rank, const PetscInt **points);

PetscErrorCode PetscCopySection(PetscOverlap sendOverlap, PetscOverlap recvOverlap, const PetscUniformSection sendSection, const PetscUniformSection recvSection);

PetscErrorCode BuildRingOverlap(MPI_Comm comm, PetscInt pStart, PetscInt pEnd, PetscOverlap *sendOverlap, PetscOverlap *recvOverlap)
{
  PetscMPIInt    numProcs, rank;
  PetscInt       r, p, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOverlapCreate(comm, sendOverlap);CHKERRQ(ierr);
  ierr = PetscOverlapCreate(comm, recvOverlap);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (numProcs == 1) PetscFunctionReturn(0);
  (*sendOverlap)->numRanks = numProcs == 2 ? 1 : 2;
  (*recvOverlap)->numRanks = numProcs == 2 ? 1 : 2;
  ierr = PetscMalloc((*sendOverlap)->numRanks * sizeof(PetscInt), &(*sendOverlap)->ranks);CHKERRQ(ierr);
  ierr = PetscMalloc((*recvOverlap)->numRanks * sizeof(PetscInt), &(*recvOverlap)->ranks);CHKERRQ(ierr);
  (*sendOverlap)->ranks[0] = !rank ? numProcs-1 : rank - 1;
  if ((*sendOverlap)->numRanks > 1) (*sendOverlap)->ranks[1] = (rank + 1)%numProcs;
  (*recvOverlap)->ranks[0] = !rank ? numProcs-1 : rank - 1;
  if ((*recvOverlap)->numRanks > 1) (*recvOverlap)->ranks[1] = (rank + 1)%numProcs;
  ierr = PetscMalloc(((*sendOverlap)->numRanks + 1) * sizeof(PetscInt), &(*sendOverlap)->pointsOffset);CHKERRQ(ierr);
  (*sendOverlap)->pointsOffset[0] = 0;
  for(r = 1; r <= (*sendOverlap)->numRanks; ++r) {
    (*sendOverlap)->pointsOffset[r] = (pEnd - pStart) + (*sendOverlap)->pointsOffset[r-1];
  }
  ierr = PetscMalloc((*sendOverlap)->pointsOffset[(*sendOverlap)->numRanks] * sizeof(PetscInt), &(*sendOverlap)->points);CHKERRQ(ierr);
  for(r = 0; r < (*sendOverlap)->numRanks; ++r) {
    for(p = (*sendOverlap)->pointsOffset[r], k = pStart; p < (*sendOverlap)->pointsOffset[r+1]; ++p, ++k) {
      (*sendOverlap)->points[p] = k;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestSameChart(MPI_Comm comm) {
  PetscOverlap   sendOverlap, recvOverlap;
  PetscInt       pStart, pEnd;
  struct _n_PetscUniformSection sendSection, recvSection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pStart = 5;
  pEnd   = 1001;
  ierr = BuildRingOverlap(comm, pStart, pEnd, &sendOverlap, &recvOverlap);CHKERRQ(ierr);
  sendSection.comm   = comm;
  sendSection.pStart = pStart;
  sendSection.pEnd   = pEnd;
  sendSection.numDof = 3;
  recvSection.comm   = comm;
  recvSection.pStart = -1;
  recvSection.pEnd   = -1;
  recvSection.numDof = -1;
  ierr = PetscCopySection(sendOverlap, recvOverlap, &sendSection, &recvSection);CHKERRQ(ierr);
  if (recvSection.pStart != sendSection.pStart) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pStart %d should be %d", recvSection.pStart, sendSection.pStart);}
  if (recvSection.pEnd   != sendSection.pEnd)   {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pEnd   %d should be %d", recvSection.pEnd,   sendSection.pEnd);}
  if (recvSection.numDof != sendSection.numDof) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section numDof %d should be %d", recvSection.numDof, sendSection.numDof);}
  ierr = PetscOverlapDestroy(&sendOverlap);CHKERRQ(ierr);
  ierr = PetscOverlapDestroy(&recvOverlap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestDifferentChart(MPI_Comm comm) {
  PetscOverlap   sendOverlap, recvOverlap;
  PetscInt       pStart, pEnd, numRecvRanks, r;
  struct _n_PetscUniformSection sendSection, recvSection;
  PetscMPIInt    numProcs, rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  pStart = rank;
  pEnd   = 1000+rank;
  ierr = BuildRingOverlap(comm, pStart, pEnd, &sendOverlap, &recvOverlap);CHKERRQ(ierr);
  sendSection.comm   = comm;
  sendSection.pStart = pStart;
  sendSection.pEnd   = pEnd;
  sendSection.numDof = 3;
  recvSection.comm   = comm;
  recvSection.pStart = -1;
  recvSection.pEnd   = -1;
  recvSection.numDof = -1;
  ierr = PetscCopySection(sendOverlap, recvOverlap, &sendSection, &recvSection);CHKERRQ(ierr);
  ierr = PetscOverlapGetNumRanks(recvOverlap, &numRecvRanks);CHKERRQ(ierr);
  pStart = numProcs;
  for(r = 0; r < numRecvRanks; ++r) {
    PetscInt recvRank;

    ierr = PetscOverlapGetRank(recvOverlap, r, &recvRank);CHKERRQ(ierr);
    pStart = PetscMin(pStart, recvRank);
  }
  if (recvSection.pStart != pStart) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pStart %d should be %d", recvSection.pStart, pStart);}
  pEnd = 1000;
  for(r = 0; r < numRecvRanks; ++r) {
    PetscInt recvRank;

    ierr = PetscOverlapGetRank(recvOverlap, r, &recvRank);CHKERRQ(ierr);
    pEnd = PetscMax(pEnd, 1000+recvRank);
  }
  if (recvSection.pEnd   != pEnd)   {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pEnd   %d should be %d", recvSection.pEnd,   pEnd);}
  if (recvSection.numDof != sendSection.numDof) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section numDof %d should be %d", recvSection.numDof, sendSection.numDof);}
  ierr = PetscOverlapDestroy(&sendOverlap);CHKERRQ(ierr);
  ierr = PetscOverlapDestroy(&recvOverlap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestSameSection(MPI_Comm comm) {
  PetscOverlap   sendOverlap, recvOverlap;
  PetscInt       pStart, pEnd, size, p;
  PetscSection   sendSection, recvSection;
  PetscInt      *sendStorage, *recvStorage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pStart = 0;
  pEnd   = 10;
  ierr = BuildRingOverlap(comm, pStart, pEnd, &sendOverlap, &recvOverlap);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &sendSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sendSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &recvSection);CHKERRQ(ierr);
  for(p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(sendSection, p, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sendSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(sendSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc(size * sizeof(PetscInt), &sendStorage);CHKERRQ(ierr);

  ierr = PetscCopySection(sendOverlap, recvOverlap, &sendSection->atlasLayout, &recvSection->atlasLayout, sendSection->atlasDof, &recvSection->atlasDof);CHKERRQ(ierr);

  if (recvSection.pStart != sendSection.pStart) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pStart %d should be %d", recvSection.pStart, sendSection.pStart);}
  if (recvSection.pEnd   != sendSection.pEnd)   {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section pEnd   %d should be %d", recvSection.pEnd,   sendSection.pEnd);}
  if (recvSection.numDof != sendSection.numDof) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Recv section numDof %d should be %d", recvSection.numDof, sendSection.numDof);}
  ierr = PetscFree(sendStorage);CHKERRQ(ierr);
  ierr = PetscOverlapDestroy(&sendOverlap);CHKERRQ(ierr);
  ierr = PetscOverlapDestroy(&recvOverlap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, 0, 0);CHKERRQ(ierr);
  ierr = TestSameChart(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = TestDifferentChart(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = TestSameSection(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
