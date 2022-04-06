
#include <petscvec.h>
#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

typedef struct {
  PetscBool useGrid;        /* Flag to use a grid layout */
  PetscInt  gridDim;        /* The grid dimension */
  PetscInt  nodeGrid[3];    /* Dimension of node grid */
  PetscInt  processGrid[3]; /* Dimension of local process grid on each node */
} PetscPartitioner_Simple;

static PetscErrorCode PetscPartitionerDestroy_Simple(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Simple_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Simple(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscPartitionerView_Simple_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  PetscInt                 num, i;
  PetscBool                flg;

  PetscFunctionBegin;
  for (i = 0; i < 3; ++i) p->processGrid[i] = p->nodeGrid[i] = 1;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscPartitioner Simple Options");
  num  = 3;
  PetscCall(PetscOptionsIntArray("-petscpartitioner_simple_node_grid", "Number of nodes in each dimension", "", p->nodeGrid, &num, &flg));
  if (flg) {p->useGrid = PETSC_TRUE; p->gridDim = num;}
  num  = 3;
  PetscCall(PetscOptionsIntArray("-petscpartitioner_simple_process_grid", "Number of local processes in each dimension for a given node", "", p->processGrid, &num, &flg));
  if (flg) {
    p->useGrid = PETSC_TRUE;
    if (p->gridDim < 0) p->gridDim = num;
    else PetscCheck(p->gridDim == num,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_INCOMP, "Process grid dimension %" PetscInt_FMT " != %" PetscInt_FMT " node grid dimension", num, p->gridDim);
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Simple_Grid(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  const PetscInt          *nodes = p->nodeGrid;
  const PetscInt          *procs = p->processGrid;
  PetscInt                *cellproc, *offsets, cells[3] = {1, 1, 1}, pcells[3] = {1, 1, 1};
  PetscInt                 Np    = 1, Nr, np, nk, nj, ni, pk, pj, pi, ck, cj, ci, i;
  MPI_Comm                 comm;
  PetscMPIInt              size;

  PetscFunctionBegin;
  if (vertSection)   PetscCall(PetscInfo(part, "PETSCPARTITIONERSIMPLE ignores vertex weights when using grid partition\n"));
  if (targetSection) PetscCall(PetscInfo(part, "PETSCPARTITIONERSIMPLE ignores partition weights when using grid partition\n"));
  PetscCall(PetscObjectGetComm((PetscObject) part, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Check grid */
  for (i = 0; i < 3; ++i) Np *= nodes[i]*procs[i];
  PetscCheck(nparts == Np,comm, PETSC_ERR_ARG_INCOMP, "Number of partitions %" PetscInt_FMT " != %" PetscInt_FMT " grid size", nparts, Np);
  PetscCheck(nparts == size,comm, PETSC_ERR_ARG_INCOMP, "Number of partitions %" PetscInt_FMT " != %d processes", nparts, size);
  PetscCheck(numVertices % nparts == 0,comm, PETSC_ERR_ARG_INCOMP, "Number of cells %" PetscInt_FMT " is not divisible by number of partitions %" PetscInt_FMT, numVertices, nparts);
  for (i = 0; i < p->gridDim; ++i) cells[i] = nodes[i]*procs[i];
  Nr = numVertices / nparts;
  while (Nr > 1) {
    for (i = 0; i < p->gridDim; ++i) {
      cells[i] *= 2;
      Nr       /= 2;
    }
  }
  PetscCheckFalse(numVertices && Nr != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Odd number of cells %" PetscInt_FMT ". Must be nprocs*2^k", numVertices);
  for (i = 0; i < p->gridDim; ++i) {
    PetscCheck(cells[i] %  (nodes[i]*procs[i]) == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "dir %" PetscInt_FMT ". Number of cells (%" PetscInt_FMT ") mod number of processors %" PetscInt_FMT, i, cells[i], nodes[i]*procs[i]);
    pcells[i] = cells[i] / (nodes[i]*procs[i]);
  }
  /* Compute sizes */
  for (np = 0; np < nparts; ++np) PetscCall(PetscSectionSetDof(partSection, np, numVertices/nparts));
  PetscCall(PetscSectionSetUp(partSection));
  PetscCall(PetscCalloc1(nparts, &offsets));
  for (np = 0; np < nparts; ++np) PetscCall(PetscSectionGetOffset(partSection, np, &offsets[np]));
  if (!numVertices) pcells[0] = pcells[1] = pcells[2] = 0;
  /* Compute partition */
  PetscCall(PetscMalloc1(numVertices, &cellproc));
  for (nk = 0; nk < nodes[2]; ++nk) {
    for (nj = 0; nj < nodes[1]; ++nj) {
      for (ni = 0; ni < nodes[0]; ++ni) {
        const PetscInt nid = (nk*nodes[1] + nj)*nodes[0] + ni;

        for (pk = 0; pk < procs[2]; ++pk) {
          for (pj = 0; pj < procs[1]; ++pj) {
            for (pi = 0; pi < procs[0]; ++pi) {
              const PetscInt pid = ((nid*procs[2] + pk)*procs[1] + pj)*procs[0] + pi;

              /* Assume that cells are originally numbered lexicographically */
              for (ck = 0; ck < pcells[2]; ++ck) {
                for (cj = 0; cj < pcells[1]; ++cj) {
                  for (ci = 0; ci < pcells[0]; ++ci) {
                    const PetscInt cid = (((nk*procs[2] + pk)*pcells[2] + ck)*cells[1] + ((nj*procs[1] + pj)*pcells[1] + cj))*cells[0] + (ni*procs[0] + pi)*pcells[0] + ci;

                    cellproc[offsets[pid]++] = cid;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (np = 1; np < nparts; ++np) PetscCheckFalse(offsets[np] - offsets[np-1] != numVertices/nparts,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Offset %" PetscInt_FMT " != %" PetscInt_FMT " partition size", offsets[np], numVertices/nparts);
  PetscCall(PetscFree(offsets));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numVertices, cellproc, PETSC_OWN_POINTER, partition));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Simple(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  MPI_Comm       comm;
  PetscInt       np, *tpwgts = NULL, sumw = 0, numVerticesGlobal  = 0;
  PetscMPIInt    size;

  PetscFunctionBegin;
  if (p->useGrid) {
    PetscCall(PetscPartitionerPartition_Simple_Grid(part, nparts, numVertices, start, adjacency, vertSection, targetSection, partSection, partition));
    PetscFunctionReturn(0);
  }
  if (vertSection) PetscCall(PetscInfo(part,"PETSCPARTITIONERSIMPLE ignores vertex weights\n"));
  PetscCall(PetscObjectGetComm((PetscObject) part, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (targetSection) {
    PetscCall(MPIU_Allreduce(&numVertices, &numVerticesGlobal, 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscCalloc1(nparts,&tpwgts));
    for (np = 0; np < nparts; ++np) {
      PetscCall(PetscSectionGetDof(targetSection,np,&tpwgts[np]));
      sumw += tpwgts[np];
    }
    if (sumw) {
      PetscInt m,mp;
      for (np = 0; np < nparts; ++np) tpwgts[np] = (tpwgts[np]*numVerticesGlobal)/sumw;
      for (np = 0, m = -1, mp = 0, sumw = 0; np < nparts; ++np) {
        if (m < tpwgts[np]) { m = tpwgts[np]; mp = np; }
        sumw += tpwgts[np];
      }
      if (sumw != numVerticesGlobal) tpwgts[mp] += numVerticesGlobal - sumw;
    }
    if (!sumw) PetscCall(PetscFree(tpwgts));
  }

  PetscCall(ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition));
  if (size == 1) {
    if (tpwgts) {
      for (np = 0; np < nparts; ++np) {
        PetscCall(PetscSectionSetDof(partSection, np, tpwgts[np]));
      }
    } else {
      for (np = 0; np < nparts; ++np) {
        PetscCall(PetscSectionSetDof(partSection, np, numVertices/nparts + ((numVertices % nparts) > np)));
      }
    }
  } else {
    if (tpwgts) {
      Vec         v;
      PetscScalar *array;
      PetscInt    st,j;
      PetscMPIInt rank;

      PetscCall(VecCreate(comm,&v));
      PetscCall(VecSetSizes(v,numVertices,numVerticesGlobal));
      PetscCall(VecSetType(v,VECSTANDARD));
      PetscCallMPI(MPI_Comm_rank(comm,&rank));
      for (np = 0,st = 0; np < nparts; ++np) {
        if (rank == np || (rank == size-1 && size < nparts && np >= size)) {
          for (j = 0; j < tpwgts[np]; j++) {
            PetscCall(VecSetValue(v,st+j,np,INSERT_VALUES));
          }
        }
        st += tpwgts[np];
      }
      PetscCall(VecAssemblyBegin(v));
      PetscCall(VecAssemblyEnd(v));
      PetscCall(VecGetArray(v,&array));
      for (j = 0; j < numVertices; ++j) {
        PetscCall(PetscSectionAddDof(partSection,PetscRealPart(array[j]),1));
      }
      PetscCall(VecRestoreArray(v,&array));
      PetscCall(VecDestroy(&v));
    } else {
      PetscMPIInt rank;
      PetscInt nvGlobal, *offsets, myFirst, myLast;

      PetscCall(PetscMalloc1(size+1,&offsets));
      offsets[0] = 0;
      PetscCallMPI(MPI_Allgather(&numVertices,1,MPIU_INT,&offsets[1],1,MPIU_INT,comm));
      for (np = 2; np <= size; np++) {
        offsets[np] += offsets[np-1];
      }
      nvGlobal = offsets[size];
      PetscCallMPI(MPI_Comm_rank(comm,&rank));
      myFirst = offsets[rank];
      myLast  = offsets[rank + 1] - 1;
      PetscCall(PetscFree(offsets));
      if (numVertices) {
        PetscInt firstPart = 0, firstLargePart = 0;
        PetscInt lastPart = 0, lastLargePart = 0;
        PetscInt rem = nvGlobal % nparts;
        PetscInt pSmall = nvGlobal/nparts;
        PetscInt pBig = nvGlobal/nparts + 1;

        if (rem) {
          firstLargePart = myFirst / pBig;
          lastLargePart  = myLast  / pBig;

          if (firstLargePart < rem) {
            firstPart = firstLargePart;
          } else {
            firstPart = rem + (myFirst - (rem * pBig)) / pSmall;
          }
          if (lastLargePart < rem) {
            lastPart = lastLargePart;
          } else {
            lastPart = rem + (myLast - (rem * pBig)) / pSmall;
          }
        } else {
          firstPart = myFirst / (nvGlobal/nparts);
          lastPart  = myLast  / (nvGlobal/nparts);
        }

        for (np = firstPart; np <= lastPart; np++) {
          PetscInt PartStart =  np    * (nvGlobal/nparts) + PetscMin(nvGlobal % nparts,np);
          PetscInt PartEnd   = (np+1) * (nvGlobal/nparts) + PetscMin(nvGlobal % nparts,np+1);

          PartStart = PetscMax(PartStart,myFirst);
          PartEnd   = PetscMin(PartEnd,myLast+1);
          PetscCall(PetscSectionSetDof(partSection,np,PartEnd-PartStart));
        }
      }
    }
  }
  PetscCall(PetscFree(tpwgts));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_Simple(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph             = PETSC_TRUE;
  part->ops->view           = PetscPartitionerView_Simple;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_Simple;
  part->ops->destroy        = PetscPartitionerDestroy_Simple;
  part->ops->partition      = PetscPartitionerPartition_Simple;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERSIMPLE = "simple" - A PetscPartitioner object

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Simple(PetscPartitioner part)
{
  PetscPartitioner_Simple *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNewLog(part, &p));
  p->gridDim = -1;
  part->data = p;

  PetscCall(PetscPartitionerInitialize_Simple(part));
  PetscFunctionReturn(0);
}
