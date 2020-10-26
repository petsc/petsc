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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(part->data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Simple_ASCII(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  PetscInt                 num, i;
  PetscBool                flg;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  for (i = 0; i < 3; ++i) p->processGrid[i] = p->nodeGrid[i] = 1;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner Simple Options");CHKERRQ(ierr);
  num  = 3;
  ierr = PetscOptionsIntArray("-petscpartitioner_simple_node_grid", "Number of nodes in each dimension", "", p->nodeGrid, &num, &flg);CHKERRQ(ierr);
  if (flg) {p->useGrid = PETSC_TRUE; p->gridDim = num;}
  num  = 3;
  ierr = PetscOptionsIntArray("-petscpartitioner_simple_process_grid", "Number of local processes in each dimension for a given node", "", p->processGrid, &num, &flg);CHKERRQ(ierr);
  if (flg) {
    p->useGrid = PETSC_TRUE;
    if (p->gridDim < 0) p->gridDim = num;
    else if (p->gridDim != num) SETERRQ2(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_INCOMP, "Process grid dimension %D != %D node grid dimension", num, p->gridDim);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Simple_Grid(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  const PetscInt          *nodes = p->nodeGrid;
  const PetscInt          *procs = p->processGrid;
  PetscInt                *cellproc, *offsets, cells[3] = {1, 1, 1}, pcells[3] = {1, 1, 1};
  PetscInt                 Np    = 1, Nlc = 1, Nr, np, nk, nj, ni, pk, pj, pi, ck, cj, ci, i;
  MPI_Comm                 comm;
  PetscMPIInt              size;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (vertSection)   {ierr = PetscInfo(part, "PETSCPARTITIONERSIMPLE ignores vertex weights when using grid partition\n");CHKERRQ(ierr);}
  if (targetSection) {ierr = PetscInfo(part, "PETSCPARTITIONERSIMPLE ignores partition weights when using grid partition\n");CHKERRQ(ierr);}
  ierr = PetscObjectGetComm((PetscObject) part, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  /* Check grid */
  for (i = 0; i < 3; ++i) Np *= nodes[i]*procs[i];
  if (nparts != Np)   SETERRQ2(comm, PETSC_ERR_ARG_INCOMP, "Number of partitions %D != %D grid size", nparts, Np);
  if (nparts != size) SETERRQ2(comm, PETSC_ERR_ARG_INCOMP, "Number of partitions %D != %D processes", nparts, size);
  if (numVertices % nparts) SETERRQ2(comm, PETSC_ERR_ARG_INCOMP, "Number of cells %D is not divisible by number of partitions %D", numVertices, nparts);
  for (i = 0; i < p->gridDim; ++i) cells[i] = nodes[i]*procs[i];
  Nr = numVertices / nparts;
  while (Nr > 1) {
    for (i = 0; i < p->gridDim; ++i) {
      cells[i] *= 2;
      Nr       /= 2;
    }
  }
  if (numVertices && Nr != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Odd number of cells %D. Must be nprocs*2^k", numVertices);
  for (i = 0; i < p->gridDim; ++i) {
    if (cells[i] %  (nodes[i]*procs[i])) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "dir %D. Number of cells (%D) mod number of processors %D", i, cells[i], nodes[i]*procs[i]);
    pcells[i] = cells[i] / (nodes[i]*procs[i]); Nlc *= pcells[i];
  }
  /* Compute sizes */
  for (np = 0; np < nparts; ++np) {ierr = PetscSectionSetDof(partSection, np, numVertices/nparts);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  ierr = PetscCalloc1(nparts, &offsets);CHKERRQ(ierr);
  for (np = 0; np < nparts; ++np) {ierr = PetscSectionGetOffset(partSection, np, &offsets[np]);CHKERRQ(ierr);}
  if (!numVertices) pcells[0] = pcells[1] = pcells[2] = 0;
  /* Compute partition */
  ierr = PetscMalloc1(numVertices, &cellproc);CHKERRQ(ierr);
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
  for (np = 1; np < nparts; ++np) if (offsets[np] - offsets[np-1] != numVertices/nparts) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Offset %D != %D partition size", offsets[np], numVertices/nparts);
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numVertices, cellproc, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Simple(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  MPI_Comm       comm;
  PetscInt       np, *tpwgts = NULL, sumw = 0, numVerticesGlobal  = 0;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (p->useGrid) {
    ierr = PetscPartitionerPartition_Simple_Grid(part, nparts, numVertices, start, adjacency, vertSection, targetSection, partSection, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (vertSection) {ierr = PetscInfo(part,"PETSCPARTITIONERSIMPLE ignores vertex weights\n");CHKERRQ(ierr);}
  ierr = PetscObjectGetComm((PetscObject) part, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (targetSection) {
    ierr = MPIU_Allreduce(&numVertices, &numVerticesGlobal, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
    ierr = PetscCalloc1(nparts,&tpwgts);CHKERRQ(ierr);
    for (np = 0; np < nparts; ++np) {
      ierr = PetscSectionGetDof(targetSection,np,&tpwgts[np]);CHKERRQ(ierr);
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
    if (!sumw) {ierr = PetscFree(tpwgts);CHKERRQ(ierr);}
  }

  ierr = ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition);CHKERRQ(ierr);
  if (size == 1) {
    if (tpwgts) {
      for (np = 0; np < nparts; ++np) {
        ierr = PetscSectionSetDof(partSection, np, tpwgts[np]);CHKERRQ(ierr);
      }
    } else {
      for (np = 0; np < nparts; ++np) {
        ierr = PetscSectionSetDof(partSection, np, numVertices/nparts + ((numVertices % nparts) > np));CHKERRQ(ierr);
      }
    }
  } else {
    if (tpwgts) {
      Vec         v;
      PetscScalar *array;
      PetscInt    st,j;
      PetscMPIInt rank;

      ierr = VecCreate(comm,&v);CHKERRQ(ierr);
      ierr = VecSetSizes(v,numVertices,numVerticesGlobal);CHKERRQ(ierr);
      ierr = VecSetType(v,VECSTANDARD);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      for (np = 0,st = 0; np < nparts; ++np) {
        if (rank == np || (rank == size-1 && size < nparts && np >= size)) {
          for (j = 0; j < tpwgts[np]; j++) {
            ierr = VecSetValue(v,st+j,np,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        st += tpwgts[np];
      }
      ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
      ierr = VecGetArray(v,&array);CHKERRQ(ierr);
      for (j = 0; j < numVertices; ++j) {
        ierr = PetscSectionAddDof(partSection,PetscRealPart(array[j]),1);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
      ierr = VecDestroy(&v);CHKERRQ(ierr);
    } else {
      PetscMPIInt rank;
      PetscInt nvGlobal, *offsets, myFirst, myLast;

      ierr = PetscMalloc1(size+1,&offsets);CHKERRQ(ierr);
      offsets[0] = 0;
      ierr = MPI_Allgather(&numVertices,1,MPIU_INT,&offsets[1],1,MPIU_INT,comm);CHKERRQ(ierr);
      for (np = 2; np <= size; np++) {
        offsets[np] += offsets[np-1];
      }
      nvGlobal = offsets[size];
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      myFirst = offsets[rank];
      myLast  = offsets[rank + 1] - 1;
      ierr = PetscFree(offsets);CHKERRQ(ierr);
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
          ierr = PetscSectionSetDof(partSection,np,PartEnd-PartStart);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree(tpwgts);CHKERRQ(ierr);
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
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  p->gridDim = -1;
  part->data = p;

  ierr = PetscPartitionerInitialize_Simple(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
