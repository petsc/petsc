#include <petscvec.h>
#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

typedef struct {
  PetscInt dummy;
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

static PetscErrorCode PetscPartitionerPartition_Simple(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  MPI_Comm       comm;
  PetscInt       np, *tpwgts = NULL, sumw = 0, numVerticesGlobal  = 0;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vertSection) { ierr = PetscInfo(part,"PETSCPARTITIONERSIMPLE ignores vertex weights\n");CHKERRQ(ierr); }
  comm = PetscObjectComm((PetscObject)part);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (targetSection) {
    ierr = MPIU_Allreduce(&numVertices, &numVerticesGlobal, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
    ierr = PetscCalloc1(nparts,&tpwgts);CHKERRQ(ierr);
    for (np = 0; np < nparts; ++np) {
      ierr = PetscSectionGetDof(targetSection,np,&tpwgts[np]);CHKERRQ(ierr);
      sumw += tpwgts[np];
    }
    if (!sumw) {
      ierr = PetscFree(tpwgts);CHKERRQ(ierr);
    } else {
      PetscInt m,mp;
      for (np = 0; np < nparts; ++np) tpwgts[np] = (tpwgts[np]*numVerticesGlobal)/sumw;
      for (np = 0, m = -1, mp = 0, sumw = 0; np < nparts; ++np) {
        if (m < tpwgts[np]) { m = tpwgts[np]; mp = np; }
        sumw += tpwgts[np];
      }
      if (sumw != numVerticesGlobal) tpwgts[mp] += numVerticesGlobal - sumw;
    }
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
  part->noGraph        = PETSC_TRUE;
  part->ops->view      = PetscPartitionerView_Simple;
  part->ops->destroy   = PetscPartitionerDestroy_Simple;
  part->ops->partition = PetscPartitionerPartition_Simple;
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
  part->data = p;

  ierr = PetscPartitionerInitialize_Simple(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
