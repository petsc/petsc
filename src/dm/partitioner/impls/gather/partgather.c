#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Gather;

static PetscErrorCode PetscPartitionerDestroy_Gather(PetscPartitioner part)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Gather_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Gather(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscPartitionerView_Gather_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Gather(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscInt       np;

  PetscFunctionBegin;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition));
  CHKERRQ(PetscSectionSetDof(partSection,0,numVertices));
  for (np = 1; np < nparts; ++np) CHKERRQ(PetscSectionSetDof(partSection, np, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_Gather(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph        = PETSC_TRUE;
  part->ops->view      = PetscPartitionerView_Gather;
  part->ops->destroy   = PetscPartitionerDestroy_Gather;
  part->ops->partition = PetscPartitionerPartition_Gather;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERGATHER = "gather" - A PetscPartitioner object

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Gather(PetscPartitioner part)
{
  PetscPartitioner_Gather *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  CHKERRQ(PetscNewLog(part, &p));
  part->data = p;

  CHKERRQ(PetscPartitionerInitialize_Gather(part));
  PetscFunctionReturn(0);
}
