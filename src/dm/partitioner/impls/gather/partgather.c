#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Gather;

static PetscErrorCode PetscPartitionerDestroy_Gather(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_Gather_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_Gather(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscPartitionerView_Gather_ASCII(part, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerPartition_Gather(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscInt np;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition));
  PetscCall(PetscSectionSetDof(partSection, 0, numVertices));
  for (np = 1; np < nparts; ++np) PetscCall(PetscSectionSetDof(partSection, np, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerInitialize_Gather(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph        = PETSC_TRUE;
  part->ops->view      = PetscPartitionerView_Gather;
  part->ops->destroy   = PetscPartitionerDestroy_Gather;
  part->ops->partition = PetscPartitionerPartition_Gather;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCPARTITIONERGATHER = "gather" - A PetscPartitioner object

  Level: intermediate

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Gather(PetscPartitioner part)
{
  PetscPartitioner_Gather *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNew(&p));
  part->data = p;

  PetscCall(PetscPartitionerInitialize_Gather(part));
  PetscFunctionReturn(PETSC_SUCCESS);
}
