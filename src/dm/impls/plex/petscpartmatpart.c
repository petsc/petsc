#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

typedef struct {
  MatPartitioning mp;
} PetscPartitioner_MatPartitioning;

static PetscErrorCode PetscPartitionerMatPartitioningGetMatPartitioning_MatPartitioning(PetscPartitioner part, MatPartitioning *mp)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;

  PetscFunctionBegin;
  *mp = p->mp;
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerMatPartitioningGetMatPartitioning - Get a MatPartitioning instance wrapped by this PetscPartitioner.

  Not Collective

  Input Parameters:
. part     - The PetscPartitioner

  Output Parameters:
. mp       - The MatPartitioning

  Level: developer

.seealso PetscPartitionerMatPartitioningSetMatPartitioning(), DMPlexDistribute(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerMatPartitioningGetMatPartitioning(PetscPartitioner part, MatPartitioning *mp)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidPointer(mp, 2);
  ierr = PetscUseMethod(part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",(PetscPartitioner,MatPartitioning*),(part,mp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_MatPartitioning(PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  ierr = MatPartitioningDestroy(&p->mp);CHKERRQ(ierr);
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  PetscViewerFormat                 format;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "MatPartitioning Graph Partitioner:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (p->mp) {ierr = MatPartitioningView(p->mp, viewer);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_MatPartitioning_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_MatPartitioning(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)p->mp,((PetscObject)part)->prefix);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(p->mp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_MatPartitioning(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
  /*PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;*/
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  /* call MatPartitioningSetAdjacency, MatPartitioningApply, ISPartitioningToSectionAndIS */
  *partition = NULL;
  ierr = PetscSectionReset(partSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_MatPartitioning(PetscPartitioner part)
{
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  part->ops->view           = PetscPartitionerView_MatPartitioning;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_MatPartitioning;
  part->ops->destroy        = PetscPartitionerDestroy_MatPartitioning;
  part->ops->partition      = PetscPartitionerPartition_MatPartitioning;
  ierr = PetscObjectComposeFunction((PetscObject)part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",PetscPartitionerMatPartitioningGetMatPartitioning_MatPartitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERMATPARTITIONING = "matpartitioning" - A PetscPartitioner object

  Level: developer

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_MatPartitioning(PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;
  ierr = PetscPartitionerInitialize_MatPartitioning(part);CHKERRQ(ierr);
  ierr = MatPartitioningCreate(PetscObjectComm((PetscObject)part), &p->mp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

