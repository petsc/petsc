#include <petscmat.h>
#include <petsc/private/partitionerimpl.h>   /*I      "petscpartitioner.h"   I*/

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

.seealso DMPlexDistribute(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerMatPartitioningGetMatPartitioning(PetscPartitioner part, MatPartitioning *mp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidPointer(mp, 2);
  CHKERRQ(PetscUseMethod(part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",(PetscPartitioner,MatPartitioning*),(part,mp)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_MatPartitioning(PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;

  PetscFunctionBegin;
  CHKERRQ(MatPartitioningDestroy(&p->mp));
  CHKERRQ(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  PetscViewerFormat                 format;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "MatPartitioning Graph Partitioner:\n"));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  if (p->mp) CHKERRQ(MatPartitioningView(p->mp, viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscPartitionerView_MatPartitioning_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_MatPartitioning(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)p->mp,((PetscObject)part)->prefix));
  CHKERRQ(MatPartitioningSetFromOptions(p->mp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_MatPartitioning(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *is)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  Mat                               matadj;
  IS                                is1, is2, is3;
  PetscReal                         *tpwgts = NULL;
  PetscInt                          numVerticesGlobal, numEdges;
  PetscInt                          *i, *j, *vwgt = NULL;
  MPI_Comm                          comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)part, &comm));

  /* TODO: MatCreateMPIAdj should maybe take global number of ROWS */
  /* TODO: And vertex distribution in PetscPartitionerPartition_ParMetis should be done using PetscSplitOwnership */
  numVerticesGlobal = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnership(comm, &numVertices, &numVerticesGlobal));

  /* copy arrays to avoid memory errors because MatMPIAdjSetPreallocation copies just pointers */
  numEdges = start[numVertices];
  CHKERRQ(PetscMalloc1(numVertices+1, &i));
  CHKERRQ(PetscMalloc1(numEdges, &j));
  CHKERRQ(PetscArraycpy(i, start, numVertices+1));
  CHKERRQ(PetscArraycpy(j, adjacency, numEdges));

  /* construct the adjacency matrix */
  CHKERRQ(MatCreateMPIAdj(comm, numVertices, numVerticesGlobal, i, j, NULL, &matadj));
  CHKERRQ(MatPartitioningSetAdjacency(p->mp, matadj));
  CHKERRQ(MatPartitioningSetNParts(p->mp, nparts));

  /* calculate partition weights */
  if (targetSection) {
    PetscReal sumt;
    PetscInt  p;

    sumt = 0.0;
    CHKERRQ(PetscMalloc1(nparts,&tpwgts));
    for (p = 0; p < nparts; ++p) {
      PetscInt tpd;

      CHKERRQ(PetscSectionGetDof(targetSection,p,&tpd));
      sumt += tpd;
      tpwgts[p] = tpd;
    }
    if (sumt) { /* METIS/ParMETIS do not like exactly zero weight */
      for (p = 0, sumt = 0.0; p < nparts; ++p) {
        tpwgts[p] = PetscMax(tpwgts[p],PETSC_SMALL);
        sumt += tpwgts[p];
      }
      for (p = 0; p < nparts; ++p) tpwgts[p] /= sumt;
      for (p = 0, sumt = 0.0; p < nparts-1; ++p) sumt += tpwgts[p];
      tpwgts[nparts - 1] = 1. - sumt;
    } else {
      CHKERRQ(PetscFree(tpwgts));
    }
  }
  CHKERRQ(MatPartitioningSetPartitionWeights(p->mp, tpwgts));

  /* calculate vertex weights */
  if (vertSection) {
    PetscInt v;

    CHKERRQ(PetscMalloc1(numVertices,&vwgt));
    for (v = 0; v < numVertices; ++v) {
      CHKERRQ(PetscSectionGetDof(vertSection, v, &vwgt[v]));
    }
  }
  CHKERRQ(MatPartitioningSetVertexWeights(p->mp, vwgt));

  /* apply the partitioning */
  CHKERRQ(MatPartitioningApply(p->mp, &is1));

  /* construct the PetscSection */
  {
    PetscInt v;
    const PetscInt *assignment_arr;

    CHKERRQ(ISGetIndices(is1, &assignment_arr));
    for (v = 0; v < numVertices; ++v) CHKERRQ(PetscSectionAddDof(partSection, assignment_arr[v], 1));
    CHKERRQ(ISRestoreIndices(is1, &assignment_arr));
  }

  /* convert assignment IS to global numbering IS */
  CHKERRQ(ISPartitioningToNumbering(is1, &is2));
  CHKERRQ(ISDestroy(&is1));

  /* renumber IS into local numbering */
  CHKERRQ(ISOnComm(is2, PETSC_COMM_SELF, PETSC_USE_POINTER, &is1));
  CHKERRQ(ISRenumber(is1, NULL, NULL, &is3));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  /* invert IS */
  CHKERRQ(ISSetPermutation(is3));
  CHKERRQ(ISInvertPermutation(is3, numVertices, &is1));
  CHKERRQ(ISDestroy(&is3));

  CHKERRQ(MatDestroy(&matadj));
  *is = is1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_MatPartitioning(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view           = PetscPartitionerView_MatPartitioning;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_MatPartitioning;
  part->ops->destroy        = PetscPartitionerDestroy_MatPartitioning;
  part->ops->partition      = PetscPartitionerPartition_MatPartitioning;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",PetscPartitionerMatPartitioningGetMatPartitioning_MatPartitioning));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  CHKERRQ(PetscNewLog(part, &p));
  part->data = p;
  CHKERRQ(PetscPartitionerInitialize_MatPartitioning(part));
  CHKERRQ(MatPartitioningCreate(PetscObjectComm((PetscObject)part), &p->mp));
  PetscFunctionReturn(0);
}
