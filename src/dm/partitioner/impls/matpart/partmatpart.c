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

.seealso `DMPlexDistribute()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerMatPartitioningGetMatPartitioning(PetscPartitioner part, MatPartitioning *mp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidPointer(mp, 2);
  PetscUseMethod(part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",(PetscPartitioner,MatPartitioning*),(part,mp));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_MatPartitioning(PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;

  PetscFunctionBegin;
  PetscCall(MatPartitioningDestroy(&p->mp));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",NULL));
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  PetscViewerFormat                 format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatPartitioning Graph Partitioner:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  if (p->mp) PetscCall(MatPartitioningView(p->mp, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_MatPartitioning(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscPartitionerView_MatPartitioning_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_MatPartitioning(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)p->mp,((PetscObject)part)->prefix));
  PetscCall(MatPartitioningSetFromOptions(p->mp));
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
  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));

  /* TODO: MatCreateMPIAdj should maybe take global number of ROWS */
  /* TODO: And vertex distribution in PetscPartitionerPartition_ParMetis should be done using PetscSplitOwnership */
  numVerticesGlobal = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(comm, &numVertices, &numVerticesGlobal));

  /* copy arrays to avoid memory errors because MatMPIAdjSetPreallocation copies just pointers */
  numEdges = start[numVertices];
  PetscCall(PetscMalloc1(numVertices+1, &i));
  PetscCall(PetscMalloc1(numEdges, &j));
  PetscCall(PetscArraycpy(i, start, numVertices+1));
  PetscCall(PetscArraycpy(j, adjacency, numEdges));

  /* construct the adjacency matrix */
  PetscCall(MatCreateMPIAdj(comm, numVertices, numVerticesGlobal, i, j, NULL, &matadj));
  PetscCall(MatPartitioningSetAdjacency(p->mp, matadj));
  PetscCall(MatPartitioningSetNParts(p->mp, nparts));

  /* calculate partition weights */
  if (targetSection) {
    PetscReal sumt;
    PetscInt  p;

    sumt = 0.0;
    PetscCall(PetscMalloc1(nparts,&tpwgts));
    for (p = 0; p < nparts; ++p) {
      PetscInt tpd;

      PetscCall(PetscSectionGetDof(targetSection,p,&tpd));
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
      PetscCall(PetscFree(tpwgts));
    }
  }
  PetscCall(MatPartitioningSetPartitionWeights(p->mp, tpwgts));

  /* calculate vertex weights */
  if (vertSection) {
    PetscInt v;

    PetscCall(PetscMalloc1(numVertices,&vwgt));
    for (v = 0; v < numVertices; ++v) {
      PetscCall(PetscSectionGetDof(vertSection, v, &vwgt[v]));
    }
  }
  PetscCall(MatPartitioningSetVertexWeights(p->mp, vwgt));

  /* apply the partitioning */
  PetscCall(MatPartitioningApply(p->mp, &is1));

  /* construct the PetscSection */
  {
    PetscInt v;
    const PetscInt *assignment_arr;

    PetscCall(ISGetIndices(is1, &assignment_arr));
    for (v = 0; v < numVertices; ++v) PetscCall(PetscSectionAddDof(partSection, assignment_arr[v], 1));
    PetscCall(ISRestoreIndices(is1, &assignment_arr));
  }

  /* convert assignment IS to global numbering IS */
  PetscCall(ISPartitioningToNumbering(is1, &is2));
  PetscCall(ISDestroy(&is1));

  /* renumber IS into local numbering */
  PetscCall(ISOnComm(is2, PETSC_COMM_SELF, PETSC_USE_POINTER, &is1));
  PetscCall(ISRenumber(is1, NULL, NULL, &is3));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  /* invert IS */
  PetscCall(ISSetPermutation(is3));
  PetscCall(ISInvertPermutation(is3, numVertices, &is1));
  PetscCall(ISDestroy(&is3));

  PetscCall(MatDestroy(&matadj));
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
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"PetscPartitionerMatPartitioningGetMatPartitioning_C",PetscPartitionerMatPartitioningGetMatPartitioning_MatPartitioning));
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERMATPARTITIONING = "matpartitioning" - A PetscPartitioner object

  Level: developer

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_MatPartitioning(PetscPartitioner part)
{
  PetscPartitioner_MatPartitioning  *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNewLog(part, &p));
  part->data = p;
  PetscCall(PetscPartitionerInitialize_MatPartitioning(part));
  PetscCall(MatPartitioningCreate(PetscObjectComm((PetscObject)part), &p->mp));
  PetscFunctionReturn(0);
}
