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

static PetscErrorCode PetscPartitionerPartition_MatPartitioning(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *is)
{
  PetscPartitioner_MatPartitioning  *p = (PetscPartitioner_MatPartitioning *) part->data;
  Mat                               matadj;
  IS                                is1, is2, is3;
  PetscReal                         *tpwgts = NULL;
  PetscInt                          numVerticesGlobal, numEdges;
  PetscInt                          *i, *j, *vwgt = NULL;
  MPI_Comm                          comm;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part, &comm);CHKERRQ(ierr);

  /* TODO: MatCreateMPIAdj should maybe take global number of ROWS */
  /* TODO: And vertex distribution in PetscPartitionerPartition_ParMetis should be done using PetscSplitOwnership */
  numVerticesGlobal = PETSC_DECIDE;
  ierr = PetscSplitOwnership(comm, &numVertices, &numVerticesGlobal);CHKERRQ(ierr);

  /* copy arrays to avoid memory errors because MatMPIAdjSetPreallocation copies just pointers */
  numEdges = start[numVertices];
  ierr = PetscMalloc1(numVertices+1, &i);CHKERRQ(ierr);
  ierr = PetscMalloc1(numEdges, &j);CHKERRQ(ierr);
  ierr = PetscArraycpy(i, start, numVertices+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(j, adjacency, numEdges);CHKERRQ(ierr);

  /* construct the adjacency matrix */
  ierr = MatCreateMPIAdj(comm, numVertices, numVerticesGlobal, i, j, NULL, &matadj);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(p->mp, matadj);CHKERRQ(ierr);
  ierr = MatPartitioningSetNParts(p->mp, nparts);CHKERRQ(ierr);

  /* calculate partition weights */
  if (targetSection) {
    PetscReal sumt;
    PetscInt  p;

    sumt = 0.0;
    ierr = PetscMalloc1(nparts,&tpwgts);CHKERRQ(ierr);
    for (p = 0; p < nparts; ++p) {
      PetscInt tpd;

      ierr = PetscSectionGetDof(targetSection,p,&tpd);CHKERRQ(ierr);
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
      ierr = PetscFree(tpwgts);CHKERRQ(ierr);
    }
  }
  ierr = MatPartitioningSetPartitionWeights(p->mp, tpwgts);CHKERRQ(ierr);

  /* calculate vertex weights */
  if (vertSection) {
    PetscInt v;

    ierr = PetscMalloc1(numVertices,&vwgt);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      ierr = PetscSectionGetDof(vertSection, v, &vwgt[v]);CHKERRQ(ierr);
    }
  }
  ierr = MatPartitioningSetVertexWeights(p->mp, vwgt);CHKERRQ(ierr);

  /* apply the partitioning */
  ierr = MatPartitioningApply(p->mp, &is1);CHKERRQ(ierr);

  /* construct the PetscSection */
  {
    PetscInt v;
    const PetscInt *assignment_arr;

    ierr = ISGetIndices(is1, &assignment_arr);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {ierr = PetscSectionAddDof(partSection, assignment_arr[v], 1);CHKERRQ(ierr);}
    ierr = ISRestoreIndices(is1, &assignment_arr);CHKERRQ(ierr);
  }

  /* convert assignment IS to global numbering IS */
  ierr = ISPartitioningToNumbering(is1, &is2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);

  /* renumber IS into local numbering */
  ierr = ISOnComm(is2, PETSC_COMM_SELF, PETSC_USE_POINTER, &is1);CHKERRQ(ierr);
  ierr = ISRenumber(is1, NULL, NULL, &is3);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  /* invert IS */
  ierr = ISSetPermutation(is3);CHKERRQ(ierr);
  ierr = ISInvertPermutation(is3, numVertices, &is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is3);CHKERRQ(ierr);

  ierr = MatDestroy(&matadj);CHKERRQ(ierr);
  *is = is1;
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

