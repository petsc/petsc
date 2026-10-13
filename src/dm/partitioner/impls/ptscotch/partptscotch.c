#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/
#include <petsc/private/matimpl.h>

#if PetscDefined(HAVE_PTSCOTCH)
EXTERN_C_BEGIN
  #include <ptscotch.h>
EXTERN_C_END
#endif

PetscBool  PTScotchPartitionerCite       = PETSC_FALSE;
const char PTScotchPartitionerCitation[] = "@article{PTSCOTCH,\n"
                                           "  author  = {C. Chevalier and F. Pellegrini},\n"
                                           "  title   = {{PT-SCOTCH}: a tool for efficient parallel graph ordering},\n"
                                           "  journal = {Parallel Computing},\n"
                                           "  volume  = {34},\n"
                                           "  number  = {6},\n"
                                           "  pages   = {318--331},\n"
                                           "  year    = {2008},\n"
                                           "  doi     = {https://doi.org/10.1016/j.parco.2007.12.001}\n"
                                           "}\n";

typedef struct {
  MPI_Comm  pcomm;
  PetscInt  strategy;
  PetscReal imbalance;
} PetscPartitioner_PTScotch;

#if PetscDefined(HAVE_PTSCOTCH)

static int PTScotch_Strategy(PetscInt strategy)
{
  switch (strategy) {
  case 0:
    return SCOTCH_STRATDEFAULT;
  case 1:
    return SCOTCH_STRATQUALITY;
  case 2:
    return SCOTCH_STRATSPEED;
  case 3:
    return SCOTCH_STRATBALANCE;
  case 4:
    return SCOTCH_STRATSAFETY;
  case 5:
    return SCOTCH_STRATSCALABILITY;
  case 6:
    return SCOTCH_STRATRECURSIVE;
  case 7:
    return SCOTCH_STRATREMAP;
  default:
    return SCOTCH_STRATDEFAULT;
  }
}

#endif /* PETSC_HAVE_PTSCOTCH */

static const char *const PTScotchStrategyList[] = {"DEFAULT", "QUALITY", "SPEED", "BALANCE", "SAFETY", "SCALABILITY", "RECURSIVE", "REMAP"};

static PetscErrorCode PetscPartitionerDestroy_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *)part->data;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_free(&p->pcomm));
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_PTScotch_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *)part->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "using partitioning strategy %s\n", PTScotchStrategyList[p->strategy]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "using load imbalance ratio %g\n", (double)p->imbalance));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_PTScotch(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscPartitionerView_PTScotch_ASCII(part, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerSetFromOptions_PTScotch(PetscPartitioner part, PetscOptionItems PetscOptionsObject)
{
  PetscPartitioner_PTScotch *p     = (PetscPartitioner_PTScotch *)part->data;
  const char *const         *slist = PTScotchStrategyList;
  PetscInt                   nlist = PETSC_STATIC_ARRAY_LENGTH(PTScotchStrategyList);
  PetscBool                  flag;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscPartitioner PTScotch Options");
  PetscCall(PetscOptionsEList("-petscpartitioner_ptscotch_strategy", "Partitioning strategy", "", slist, nlist, slist[p->strategy], &p->strategy, &flag));
  PetscCall(PetscOptionsReal("-petscpartitioner_ptscotch_imbalance", "Load imbalance ratio", "", p->imbalance, &p->imbalance, &flag));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerPartition_PTScotch(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection edgeSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
#if PetscDefined(HAVE_PTSCOTCH)
  MPI_Comm    comm;
  PetscInt    nvtxs = numVertices; /* The number of vertices in full graph */
  PetscInt   *vtxdist;             /* Distribution of vertices across processes */
  PetscInt   *xadj   = start;      /* Start of edge list for each vertex */
  PetscInt   *adjncy = adjacency;  /* Edge lists for all vertices */
  PetscInt   *vwgt   = NULL;       /* Vertex weights */
  PetscInt   *adjwgt = NULL;       /* Edge weights */
  PetscInt    v, i, *assignment, *points;
  PetscMPIInt size, p;
  PetscInt   *tpwgts = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscMalloc2(size + 1, &vtxdist, PetscMax(nvtxs, 1), &assignment));
  /* Calculate vertex distribution */
  vtxdist[0] = 0;
  PetscCallMPI(MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm));
  for (p = 2; p <= size; ++p) vtxdist[p] += vtxdist[p - 1];
  /* null graph */
  if (vtxdist[size] == 0) {
    PetscCall(PetscFree2(vtxdist, assignment));
    PetscCall(ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Calculate vertex weights */
  if (vertSection) {
    PetscCall(PetscMalloc1(nvtxs, &vwgt));
    for (v = 0; v < nvtxs; ++v) PetscCall(PetscSectionGetDof(vertSection, v, &vwgt[v]));
  }
  // Weight edges
  if (edgeSection) {
    PetscCall(PetscMalloc1(xadj[nvtxs], &adjwgt));
    for (PetscInt e = 0; e < xadj[nvtxs]; ++e) PetscCall(PetscSectionGetDof(edgeSection, e, &adjwgt[e]));
  }

  /* Calculate partition weights */
  if (targetSection) {
    PetscInt sumw;

    PetscCall(PetscCalloc1(nparts, &tpwgts));
    for (p = 0, sumw = 0; p < nparts; ++p) {
      PetscCall(PetscSectionGetDof(targetSection, p, &tpwgts[p]));
      sumw += tpwgts[p];
    }
    if (!sumw) PetscCall(PetscFree(tpwgts));
  }

  {
    PetscPartitioner_PTScotch *pts     = (PetscPartitioner_PTScotch *)part->data;
    PetscBool                  usevwgt = PETSC_TRUE;
    int                        strat   = PTScotch_Strategy(pts->strategy);

    PetscCall(PetscOptionsDeprecatedNoObject(comm, NULL, "-petscpartititoner_ptscotch_vertex_weight", "-petscpartitioner_use_vertex_weights", "3.13", NULL));
    /*
       Cannot remove the PetscOptionsGetBool() below since the PetscOptionsDeprecatedNoObject() above is called after the non-deprecated version
       has already been checked in PetscPartitionerSetFromOptions().
    */
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_use_vertex_weight", &usevwgt, NULL));
    /* edgeSection is the same on every process, so it reports edge weights where adjwgt cannot:
       a process owning no edges allocates nothing and passes a null array. */
    PetscCall(PetscPTSCOTCHPartitionGraph_Private(pts->pcomm, vtxdist, xadj, adjncy, usevwgt ? vwgt : NULL, adjwgt, (PetscBool)(edgeSection != NULL), nparts, tpwgts, strat, (double)pts->imbalance, assignment));
  }
  PetscCall(PetscFree(vwgt));
  PetscCall(PetscFree(adjwgt));
  PetscCall(PetscFree(tpwgts));

  /* Convert to PetscSection+IS */
  for (v = 0; v < nvtxs; ++v) PetscCall(PetscSectionAddDof(partSection, assignment[v], 1));
  PetscCall(PetscMalloc1(nvtxs, &points));
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  PetscCheck(i == nvtxs, comm, PETSC_ERR_PLIB, "Number of points %" PetscInt_FMT " should be %" PetscInt_FMT, i, nvtxs);
  PetscCall(ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition));

  PetscCall(PetscFree2(vtxdist, assignment));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(PetscObjectComm((PetscObject)part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-ptscotch.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_PTScotch(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph             = PETSC_FALSE;
  part->ops->view           = PetscPartitionerView_PTScotch;
  part->ops->destroy        = PetscPartitionerDestroy_PTScotch;
  part->ops->partition      = PetscPartitionerPartition_PTScotch;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_PTScotch;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCPARTITIONERPTSCOTCH = "ptscotch" - A PetscPartitioner object using the PT-Scotch library

  Level: intermediate

  Options Database Keys:
+  -petscpartitioner_ptscotch_strategy (default|quality|speed|balance|safety|scalability|recursive|remap) - PT-SCOTCH strategy.
-  -petscpartitioner_ptscotch_imbalance ratio                                                             - Load imbalance ratio

  Notes: when the graph is on a single process, this partitioner actually uses Scotch and not PT-Scotch

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNew(&p));
  part->data = p;

  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)part), &p->pcomm));
  p->strategy  = 0;
  p->imbalance = 0.01;

  PetscCall(PetscPartitionerInitialize_PTScotch(part));
  PetscCall(PetscCitationsRegister(PTScotchPartitionerCitation, &PTScotchPartitionerCite));
  PetscFunctionReturn(PETSC_SUCCESS);
}
