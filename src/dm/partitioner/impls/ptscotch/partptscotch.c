#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

#if defined(PETSC_HAVE_PTSCOTCH)
EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END
#endif

PetscBool  PTScotchPartitionerCite = PETSC_FALSE;
const char PTScotchPartitionerCitation[] =
  "@article{PTSCOTCH,\n"
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

#if defined(PETSC_HAVE_PTSCOTCH)

#define CHKERRPTSCOTCH(...) do { PetscCheck(!(__VA_ARGS__),PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling PT-Scotch library"); } while (0)

static int PTScotch_Strategy(PetscInt strategy)
{
  switch (strategy) {
  case  0: return SCOTCH_STRATDEFAULT;
  case  1: return SCOTCH_STRATQUALITY;
  case  2: return SCOTCH_STRATSPEED;
  case  3: return SCOTCH_STRATBALANCE;
  case  4: return SCOTCH_STRATSAFETY;
  case  5: return SCOTCH_STRATSCALABILITY;
  case  6: return SCOTCH_STRATRECURSIVE;
  case  7: return SCOTCH_STRATREMAP;
  default: return SCOTCH_STRATDEFAULT;
  }
}

static PetscErrorCode PTScotch_PartGraph_Seq(SCOTCH_Num strategy, double imbalance, SCOTCH_Num n, SCOTCH_Num xadj[], SCOTCH_Num adjncy[],
                                             SCOTCH_Num vtxwgt[], SCOTCH_Num adjwgt[], SCOTCH_Num nparts, SCOTCH_Num tpart[], SCOTCH_Num part[])
{
  SCOTCH_Arch    archdat;
  SCOTCH_Graph   grafdat;
  SCOTCH_Strat   stradat;
  SCOTCH_Num     vertnbr = n;
  SCOTCH_Num     edgenbr = xadj[n];
  SCOTCH_Num*    velotab = vtxwgt;
  SCOTCH_Num*    edlotab = adjwgt;
  SCOTCH_Num     flagval = strategy;
  double         kbalval = imbalance;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    CHKERRQ(PetscOptionsDeprecatedNoObject("-petscpartititoner_ptscotch_vertex_weight",NULL,"3.13","Use -petscpartitioner_use_vertex_weights"));
    CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL));
    if (!flg) velotab = NULL;
  }
  CHKERRPTSCOTCH(SCOTCH_graphInit(&grafdat));
  CHKERRPTSCOTCH(SCOTCH_graphBuild(&grafdat, 0, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, adjncy, edlotab));
  CHKERRPTSCOTCH(SCOTCH_stratInit(&stradat));
  CHKERRPTSCOTCH(SCOTCH_stratGraphMapBuild(&stradat, flagval, nparts, kbalval));
  CHKERRPTSCOTCH(SCOTCH_archInit(&archdat));
  if (tpart) {
    CHKERRPTSCOTCH(SCOTCH_archCmpltw(&archdat, nparts, tpart));
  } else {
    CHKERRPTSCOTCH(SCOTCH_archCmplt(&archdat, nparts));
  }
  CHKERRPTSCOTCH(SCOTCH_graphMap(&grafdat, &archdat, &stradat, part));
  SCOTCH_archExit(&archdat);
  SCOTCH_stratExit(&stradat);
  SCOTCH_graphExit(&grafdat);
  PetscFunctionReturn(0);
}

static PetscErrorCode PTScotch_PartGraph_MPI(SCOTCH_Num strategy, double imbalance, SCOTCH_Num vtxdist[], SCOTCH_Num xadj[], SCOTCH_Num adjncy[],
                                             SCOTCH_Num vtxwgt[], SCOTCH_Num adjwgt[], SCOTCH_Num nparts, SCOTCH_Num tpart[], SCOTCH_Num part[], MPI_Comm comm)
{
  PetscMPIInt     procglbnbr;
  PetscMPIInt     proclocnum;
  SCOTCH_Arch     archdat;
  SCOTCH_Dgraph   grafdat;
  SCOTCH_Dmapping mappdat;
  SCOTCH_Strat    stradat;
  SCOTCH_Num      vertlocnbr;
  SCOTCH_Num      edgelocnbr;
  SCOTCH_Num*     veloloctab = vtxwgt;
  SCOTCH_Num*     edloloctab = adjwgt;
  SCOTCH_Num      flagval = strategy;
  double          kbalval = imbalance;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    CHKERRQ(PetscOptionsDeprecatedNoObject("-petscpartititoner_ptscotch_vertex_weight",NULL,"3.13","Use -petscpartitioner_use_vertex_weights"));
    CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL));
    if (!flg) veloloctab = NULL;
  }
  CHKERRMPI(MPI_Comm_size(comm, &procglbnbr));
  CHKERRMPI(MPI_Comm_rank(comm, &proclocnum));
  vertlocnbr = vtxdist[proclocnum + 1] - vtxdist[proclocnum];
  edgelocnbr = xadj[vertlocnbr];

  CHKERRPTSCOTCH(SCOTCH_dgraphInit(&grafdat, comm));
  CHKERRPTSCOTCH(SCOTCH_dgraphBuild(&grafdat, 0, vertlocnbr, vertlocnbr, xadj, xadj + 1, veloloctab, NULL, edgelocnbr, edgelocnbr, adjncy, NULL, edloloctab));
  CHKERRPTSCOTCH(SCOTCH_stratInit(&stradat));
  CHKERRQ(SCOTCH_stratDgraphMapBuild(&stradat, flagval, procglbnbr, nparts, kbalval));
  CHKERRPTSCOTCH(SCOTCH_archInit(&archdat));
  if (tpart) { /* target partition weights */
    CHKERRPTSCOTCH(SCOTCH_archCmpltw(&archdat, nparts, tpart));
  } else {
    CHKERRPTSCOTCH(SCOTCH_archCmplt(&archdat, nparts));
  }
  CHKERRPTSCOTCH(SCOTCH_dgraphMapInit(&grafdat, &mappdat, &archdat, part));
  CHKERRPTSCOTCH(SCOTCH_dgraphMapCompute(&grafdat, &mappdat, &stradat));
  SCOTCH_dgraphMapExit(&grafdat, &mappdat);
  SCOTCH_archExit(&archdat);
  SCOTCH_stratExit(&stradat);
  SCOTCH_dgraphExit(&grafdat);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_PTSCOTCH */

static const char *const
PTScotchStrategyList[] = {
  "DEFAULT",
  "QUALITY",
  "SPEED",
  "BALANCE",
  "SAFETY",
  "SCALABILITY",
  "RECURSIVE",
  "REMAP"
};

static PetscErrorCode PetscPartitionerDestroy_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_free(&p->pcomm));
  CHKERRQ(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"using partitioning strategy %s\n",PTScotchStrategyList[p->strategy]));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"using load imbalance ratio %g\n",(double)p->imbalance));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscPartitionerView_PTScotch_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_PTScotch(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  const char *const         *slist = PTScotchStrategyList;
  PetscInt                  nlist = (PetscInt)(sizeof(PTScotchStrategyList)/sizeof(PTScotchStrategyList[0]));
  PetscBool                 flag;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject, "PetscPartitioner PTScotch Options"));
  CHKERRQ(PetscOptionsEList("-petscpartitioner_ptscotch_strategy","Partitioning strategy","",slist,nlist,slist[p->strategy],&p->strategy,&flag));
  CHKERRQ(PetscOptionsReal("-petscpartitioner_ptscotch_imbalance","Load imbalance ratio","",p->imbalance,&p->imbalance,&flag));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_PTScotch(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_PTSCOTCH)
  MPI_Comm     comm;
  PetscInt     nvtxs    = numVertices; /* The number of vertices in full graph */
  PetscInt    *vtxdist;         /* Distribution of vertices across processes */
  PetscInt    *xadj     = start; /* Start of edge list for each vertex */
  PetscInt    *adjncy   = adjacency; /* Edge lists for all vertices */
  PetscInt    *vwgt     = NULL; /* Vertex weights */
  PetscInt    *adjwgt   = NULL; /* Edge weights */
  PetscInt     v, i, *assignment, *points;
  PetscMPIInt  size, rank, p;
  PetscBool    hasempty = PETSC_FALSE;
  PetscInt     *tpwgts  = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)part,&comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscMalloc2(size+1,&vtxdist,PetscMax(nvtxs,1),&assignment));
  /* Calculate vertex distribution */
  vtxdist[0] = 0;
  CHKERRMPI(MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm));
  for (p = 2; p <= size; ++p) {
    hasempty = (PetscBool)(hasempty || !vtxdist[p-1] || !vtxdist[p]);
    vtxdist[p] += vtxdist[p-1];
  }
  /* null graph */
  if (vtxdist[size] == 0) {
    CHKERRQ(PetscFree2(vtxdist, assignment));
    CHKERRQ(ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(0);
  }

  /* Calculate vertex weights */
  if (vertSection) {
    CHKERRQ(PetscMalloc1(nvtxs,&vwgt));
    for (v = 0; v < nvtxs; ++v) CHKERRQ(PetscSectionGetDof(vertSection, v, &vwgt[v]));
  }

  /* Calculate partition weights */
  if (targetSection) {
    PetscInt sumw;

    CHKERRQ(PetscCalloc1(nparts,&tpwgts));
    for (p = 0, sumw = 0; p < nparts; ++p) {
      CHKERRQ(PetscSectionGetDof(targetSection,p,&tpwgts[p]));
      sumw += tpwgts[p];
    }
    if (!sumw) CHKERRQ(PetscFree(tpwgts));
  }

  {
    PetscPartitioner_PTScotch *pts = (PetscPartitioner_PTScotch *) part->data;
    int                       strat = PTScotch_Strategy(pts->strategy);
    double                    imbal = (double)pts->imbalance;

    for (p = 0; !vtxdist[p+1] && p < size; ++p);
    if (vtxdist[p+1] == vtxdist[size]) {
      if (rank == p) CHKERRQ(PTScotch_PartGraph_Seq(strat, imbal, nvtxs, xadj, adjncy, vwgt, adjwgt, nparts, tpwgts, assignment));
    } else {
      MPI_Comm pcomm = pts->pcomm;

      if (hasempty) {
        PetscInt cnt;

        CHKERRMPI(MPI_Comm_split(pts->pcomm,!!nvtxs,rank,&pcomm));
        for (p=0,cnt=0;p<size;p++) {
          if (vtxdist[p+1] != vtxdist[p]) {
            vtxdist[cnt+1] = vtxdist[p+1];
            cnt++;
          }
        }
      };
      if (nvtxs) CHKERRQ(PTScotch_PartGraph_MPI(strat, imbal, vtxdist, xadj, adjncy, vwgt, adjwgt, nparts, tpwgts, assignment, pcomm));
      if (hasempty) CHKERRMPI(MPI_Comm_free(&pcomm));
    }
  }
  CHKERRQ(PetscFree(vwgt));
  CHKERRQ(PetscFree(tpwgts));

  /* Convert to PetscSection+IS */
  for (v = 0; v < nvtxs; ++v) CHKERRQ(PetscSectionAddDof(partSection, assignment[v], 1));
  CHKERRQ(PetscMalloc1(nvtxs, &points));
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  PetscCheck(i == nvtxs,comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  CHKERRQ(ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition));

  CHKERRQ(PetscFree2(vtxdist,assignment));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-ptscotch.");
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
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERPTSCOTCH = "ptscotch" - A PetscPartitioner object using the PT-Scotch library

  Level: intermediate

  Options Database Keys:
+  -petscpartitioner_ptscotch_strategy <string> - PT-Scotch strategy. Choose one of default quality speed balance safety scalability recursive remap
-  -petscpartitioner_ptscotch_imbalance <val> - Load imbalance ratio

  Notes: when the graph is on a single process, this partitioner actually uses Scotch and not PT-Scotch

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  CHKERRQ(PetscNewLog(part, &p));
  part->data = p;

  CHKERRMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)part),&p->pcomm));
  p->strategy  = 0;
  p->imbalance = 0.01;

  CHKERRQ(PetscPartitionerInitialize_PTScotch(part));
  CHKERRQ(PetscCitationsRegister(PTScotchPartitionerCitation, &PTScotchPartitionerCite));
  PetscFunctionReturn(0);
}
