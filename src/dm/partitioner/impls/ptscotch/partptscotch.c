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

#define CHKERRPTSCOTCH(ierr) do { if (ierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling PT-Scotch library"); } while (0)

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsDeprecatedNoObject("-petscpartititoner_ptscotch_vertex_weight",NULL,"3.13","Use -petscpartitioner_use_vertex_weights");CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL);CHKERRQ(ierr);
    if (!flg) velotab = NULL;
  }
  ierr = SCOTCH_graphInit(&grafdat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_graphBuild(&grafdat, 0, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, adjncy, edlotab);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratInit(&stradat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratGraphMapBuild(&stradat, flagval, nparts, kbalval);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_archInit(&archdat);CHKERRPTSCOTCH(ierr);
  if (tpart) {
    ierr = SCOTCH_archCmpltw(&archdat, nparts, tpart);CHKERRPTSCOTCH(ierr);
  } else {
    ierr = SCOTCH_archCmplt(&archdat, nparts);CHKERRPTSCOTCH(ierr);
  }
  ierr = SCOTCH_graphMap(&grafdat, &archdat, &stradat, part);CHKERRPTSCOTCH(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsDeprecatedNoObject("-petscpartititoner_ptscotch_vertex_weight",NULL,"3.13","Use -petscpartitioner_use_vertex_weights");CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL);CHKERRQ(ierr);
    if (!flg) veloloctab = NULL;
  }
  ierr = MPI_Comm_size(comm, &procglbnbr);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &proclocnum);CHKERRMPI(ierr);
  vertlocnbr = vtxdist[proclocnum + 1] - vtxdist[proclocnum];
  edgelocnbr = xadj[vertlocnbr];

  ierr = SCOTCH_dgraphInit(&grafdat, comm);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_dgraphBuild(&grafdat, 0, vertlocnbr, vertlocnbr, xadj, xadj + 1, veloloctab, NULL, edgelocnbr, edgelocnbr, adjncy, NULL, edloloctab);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratInit(&stradat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratDgraphMapBuild(&stradat, flagval, procglbnbr, nparts, kbalval);CHKERRQ(ierr);
  ierr = SCOTCH_archInit(&archdat);CHKERRPTSCOTCH(ierr);
  if (tpart) { /* target partition weights */
    ierr = SCOTCH_archCmpltw(&archdat, nparts, tpart);CHKERRPTSCOTCH(ierr);
  } else {
    ierr = SCOTCH_archCmplt(&archdat, nparts);CHKERRPTSCOTCH(ierr);
  }
  ierr = SCOTCH_dgraphMapInit(&grafdat, &mappdat, &archdat, part);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_dgraphMapCompute(&grafdat, &mappdat, &stradat);CHKERRPTSCOTCH(ierr);
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
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_free(&p->pcomm);CHKERRMPI(ierr);
  ierr = PetscFree(part->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"using partitioning strategy %s\n",PTScotchStrategyList[p->strategy]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"using load imbalance ratio %g\n",(double)p->imbalance);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_PTScotch_ASCII(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_PTScotch(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  const char *const         *slist = PTScotchStrategyList;
  PetscInt                  nlist = (PetscInt)(sizeof(PTScotchStrategyList)/sizeof(PTScotchStrategyList[0]));
  PetscBool                 flag;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner PTScotch Options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-petscpartitioner_ptscotch_strategy","Partitioning strategy","",slist,nlist,slist[p->strategy],&p->strategy,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-petscpartitioner_ptscotch_imbalance","Load imbalance ratio","",p->imbalance,&p->imbalance,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_PTScotch(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_PTSCOTCH)
  MPI_Comm       comm;
  PetscInt       nvtxs = numVertices;   /* The number of vertices in full graph */
  PetscInt       *vtxdist;              /* Distribution of vertices across processes */
  PetscInt       *xadj   = start;       /* Start of edge list for each vertex */
  PetscInt       *adjncy = adjacency;   /* Edge lists for all vertices */
  PetscInt       *vwgt   = NULL;        /* Vertex weights */
  PetscInt       *adjwgt = NULL;        /* Edge weights */
  PetscInt       v, i, *assignment, *points;
  PetscMPIInt    size, rank, p;
  PetscBool      hasempty = PETSC_FALSE;
  PetscInt       *tpwgts = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)part,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscMalloc2(size+1,&vtxdist,PetscMax(nvtxs,1),&assignment);CHKERRQ(ierr);
  /* Calculate vertex distribution */
  vtxdist[0] = 0;
  ierr = MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm);CHKERRMPI(ierr);
  for (p = 2; p <= size; ++p) {
    hasempty = (PetscBool)(hasempty || !vtxdist[p-1] || !vtxdist[p]);
    vtxdist[p] += vtxdist[p-1];
  }
  /* null graph */
  if (vtxdist[size] == 0) {
    ierr = PetscFree2(vtxdist, assignment);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Calculate vertex weights */
  if (vertSection) {
    ierr = PetscMalloc1(nvtxs,&vwgt);CHKERRQ(ierr);
    for (v = 0; v < nvtxs; ++v) {
      ierr = PetscSectionGetDof(vertSection, v, &vwgt[v]);CHKERRQ(ierr);
    }
  }

  /* Calculate partition weights */
  if (targetSection) {
    PetscInt sumw;

    ierr = PetscCalloc1(nparts,&tpwgts);CHKERRQ(ierr);
    for (p = 0, sumw = 0; p < nparts; ++p) {
      ierr = PetscSectionGetDof(targetSection,p,&tpwgts[p]);CHKERRQ(ierr);
      sumw += tpwgts[p];
    }
    if (!sumw) {ierr = PetscFree(tpwgts);CHKERRQ(ierr);}
  }

  {
    PetscPartitioner_PTScotch *pts = (PetscPartitioner_PTScotch *) part->data;
    int                       strat = PTScotch_Strategy(pts->strategy);
    double                    imbal = (double)pts->imbalance;

    for (p = 0; !vtxdist[p+1] && p < size; ++p);
    if (vtxdist[p+1] == vtxdist[size]) {
      if (rank == p) {
        ierr = PTScotch_PartGraph_Seq(strat, imbal, nvtxs, xadj, adjncy, vwgt, adjwgt, nparts, tpwgts, assignment);CHKERRQ(ierr);
      }
    } else {
      MPI_Comm pcomm = pts->pcomm;

      if (hasempty) {
        PetscInt cnt;

        ierr = MPI_Comm_split(pts->pcomm,!!nvtxs,rank,&pcomm);CHKERRMPI(ierr);
        for (p=0,cnt=0;p<size;p++) {
          if (vtxdist[p+1] != vtxdist[p]) {
            vtxdist[cnt+1] = vtxdist[p+1];
            cnt++;
          }
        }
      };
      if (nvtxs) {
        ierr = PTScotch_PartGraph_MPI(strat, imbal, vtxdist, xadj, adjncy, vwgt, adjwgt, nparts, tpwgts, assignment, pcomm);CHKERRQ(ierr);
      }
      if (hasempty) {
        ierr = MPI_Comm_free(&pcomm);CHKERRMPI(ierr);
      }
    }
  }
  ierr = PetscFree(vwgt);CHKERRQ(ierr);
  ierr = PetscFree(tpwgts);CHKERRQ(ierr);

  /* Convert to PetscSection+IS */
  for (v = 0; v < nvtxs; ++v) {ierr = PetscSectionAddDof(partSection, assignment[v], 1);CHKERRQ(ierr);}
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);

  ierr = PetscFree2(vtxdist,assignment);CHKERRQ(ierr);
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
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)part),&p->pcomm);CHKERRMPI(ierr);
  p->strategy  = 0;
  p->imbalance = 0.01;

  ierr = PetscPartitionerInitialize_PTScotch(part);CHKERRQ(ierr);
  ierr = PetscCitationsRegister(PTScotchPartitionerCitation, &PTScotchPartitionerCite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
