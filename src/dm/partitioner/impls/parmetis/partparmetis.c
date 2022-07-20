#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

#if defined(PETSC_HAVE_PARMETIS)
#include <parmetis.h>
#endif

PetscBool  ParMetisPartitionerCite = PETSC_FALSE;
const char ParMetisPartitionerCitation[] =
  "@article{KarypisKumar98,\n"
  "  author  = {George Karypis and Vipin Kumar},\n"
  "  title   = {A Parallel Algorithm for Multilevel Graph Partitioning and Sparse Matrix Ordering},\n"
  "  journal = {Journal of Parallel and Distributed Computing},\n"
  "  volume  = {48},\n"
  "  pages   = {71--85},\n"
  "  year    = {1998}\n"
  "  doi     = {https://doi.org/10.1006/jpdc.1997.1403}\n"
  "}\n";

typedef struct {
  MPI_Comm  pcomm;
  PetscInt  ptype;
  PetscReal imbalanceRatio;
  PetscInt  debugFlag;
  PetscInt  randomSeed;
} PetscPartitioner_ParMetis;

static const char *ptypes[] = {"kway", "rb"};

static PetscErrorCode PetscPartitionerDestroy_ParMetis(PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_free(&p->pcomm));
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_ParMetis_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ParMetis type: %s\n", ptypes[p->ptype]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "load imbalance ratio %g\n", (double) p->imbalanceRatio));
  PetscCall(PetscViewerASCIIPrintf(viewer, "debug flag %" PetscInt_FMT "\n", p->debugFlag));
  PetscCall(PetscViewerASCIIPrintf(viewer, "random seed %" PetscInt_FMT "\n", p->randomSeed));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_ParMetis(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscPartitionerView_ParMetis_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_ParMetis(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscPartitioner ParMetis Options");
  PetscCall(PetscOptionsEList("-petscpartitioner_parmetis_type", "Partitioning method", "", ptypes, 2, ptypes[p->ptype], &p->ptype, NULL));
  PetscCall(PetscOptionsReal("-petscpartitioner_parmetis_imbalance_ratio", "Load imbalance ratio limit", "", p->imbalanceRatio, &p->imbalanceRatio, NULL));
  PetscCall(PetscOptionsInt("-petscpartitioner_parmetis_debug", "Debugging flag", "", p->debugFlag, &p->debugFlag, NULL));
  PetscCall(PetscOptionsInt("-petscpartitioner_parmetis_seed", "Random seed", "", p->randomSeed, &p->randomSeed, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_ParMetis(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_PARMETIS)
  PetscPartitioner_ParMetis *pm = (PetscPartitioner_ParMetis *) part->data;
  MPI_Comm       comm;
  PetscInt       nvtxs       = numVertices; /* The number of vertices in full graph */
  PetscInt      *vtxdist;                   /* Distribution of vertices across processes */
  PetscInt      *xadj        = start;       /* Start of edge list for each vertex */
  PetscInt      *adjncy      = adjacency;   /* Edge lists for all vertices */
  PetscInt      *vwgt        = NULL;        /* Vertex weights */
  PetscInt      *adjwgt      = NULL;        /* Edge weights */
  PetscInt       wgtflag     = 0;           /* Indicates which weights are present */
  PetscInt       numflag     = 0;           /* Indicates initial offset (0 or 1) */
  PetscInt       ncon        = 1;           /* The number of weights per vertex */
  PetscInt       metis_ptype = pm->ptype;   /* kway or recursive bisection */
  real_t        *tpwgts;                    /* The fraction of vertex weights assigned to each partition */
  real_t        *ubvec;                     /* The balance intolerance for vertex weights */
  PetscInt       options[64];               /* Options */
  PetscInt       v, i, *assignment, *points;
  PetscMPIInt    p, size, rank;
  PetscBool      hasempty = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) part, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Calculate vertex distribution */
  PetscCall(PetscMalloc4(size+1,&vtxdist,nparts*ncon,&tpwgts,ncon,&ubvec,nvtxs,&assignment));
  vtxdist[0] = 0;
  PetscCallMPI(MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm));
  for (p = 2; p <= size; ++p) {
    hasempty = (PetscBool)(hasempty || !vtxdist[p-1] || !vtxdist[p]);
    vtxdist[p] += vtxdist[p-1];
  }
  /* null graph */
  if (vtxdist[size] == 0) {
    PetscCall(PetscFree4(vtxdist,tpwgts,ubvec,assignment));
    PetscCall(ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(0);
  }
  /* Calculate partition weights */
  if (targetSection) {
    PetscInt p;
    real_t   sumt = 0.0;

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
    }
  } else {
    for (p = 0; p < nparts; ++p) tpwgts[p] = 1.0/nparts;
  }
  ubvec[0] = pm->imbalanceRatio;

  /* Weight cells */
  if (vertSection) {
    PetscCall(PetscMalloc1(nvtxs,&vwgt));
    for (v = 0; v < nvtxs; ++v) {
      PetscCall(PetscSectionGetDof(vertSection, v, &vwgt[v]));
    }
    wgtflag |= 2; /* have weights on graph vertices */
  }

  for (p = 0; !vtxdist[p+1] && p < size; ++p);
  if (vtxdist[p+1] == vtxdist[size]) {
    if (rank == p) {
      int err;
      err = METIS_SetDefaultOptions(options); /* initialize all defaults */
      options[METIS_OPTION_DBGLVL] = pm->debugFlag;
      options[METIS_OPTION_SEED]   = pm->randomSeed;
      PetscCheck(err == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_SetDefaultOptions()");
      if (metis_ptype == 1) {
        PetscStackPushExternal("METIS_PartGraphRecursive");
        err = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment);
        PetscStackPop;
        PetscCheck(err == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphRecursive()");
      } else {
        /*
         It would be nice to activate the two options below, but they would need some actual testing.
         - Turning on these options may exercise path of the METIS code that have bugs and may break production runs.
         - If CONTIG is set to 1, METIS will exit with error if the graph is disconnected, despite the manual saying the option is ignored in such case.
        */
        /* options[METIS_OPTION_CONTIG]  = 1; */ /* try to produce partitions that are contiguous */
        /* options[METIS_OPTION_MINCONN] = 1; */ /* minimize the maximum degree of the subdomain graph */
        PetscStackPushExternal("METIS_PartGraphKway");
        err = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment);
        PetscStackPop;
        PetscCheck(err == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      }
    }
  } else {
    MPI_Comm pcomm = pm->pcomm;

    options[0] = 1; /*use options */
    options[1] = pm->debugFlag;
    options[2] = (pm->randomSeed == -1) ? 15 : pm->randomSeed; /* default is GLOBAL_SEED=15 from `libparmetis/defs.h` */

    if (hasempty) { /* parmetis does not support empty graphs on some of the processes */
      PetscInt cnt;

      PetscCallMPI(MPI_Comm_split(pm->pcomm,!!nvtxs,rank,&pcomm));
      for (p=0,cnt=0;p<size;p++) {
        if (vtxdist[p+1] != vtxdist[p]) {
          vtxdist[cnt+1] = vtxdist[p+1];
          cnt++;
        }
      }
    }
    if (nvtxs) {
      int err;
      PetscStackPushExternal("ParMETIS_V3_PartKway");
      err = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment, &pcomm);
      PetscStackPop;
      PetscCheck(err == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error %d in ParMETIS_V3_PartKway()", err);
    }
    if (hasempty) {
      PetscCallMPI(MPI_Comm_free(&pcomm));
    }
  }

  /* Convert to PetscSection+IS */
  for (v = 0; v < nvtxs; ++v) PetscCall(PetscSectionAddDof(partSection, assignment[v], 1));
  PetscCall(PetscMalloc1(nvtxs, &points));
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  PetscCheck(i == nvtxs,comm, PETSC_ERR_PLIB, "Number of points %" PetscInt_FMT " should be %" PetscInt_FMT, i, nvtxs);
  PetscCall(ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition));
  PetscCall(PetscFree4(vtxdist,tpwgts,ubvec,assignment));
  PetscCall(PetscFree(vwgt));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-parmetis.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_ParMetis(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph             = PETSC_FALSE;
  part->ops->view           = PetscPartitionerView_ParMetis;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_ParMetis;
  part->ops->destroy        = PetscPartitionerDestroy_ParMetis;
  part->ops->partition      = PetscPartitionerPartition_ParMetis;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERPARMETIS = "parmetis" - A PetscPartitioner object using the ParMETIS library

  Level: intermediate

  Options Database Keys:
+  -petscpartitioner_parmetis_type <string> - ParMETIS partitioning type. Either "kway" or "rb" (recursive bisection)
.  -petscpartitioner_parmetis_imbalance_ratio <value> - Load imbalance ratio limit
.  -petscpartitioner_parmetis_debug <int> - Debugging flag passed to ParMETIS/METIS routines
-  -petscpartitioner_parmetis_seed <int> - Random seed

  Notes: when the graph is on a single process, this partitioner actually calls METIS and not ParMETIS

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_ParMetis(PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNewLog(part, &p));
  part->data = p;

  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)part),&p->pcomm));
  p->ptype          = 0;
  p->imbalanceRatio = 1.05;
  p->debugFlag      = 0;
  p->randomSeed     = -1; /* defaults to GLOBAL_SEED=15 from `libparmetis/defs.h` */

  PetscCall(PetscPartitionerInitialize_ParMetis(part));
  PetscCall(PetscCitationsRegister(ParMetisPartitionerCitation, &ParMetisPartitionerCite));
  PetscFunctionReturn(0);
}
