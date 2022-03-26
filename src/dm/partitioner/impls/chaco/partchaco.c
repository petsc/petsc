#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

PetscBool  ChacoPartitionerCite = PETSC_FALSE;
const char ChacoPartitionerCitation[] =
  "@inproceedings{Chaco95,\n"
  "  author    = {Bruce Hendrickson and Robert Leland},\n"
  "  title     = {A multilevel algorithm for partitioning graphs},\n"
  "  booktitle = {Supercomputing '95: Proceedings of the 1995 ACM/IEEE Conference on Supercomputing (CDROM)},"
  "  isbn      = {0-89791-816-9},\n"
  "  pages     = {28},\n"
  "  doi       = {https://doi.acm.org/10.1145/224170.224228},\n"
  "  publisher = {ACM Press},\n"
  "  address   = {New York},\n"
  "  year      = {1995}\n"
  "}\n";

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Chaco;

static PetscErrorCode PetscPartitionerDestroy_Chaco(PetscPartitioner part)
{
  PetscPartitioner_Chaco *p = (PetscPartitioner_Chaco *) part->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(p));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Chaco_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Chaco(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscPartitionerView_Chaco_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHACO)
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
#include <chaco.h>
#else
/* Older versions of Chaco do not have an include file */
PETSC_EXTERN int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);
#endif
extern int FREE_GRAPH;
#endif

static PetscErrorCode PetscPartitionerPartition_Chaco(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_CHACO)
  enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
  MPI_Comm       comm;
  int            nvtxs          = numVertices; /* number of vertices in full graph */
  int           *vwgts          = NULL;   /* weights for all vertices */
  float         *ewgts          = NULL;   /* weights for all edges */
  float         *x              = NULL, *y = NULL, *z = NULL; /* coordinates for inertial method */
  char          *outassignname  = NULL;   /*  name of assignment output file */
  char          *outfilename    = NULL;   /* output file name */
  int            architecture   = 1;      /* 0 => hypercube, d => d-dimensional mesh */
  int            ndims_tot      = 0;      /* total number of cube dimensions to divide */
  int            mesh_dims[3];            /* dimensions of mesh of processors */
  double        *goal          = NULL;    /* desired set sizes for each set */
  int            global_method = 1;       /* global partitioning algorithm */
  int            local_method  = 1;       /* local partitioning algorithm */
  int            rqi_flag      = 0;       /* should I use RQI/Symmlq eigensolver? */
  int            vmax          = 200;     /* how many vertices to coarsen down to? */
  int            ndims         = 1;       /* number of eigenvectors (2^d sets) */
  double         eigtol        = 0.001;   /* tolerance on eigenvectors */
  long           seed          = 123636512; /* for random graph mutations */
#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
  int           *assignment;              /* Output partition */
#else
  short int     *assignment;              /* Output partition */
#endif
  int            fd_stdout, fd_pipe[2];
  PetscInt      *points;
  int            i, v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)part,&comm));
  if (PetscDefined (USE_DEBUG)) {
    int ival,isum;
    PetscBool distributed;

    ival = (numVertices > 0);
    PetscCallMPI(MPI_Allreduce(&ival, &isum, 1, MPI_INT, MPI_SUM, comm));
    distributed = (isum > 1) ? PETSC_TRUE : PETSC_FALSE;
    PetscCheck(!distributed,comm, PETSC_ERR_SUP, "Chaco cannot partition a distributed graph");
  }
  if (!numVertices) { /* distributed case, return if not holding the graph */
    PetscCall(ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(0);
  }
  FREE_GRAPH = 0; /* Do not let Chaco free my memory */
  for (i = 0; i < start[numVertices]; ++i) ++adjacency[i];

  if (global_method == INERTIAL_METHOD) {
    /* manager.createCellCoordinates(nvtxs, &x, &y, &z); */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Inertial partitioning not yet supported");
  }
  mesh_dims[0] = nparts;
  mesh_dims[1] = 1;
  mesh_dims[2] = 1;
  PetscCall(PetscMalloc1(nvtxs, &assignment));
  /* Chaco outputs to stdout. We redirect this to a buffer. */
  /* TODO: check error codes for UNIX calls */
#if defined(PETSC_HAVE_UNISTD_H)
  {
    int piperet;
    piperet = pipe(fd_pipe);
    PetscCheck(!piperet,PETSC_COMM_SELF,PETSC_ERR_SYS,"Could not create pipe");
    fd_stdout = dup(1);
    close(1);
    dup2(fd_pipe[1], 1);
  }
#endif
  if (part->usevwgt) PetscCall(PetscInfo(part,"PETSCPARTITIONERCHACO ignores vertex weights\n"));
  ierr = interface(nvtxs, (int*) start, (int*) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                   assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                   vmax, ndims, eigtol, seed);
#if defined(PETSC_HAVE_UNISTD_H)
  {
    char msgLog[10000];
    int  count;

    fflush(stdout);
    count = read(fd_pipe[0], msgLog, (10000-1)*sizeof(char));
    if (count < 0) count = 0;
    msgLog[count] = 0;
    close(1);
    dup2(fd_stdout, 1);
    close(fd_stdout);
    close(fd_pipe[0]);
    close(fd_pipe[1]);
    PetscCheck(!ierr,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in Chaco library: %s", msgLog);
  }
#else
  PetscCheck(!ierr,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in Chaco library: %s", "error in stdout");
#endif
  /* Convert to PetscSection+IS */
  for (v = 0; v < nvtxs; ++v) {
    PetscCall(PetscSectionAddDof(partSection, assignment[v], 1));
  }
  PetscCall(PetscMalloc1(nvtxs, &points));
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  PetscCheckFalse(i != nvtxs,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  PetscCall(ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition));
  if (global_method == INERTIAL_METHOD) {
    /* manager.destroyCellCoordinates(nvtxs, &x, &y, &z); */
  }
  PetscCall(PetscFree(assignment));
  for (i = 0; i < start[numVertices]; ++i) --adjacency[i];
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-chaco.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_Chaco(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph        = PETSC_FALSE;
  part->ops->view      = PetscPartitionerView_Chaco;
  part->ops->destroy   = PetscPartitionerDestroy_Chaco;
  part->ops->partition = PetscPartitionerPartition_Chaco;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERCHACO = "chaco" - A PetscPartitioner object using the Chaco library

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Chaco(PetscPartitioner part)
{
  PetscPartitioner_Chaco *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNewLog(part, &p));
  part->data = p;

  PetscCall(PetscPartitionerInitialize_Chaco(part));
  PetscCall(PetscCitationsRegister(ChacoPartitionerCitation, &ChacoPartitionerCite));
  PetscFunctionReturn(0);
}
