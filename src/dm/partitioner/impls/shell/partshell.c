#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/

typedef struct {
  PetscSection section;   /* Sizes for each partition */
  IS           partition; /* Points in each partition */
  PetscBool    random;    /* Flag for a random partition */
} PetscPartitioner_Shell;

static PetscErrorCode PetscPartitionerReset_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;

  PetscFunctionBegin;
  PetscCall(PetscSectionDestroy(&p->section));
  PetscCall(ISDestroy(&p->partition));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerDestroy_Shell(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscCall(PetscPartitionerReset_Shell(part));
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_Shell_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;

  PetscFunctionBegin;
  if (p->random) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "using random partition\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerView_Shell(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscPartitionerView_Shell_ASCII(part, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Shell(PetscPartitioner part, PetscOptionItems PetscOptionsObject)
{
  PetscInt    sizes[16], points[1024];
  PetscInt    Npart = 16, Npoints = 1024;
  PetscBool   random = PETSC_FALSE, set, flgSizes, flgPoints;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)part), &rank));
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscPartitioner Shell Options");
  PetscCall(PetscPartitionerShellGetRandom(part, &random));
  PetscCall(PetscOptionsBool("-petscpartitioner_shell_random", "Use a random partition", "PetscPartitionerView", PETSC_FALSE, &random, &set));
  if (set) PetscCall(PetscPartitionerShellSetRandom(part, random));
  PetscCall(PetscOptionsIntArray("-petscpartitioner_shell_sizes", "The size of each partition on rank 0", "PetscPartitionerShellSetPartition", sizes, &Npart, &flgSizes));
  PetscCall(PetscOptionsIntArray("-petscpartitioner_shell_points", "The points in each partition on rank 0", "PetscPartitionerShellSetPartition", points, &Npoints, &flgPoints));
  PetscCheck(!(flgSizes ^ flgPoints), PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_WRONG, "Must specify both the partition sizes and points");
  if (flgSizes) {
    PetscInt Np = 0;

    for (PetscInt i = 0; i < Npart; ++i) Np += sizes[i];
    PetscCheck(Np == Npoints, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_WRONG, "Number of input points %" PetscInt_FMT " != %" PetscInt_FMT " sum of partition sizes", Npoints, Np);
    if (!rank) PetscCall(PetscPartitionerShellSetPartition(part, Npart, sizes, points));
    else {
      PetscCall(PetscArrayzero(sizes, Npart));
      PetscCall(PetscPartitionerShellSetPartition(part, Npart, sizes, points));
    }
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerPartition_Shell(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection edgeSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;
  PetscInt                np;

  PetscFunctionBegin;
  if (p->random) {
    PetscRandom r;
    PetscInt   *sizes, *points, v, p;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)part), &rank));
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
    PetscCall(PetscRandomSetInterval(r, 0.0, (PetscScalar)nparts));
    PetscCall(PetscRandomSetFromOptions(r));
    PetscCall(PetscCalloc2(nparts, &sizes, numVertices, &points));
    for (v = 0; v < numVertices; ++v) points[v] = v;
    for (p = 0; p < nparts; ++p) sizes[p] = numVertices / nparts + (PetscInt)(p < numVertices % nparts);
    for (v = numVertices - 1; v > 0; --v) {
      PetscReal val;
      PetscInt  w, tmp;

      PetscCall(PetscRandomSetInterval(r, 0.0, (PetscScalar)(v + 1)));
      PetscCall(PetscRandomGetValueReal(r, &val));
      w         = PetscFloorReal(val);
      tmp       = points[v];
      points[v] = points[w];
      points[w] = tmp;
    }
    PetscCall(PetscRandomDestroy(&r));
    PetscCall(PetscPartitionerShellSetPartition(part, nparts, sizes, points));
    PetscCall(PetscFree2(sizes, points));
  }
  PetscCheck(p->section, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_WRONG, "Shell partitioner information not provided. Please call PetscPartitionerShellSetPartition()");
  PetscCall(PetscSectionGetChart(p->section, NULL, &np));
  PetscCheck(nparts == np, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of requested partitions %" PetscInt_FMT " != configured partitions %" PetscInt_FMT, nparts, np);
  PetscCall(ISGetLocalSize(p->partition, &np));
  PetscCheck(numVertices == np, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input vertices %" PetscInt_FMT " != configured vertices %" PetscInt_FMT, numVertices, np);
  PetscCall(PetscSectionCopy(p->section, partSection));
  *partition = p->partition;
  PetscCall(PetscObjectReference((PetscObject)p->partition));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerInitialize_Shell(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->noGraph             = PETSC_TRUE; /* PetscPartitionerShell cannot overload the partition call, so it is safe for now */
  part->ops->view           = PetscPartitionerView_Shell;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_Shell;
  part->ops->reset          = PetscPartitionerReset_Shell;
  part->ops->destroy        = PetscPartitionerDestroy_Shell;
  part->ops->partition      = PetscPartitionerPartition_Shell;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCPARTITIONERSHELL = "shell" - A PetscPartitioner object

  Level: intermediate

  Options Database Keys:
.  -petscpartitioner_shell_random - Use a random partition

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNew(&p));
  part->data = p;

  PetscCall(PetscPartitionerInitialize_Shell(part));
  p->random = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerShellSetPartition - Set an artificial partition for a mesh

  Collective

  Input Parameters:
+ part   - The `PetscPartitioner`
. size   - The number of partitions
. sizes  - array of length size (or `NULL`) providing the number of points in each partition
- points - array of length sum(sizes) (may be `NULL` iff sizes is `NULL`), a permutation of the points that groups those assigned to each partition in order (i.e., partition 0 first, partition 1 next, etc.)

  Level: developer

  Note:
  It is safe to free the sizes and points arrays after use in this routine.

.seealso: `DMPlexDistribute()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner part, PetscInt size, const PetscInt sizes[], const PetscInt points[])
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;
  PetscInt                proc, numPoints;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  if (sizes) PetscAssertPointer(sizes, 3);
  if (points) PetscAssertPointer(points, 4);
  PetscCall(PetscSectionDestroy(&p->section));
  PetscCall(ISDestroy(&p->partition));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)part), &p->section));
  PetscCall(PetscSectionSetChart(p->section, 0, size));
  if (sizes) {
    for (proc = 0; proc < size; ++proc) PetscCall(PetscSectionSetDof(p->section, proc, sizes[proc]));
  }
  PetscCall(PetscSectionSetUp(p->section));
  PetscCall(PetscSectionGetStorageSize(p->section, &numPoints));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)part), numPoints, points, PETSC_COPY_VALUES, &p->partition));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerShellSetRandom - Set the flag to use a random partition

  Collective

  Input Parameters:
+ part   - The `PetscPartitioner`
- random - The flag to use a random partition

  Level: intermediate

.seealso: `PetscPartitionerShellGetRandom()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerShellSetRandom(PetscPartitioner part, PetscBool random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  p->random = random;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerShellGetRandom - get the flag to use a random partition

  Collective

  Input Parameter:
. part - The `PetscPartitioner`

  Output Parameter:
. random - The flag to use a random partition

  Level: intermediate

.seealso: `PetscPartitionerShellSetRandom()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerShellGetRandom(PetscPartitioner part, PetscBool *random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *)part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  PetscAssertPointer(random, 2);
  *random = p->random;
  PetscFunctionReturn(PETSC_SUCCESS);
}
