#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

typedef struct {
  PetscSection section;   /* Sizes for each partition */
  IS           partition; /* Points in each partition */
  PetscBool    random;    /* Flag for a random partition */
} PetscPartitioner_Shell;

static PetscErrorCode PetscPartitionerReset_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionDestroy(&p->section));
  CHKERRQ(ISDestroy(&p->partition));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_Shell(PetscPartitioner part)
{
  PetscFunctionBegin;
  CHKERRQ(PetscPartitionerReset_Shell(part));
  CHKERRQ(PetscFree(part->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  if (p->random) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "using random partition\n"));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) CHKERRQ(PetscPartitionerView_Shell_ASCII(part, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Shell(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscBool      random = PETSC_FALSE, set;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject, "PetscPartitioner Shell Options"));
  CHKERRQ(PetscPartitionerShellGetRandom(part, &random));
  CHKERRQ(PetscOptionsBool("-petscpartitioner_shell_random", "Use a random partition", "PetscPartitionerView", PETSC_FALSE, &random, &set));
  if (set) CHKERRQ(PetscPartitionerShellSetRandom(part, random));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Shell(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscInt                np;

  PetscFunctionBegin;
  if (p->random) {
    PetscRandom r;
    PetscInt   *sizes, *points, v, p;
    PetscMPIInt rank;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) part), &rank));
    CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &r));
    CHKERRQ(PetscRandomSetInterval(r, 0.0, (PetscScalar) nparts));
    CHKERRQ(PetscRandomSetFromOptions(r));
    CHKERRQ(PetscCalloc2(nparts, &sizes, numVertices, &points));
    for (v = 0; v < numVertices; ++v) {points[v] = v;}
    for (p = 0; p < nparts; ++p) {sizes[p] = numVertices/nparts + (PetscInt) (p < numVertices % nparts);}
    for (v = numVertices-1; v > 0; --v) {
      PetscReal val;
      PetscInt  w, tmp;

      CHKERRQ(PetscRandomSetInterval(r, 0.0, (PetscScalar) (v+1)));
      CHKERRQ(PetscRandomGetValueReal(r, &val));
      w    = PetscFloorReal(val);
      tmp       = points[v];
      points[v] = points[w];
      points[w] = tmp;
    }
    CHKERRQ(PetscRandomDestroy(&r));
    CHKERRQ(PetscPartitionerShellSetPartition(part, nparts, sizes, points));
    CHKERRQ(PetscFree2(sizes, points));
  }
  PetscCheck(p->section,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_WRONG, "Shell partitioner information not provided. Please call PetscPartitionerShellSetPartition()");
  CHKERRQ(PetscSectionGetChart(p->section, NULL, &np));
  PetscCheckFalse(nparts != np,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of requested partitions %d != configured partitions %d", nparts, np);
  CHKERRQ(ISGetLocalSize(p->partition, &np));
  PetscCheckFalse(numVertices != np,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input vertices %d != configured vertices %d", numVertices, np);
  CHKERRQ(PetscSectionCopy(p->section, partSection));
  *partition = p->partition;
  CHKERRQ(PetscObjectReference((PetscObject) p->partition));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERSHELL = "shell" - A PetscPartitioner object

  Level: intermediate

  Options Database Keys:
.  -petscpartitioner_shell_random - Use a random partition

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  CHKERRQ(PetscNewLog(part, &p));
  part->data = p;

  CHKERRQ(PetscPartitionerInitialize_Shell(part));
  p->random = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerShellSetPartition - Set an artifical partition for a mesh

  Collective on PetscPartitioner

  Input Parameters:
+ part   - The PetscPartitioner
. size   - The number of partitions
. sizes  - array of length size (or NULL) providing the number of points in each partition
- points - array of length sum(sizes) (may be NULL iff sizes is NULL), a permutation of the points that groups those assigned to each partition in order (i.e., partition 0 first, partition 1 next, etc.)

  Level: developer

  Notes:
    It is safe to free the sizes and points arrays after use in this routine.

.seealso DMPlexDistribute(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner part, PetscInt size, const PetscInt sizes[], const PetscInt points[])
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscInt                proc, numPoints;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  if (sizes)  {PetscValidIntPointer(sizes, 3);}
  if (points) {PetscValidIntPointer(points, 4);}
  CHKERRQ(PetscSectionDestroy(&p->section));
  CHKERRQ(ISDestroy(&p->partition));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) part), &p->section));
  CHKERRQ(PetscSectionSetChart(p->section, 0, size));
  if (sizes) {
    for (proc = 0; proc < size; ++proc) {
      CHKERRQ(PetscSectionSetDof(p->section, proc, sizes[proc]));
    }
  }
  CHKERRQ(PetscSectionSetUp(p->section));
  CHKERRQ(PetscSectionGetStorageSize(p->section, &numPoints));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) part), numPoints, points, PETSC_COPY_VALUES, &p->partition));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerShellSetRandom - Set the flag to use a random partition

  Collective on PetscPartitioner

  Input Parameters:
+ part   - The PetscPartitioner
- random - The flag to use a random partition

  Level: intermediate

.seealso PetscPartitionerShellGetRandom(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellSetRandom(PetscPartitioner part, PetscBool random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  p->random = random;
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerShellGetRandom - get the flag to use a random partition

  Collective on PetscPartitioner

  Input Parameter:
. part   - The PetscPartitioner

  Output Parameter:
. random - The flag to use a random partition

  Level: intermediate

.seealso PetscPartitionerShellSetRandom(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellGetRandom(PetscPartitioner part, PetscBool *random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  PetscValidBoolPointer(random, 2);
  *random = p->random;
  PetscFunctionReturn(0);
}
