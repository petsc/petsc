#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

typedef struct {
  PetscSection section;   /* Sizes for each partition */
  IS           partition; /* Points in each partition */
  PetscBool    random;    /* Flag for a random partition */
} PetscPartitioner_Shell;

static PetscErrorCode PetscPartitionerReset_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscSectionDestroy(&p->section);CHKERRQ(ierr);
  ierr = ISDestroy(&p->partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_Shell(PetscPartitioner part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPartitionerReset_Shell(part);CHKERRQ(ierr);
  ierr = PetscFree(part->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell_ASCII(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (p->random) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "using random partition\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Shell_ASCII(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Shell(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscBool      random = PETSC_FALSE, set;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner Shell Options");CHKERRQ(ierr);
  ierr = PetscPartitionerShellGetRandom(part, &random);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscpartitioner_shell_random", "Use a random partition", "PetscPartitionerView", PETSC_FALSE, &random, &set);CHKERRQ(ierr);
  if (set) {ierr = PetscPartitionerShellSetRandom(part, random);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Shell(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscInt                np;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (p->random) {
    PetscRandom r;
    PetscInt   *sizes, *points, v, p;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) part), &rank);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, (PetscScalar) nparts);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscCalloc2(nparts, &sizes, numVertices, &points);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {points[v] = v;}
    for (p = 0; p < nparts; ++p) {sizes[p] = numVertices/nparts + (PetscInt) (p < numVertices % nparts);}
    for (v = numVertices-1; v > 0; --v) {
      PetscReal val;
      PetscInt  w, tmp;

      ierr = PetscRandomSetInterval(r, 0.0, (PetscScalar) (v+1));CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(r, &val);CHKERRQ(ierr);
      w    = PetscFloorReal(val);
      tmp       = points[v];
      points[v] = points[w];
      points[w] = tmp;
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, nparts, sizes, points);CHKERRQ(ierr);
    ierr = PetscFree2(sizes, points);CHKERRQ(ierr);
  }
  if (!p->section) SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_WRONG, "Shell partitioner information not provided. Please call PetscPartitionerShellSetPartition()");
  ierr = PetscSectionGetChart(p->section, NULL, &np);CHKERRQ(ierr);
  if (nparts != np) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of requested partitions %d != configured partitions %d", nparts, np);
  ierr = ISGetLocalSize(p->partition, &np);CHKERRQ(ierr);
  if (numVertices != np) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input vertices %d != configured vertices %d", numVertices, np);
  ierr = PetscSectionCopy(p->section, partSection);CHKERRQ(ierr);
  *partition = p->partition;
  ierr = PetscObjectReference((PetscObject) p->partition);CHKERRQ(ierr);
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
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = PetscPartitionerInitialize_Shell(part);CHKERRQ(ierr);
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
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(part, PETSCPARTITIONER_CLASSID, 1, PETSCPARTITIONERSHELL);
  if (sizes)  {PetscValidPointer(sizes, 3);}
  if (points) {PetscValidPointer(points, 4);}
  ierr = PetscSectionDestroy(&p->section);CHKERRQ(ierr);
  ierr = ISDestroy(&p->partition);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) part), &p->section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(p->section, 0, size);CHKERRQ(ierr);
  if (sizes) {
    for (proc = 0; proc < size; ++proc) {
      ierr = PetscSectionSetDof(p->section, proc, sizes[proc]);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(p->section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(p->section, &numPoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) part), numPoints, points, PETSC_COPY_VALUES, &p->partition);CHKERRQ(ierr);
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
  PetscValidPointer(random, 2);
  *random = p->random;
  PetscFunctionReturn(0);
}
