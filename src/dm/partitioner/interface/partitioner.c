#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/

/*@
  PetscPartitionerSetType - Builds a particular `PetscPartitioner`

  Collective

  Input Parameters:
+ part - The `PetscPartitioner` object
- name - The kind of partitioner

  Options Database Key:
. -petscpartitioner_type <type> - Sets the `PetscPartitioner` type

  Level: intermediate

  Note:
.vb
 PETSCPARTITIONERCHACO    - The Chaco partitioner (--download-chaco)
 PETSCPARTITIONERPARMETIS - The ParMetis partitioner (--download-parmetis)
 PETSCPARTITIONERSHELL    - A shell partitioner implemented by the user
 PETSCPARTITIONERSIMPLE   - A simple partitioner that divides cells into equal, contiguous chunks
 PETSCPARTITIONERGATHER   - Gathers all cells onto process 0
.ve

.seealso: `PetscPartitionerGetType()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerSetType(PetscPartitioner part, PetscPartitionerType name)
{
  PetscErrorCode (*r)(PetscPartitioner);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)part, name, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscPartitionerRegisterAll());
  PetscCall(PetscFunctionListFind(PetscPartitionerList, name, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscPartitioner type: %s", name);

  PetscTryTypeMethod(part, destroy);
  part->noGraph = PETSC_FALSE;
  PetscCall(PetscMemzero(part->ops, sizeof(*part->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)part, name));
  PetscCall((*r)(part));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerGetType - Gets the PetscPartitioner type name (as a string) from the object.

  Not Collective

  Input Parameter:
. part - The PetscPartitioner

  Output Parameter:
. name - The PetscPartitioner type name

  Level: intermediate

.seealso: `PetscPartitionerSetType()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerGetType(PetscPartitioner part, PetscPartitionerType *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscAssertPointer(name, 2);
  *name = ((PetscObject)part)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerViewFromOptions - View a `PetscPartitioner` object based on options in the options database

  Collective

  Input Parameters:
+ A    - the `PetscPartitioner` object
. obj  - Optional `PetscObject` that provides the options prefix
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for the various forms of viewers that may be used

.seealso: `PetscPartitionerView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode PetscPartitionerViewFromOptions(PetscPartitioner A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerView - Views a `PetscPartitioner`

  Collective

  Input Parameters:
+ part - the `PetscPartitioner` object to view
- v    - the viewer

  Level: developer

.seealso: `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerView(PetscPartitioner part, PetscViewer v)
{
  PetscMPIInt size;
  PetscBool   isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)part), &v));
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
  if (isascii && part->printHeader) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)part), &size));
    PetscCall(PetscViewerASCIIPrintf(v, "Graph Partitioner: %d MPI Process%s\n", size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(v, "  type: %s\n", ((PetscObject)part)->type_name));
    PetscCall(PetscViewerASCIIPrintf(v, "  edge cut: %" PetscInt_FMT "\n", part->edgeCut));
    PetscCall(PetscViewerASCIIPrintf(v, "  balance: %.2g\n", (double)part->balance));
    PetscCall(PetscViewerASCIIPrintf(v, "  use vertex weights: %d\n", part->usevwgt));
    PetscCall(PetscViewerASCIIPrintf(v, "  use edge weights: %d\n", part->useewgt));
  }
  PetscTryTypeMethod(part, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerGetDefaultType(MPI_Comm comm, const char **defaultType)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    *defaultType = PETSCPARTITIONERSIMPLE;
  } else {
#if defined(PETSC_HAVE_PARMETIS)
    *defaultType = PETSCPARTITIONERPARMETIS;
#elif defined(PETSC_HAVE_PTSCOTCH)
    *defaultType = PETSCPARTITIONERPTSCOTCH;
#elif defined(PETSC_HAVE_CHACO)
    *defaultType = PETSCPARTITIONERCHACO;
#else
    *defaultType = PETSCPARTITIONERSIMPLE;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerSetFromOptions - sets parameters in a `PetscPartitioner` from the options database

  Collective

  Input Parameter:
. part - the `PetscPartitioner` object to set options for

  Options Database Keys:
+ -petscpartitioner_type <type>        - Sets the `PetscPartitioner` type; use -help for a list of available types
. -petscpartitioner_use_vertex_weights - Uses weights associated with the graph vertices
- -petscpartitioner_view_graph         - View the graph each time PetscPartitionerPartition is called. Viewer can be customized, see `PetscOptionsCreateViewer()`

  Level: developer

.seealso: `PetscPartitionerView()`, `PetscPartitionerSetType()`, `PetscPartitionerPartition()`
@*/
PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner part)
{
  const char *currentType = NULL;
  char        name[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)part);
  PetscCall(PetscPartitionerGetType(part, &currentType));
  PetscCall(PetscOptionsFList("-petscpartitioner_type", "Graph partitioner", "PetscPartitionerSetType", PetscPartitionerList, currentType, name, sizeof(name), &flg));
  if (flg) PetscCall(PetscPartitionerSetType(part, name));
  PetscCall(PetscOptionsBool("-petscpartitioner_use_vertex_weights", "Use vertex weights", "", part->usevwgt, &part->usevwgt, NULL));
  PetscCall(PetscOptionsBool("-petscpartitioner_use_edge_weights", "Use edge weights", "", part->useewgt, &part->useewgt, NULL));
  PetscTryTypeMethod(part, setfromoptions, PetscOptionsObject);
  PetscCall(PetscViewerDestroy(&part->viewer));
  PetscCall(PetscViewerDestroy(&part->viewerGraph));
  PetscCall(PetscOptionsCreateViewer(((PetscObject)part)->comm, ((PetscObject)part)->options, ((PetscObject)part)->prefix, "-petscpartitioner_view", &part->viewer, &part->viewerFmt, NULL));
  PetscCall(PetscOptionsCreateViewer(((PetscObject)part)->comm, ((PetscObject)part)->options, ((PetscObject)part)->prefix, "-petscpartitioner_view_graph", &part->viewerGraph, NULL, NULL));
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)part, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerSetUp - Construct data structures for the `PetscPartitioner`

  Collective

  Input Parameter:
. part - the `PetscPartitioner` object to setup

  Level: developer

.seealso: `PetscPartitionerView()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerSetUp(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscTryTypeMethod(part, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerReset - Resets data structures for the `PetscPartitioner`

  Collective

  Input Parameter:
. part - the `PetscPartitioner` object to reset

  Level: developer

.seealso: `PetscPartitionerSetUp()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerReset(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscTryTypeMethod(part, reset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerDestroy - Destroys a `PetscPartitioner` object

  Collective

  Input Parameter:
. part - the `PetscPartitioner` object to destroy

  Level: developer

.seealso: `PetscPartitionerView()`
@*/
PetscErrorCode PetscPartitionerDestroy(PetscPartitioner *part)
{
  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*part, PETSCPARTITIONER_CLASSID, 1);

  if (--((PetscObject)*part)->refct > 0) {
    *part = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  ((PetscObject)*part)->refct = 0;

  PetscCall(PetscPartitionerReset(*part));

  PetscCall(PetscViewerDestroy(&(*part)->viewer));
  PetscCall(PetscViewerDestroy(&(*part)->viewerGraph));
  PetscTryTypeMethod(*part, destroy);
  PetscCall(PetscHeaderDestroy(part));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerPartition - Partition a graph

  Collective

  Input Parameters:
+ part          - The `PetscPartitioner`
. nparts        - Number of partitions
. numVertices   - Number of vertices in the local part of the graph
. start         - row pointers for the local part of the graph (CSR style)
. adjacency     - adjacency list (CSR style)
. vertexSection - PetscSection describing the absolute weight of each local vertex (can be `NULL`)
. edgeSection   - PetscSection describing the absolute weight of each local edge (can be `NULL`)
- targetSection - PetscSection describing the absolute weight of each partition (can be `NULL`)

  Output Parameters:
+ partSection - The `PetscSection` giving the division of points by partition
- partition   - The list of points by partition

  Options Database Keys:
+ -petscpartitioner_view       - View the partitioner information
- -petscpartitioner_view_graph - View the graph we are partitioning

  Level: developer

  Notes:
  The chart of the vertexSection (if present) must contain [0,numVertices), with the number of dofs in the section specifying the absolute weight for each vertex.
  The chart of the targetSection (if present) must contain [0,nparts), with the number of dofs in the section specifying the absolute weight for each partition. This information must be the same across processes, PETSc does not check it.

.seealso: `PetscPartitionerCreate()`, `PetscPartitionerSetType()`, `PetscSectionCreate()`, `PetscSectionSetChart()`, `PetscSectionSetDof()`
@*/
PetscErrorCode PetscPartitionerPartition(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertexSection, PetscSection edgeSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(part, nparts, 2);
  PetscCheck(nparts > 0, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_OUTOFRANGE, "Number of parts must be positive");
  PetscCheck(numVertices >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices must be non-negative");
  if (numVertices && !part->noGraph) {
    PetscAssertPointer(start, 4);
    PetscAssertPointer(start + numVertices, 4);
    if (start[numVertices]) PetscAssertPointer(adjacency, 5);
  }
  if (vertexSection) {
    PetscInt s, e;

    PetscValidHeaderSpecific(vertexSection, PETSC_SECTION_CLASSID, 6);
    PetscCall(PetscSectionGetChart(vertexSection, &s, &e));
    PetscCheck(s <= 0 && e >= numVertices, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid vertexSection chart [%" PetscInt_FMT ",%" PetscInt_FMT ")", s, e);
  }
  if (edgeSection) {
    PetscInt s, e;

    PetscValidHeaderSpecific(edgeSection, PETSC_SECTION_CLASSID, 7);
    PetscCall(PetscSectionGetChart(edgeSection, &s, &e));
    PetscCheck(s <= 0 && e >= start[numVertices], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid edgeSection chart [%" PetscInt_FMT ",%" PetscInt_FMT ")", s, e);
  }
  if (targetSection) {
    PetscInt s, e;

    PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 8);
    PetscCall(PetscSectionGetChart(targetSection, &s, &e));
    PetscCheck(s <= 0 && e >= nparts, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid targetSection chart [%" PetscInt_FMT ",%" PetscInt_FMT ")", s, e);
  }
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 9);
  PetscAssertPointer(partition, 10);

  PetscCall(PetscSectionReset(partSection));
  PetscCall(PetscSectionSetChart(partSection, 0, nparts));
  if (nparts == 1) { /* quick */
    PetscCall(PetscSectionSetDof(partSection, 0, numVertices));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)part), numVertices, 0, 1, partition));
  } else PetscUseTypeMethod(part, partition, nparts, numVertices, start, adjacency, vertexSection, edgeSection, targetSection, partSection, partition);
  PetscCall(PetscSectionSetUp(partSection));
  if (part->viewerGraph) {
    PetscViewer viewer = part->viewerGraph;
    PetscBool   isascii;
    PetscInt    v, i;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
    if (isascii) {
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]Nv: %" PetscInt_FMT "\n", rank, numVertices));
      for (v = 0; v < numVertices; ++v) {
        const PetscInt s = start[v];
        const PetscInt e = start[v + 1];

        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]  ", rank));
        for (i = s; i < e; ++i) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT " ", adjacency[i]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%" PetscInt_FMT "-%" PetscInt_FMT ")\n", s, e));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  }
  if (part->viewer) {
    PetscCall(PetscViewerPushFormat(part->viewer, part->viewerFmt));
    PetscCall(PetscPartitionerView(part, part->viewer));
    PetscCall(PetscViewerPopFormat(part->viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerCreate - Creates an empty `PetscPartitioner` object. The type can then be set with `PetscPartitionerSetType()`.

  Collective

  Input Parameter:
. comm - The communicator for the `PetscPartitioner` object

  Output Parameter:
. part - The `PetscPartitioner` object

  Level: beginner

.seealso: `PetscPartitionerSetType()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerCreate(MPI_Comm comm, PetscPartitioner *part)
{
  PetscPartitioner p;
  const char      *partitionerType = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(part, 2);
  *part = NULL;
  PetscCall(PetscPartitionerInitializePackage());

  PetscCall(PetscHeaderCreate(p, PETSCPARTITIONER_CLASSID, "PetscPartitioner", "Graph Partitioner", "PetscPartitioner", comm, PetscPartitionerDestroy, PetscPartitionerView));
  PetscCall(PetscPartitionerGetDefaultType(comm, &partitionerType));
  PetscCall(PetscPartitionerSetType(p, partitionerType));

  p->usevwgt     = PETSC_TRUE;
  p->printHeader = PETSC_TRUE;

  *part = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}
