#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

/*@C
  PetscPartitionerSetType - Builds a particular PetscPartitioner

  Collective on PetscPartitioner

  Input Parameters:
+ part - The PetscPartitioner object
- name - The kind of partitioner

  Options Database Key:
. -petscpartitioner_type <type> - Sets the PetscPartitioner type; use -help for a list of available types

  Note:
$ PETSCPARTITIONERCHACO    - The Chaco partitioner (--download-chaco)
$ PETSCPARTITIONERPARMETIS - The ParMetis partitioner (--download-parmetis)
$ PETSCPARTITIONERSHELL    - A shell partitioner implemented by the user
$ PETSCPARTITIONERSIMPLE   - A simple partitioner that divides cells into equal, contiguous chunks
$ PETSCPARTITIONERGATHER   - Gathers all cells onto process 0

  Level: intermediate

.seealso: `PetscPartitionerGetType()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode PetscPartitionerSetType(PetscPartitioner part, PetscPartitionerType name)
{
  PetscErrorCode (*r)(PetscPartitioner);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject) part, name, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscPartitionerRegisterAll());
  PetscCall(PetscFunctionListFind(PetscPartitionerList, name, &r));
  PetscCheck(r,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscPartitioner type: %s", name);

  if (part->ops->destroy) PetscCall((*part->ops->destroy)(part));
  part->noGraph = PETSC_FALSE;
  PetscCall(PetscMemzero(part->ops, sizeof(*part->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject) part, name));
  PetscCall((*r)(part));
  PetscFunctionReturn(0);
}

/*@C
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
  PetscValidPointer(name, 2);
  *name = ((PetscObject) part)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PetscPartitionerViewFromOptions - View from Options

   Collective on PetscPartitioner

   Input Parameters:
+  A - the PetscPartitioner object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso: `PetscPartitionerView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode PetscPartitionerViewFromOptions(PetscPartitioner A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCPARTITIONER_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerView - Views a PetscPartitioner

  Collective on PetscPartitioner

  Input Parameters:
+ part - the PetscPartitioner object to view
- v    - the viewer

  Level: developer

.seealso: `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerView(PetscPartitioner part, PetscViewer v)
{
  PetscMPIInt    size;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) part), &v));
  PetscCall(PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) part), &size));
    PetscCall(PetscViewerASCIIPrintf(v, "Graph Partitioner: %d MPI Process%s\n", size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(v, "  type: %s\n", ((PetscObject)part)->type_name));
    PetscCall(PetscViewerASCIIPrintf(v, "  edge cut: %" PetscInt_FMT "\n", part->edgeCut));
    PetscCall(PetscViewerASCIIPrintf(v, "  balance: %.2g\n", (double)part->balance));
    PetscCall(PetscViewerASCIIPrintf(v, "  use vertex weights: %d\n", part->usevwgt));
  }
  if (part->ops->view) PetscCall((*part->ops->view)(part, v));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerGetDefaultType(MPI_Comm comm, const char **defaultType)
{
  PetscMPIInt    size;

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
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerSetFromOptions - sets parameters in a PetscPartitioner from the options database

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to set options for

  Options Database Keys:
+  -petscpartitioner_type <type> - Sets the PetscPartitioner type; use -help for a list of available types
.  -petscpartitioner_use_vertex_weights - Uses weights associated with the graph vertices
-  -petscpartitioner_view_graph - View the graph each time PetscPartitionerPartition is called. Viewer can be customized, see PetscOptionsGetViewer()

  Level: developer

.seealso: `PetscPartitionerView()`, `PetscPartitionerSetType()`, `PetscPartitionerPartition()`
@*/
PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner part)
{
  const char    *currentType = NULL;
  char           name[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject) part);
  PetscCall(PetscPartitionerGetType(part, &currentType));
  PetscCall(PetscOptionsFList("-petscpartitioner_type", "Graph partitioner", "PetscPartitionerSetType", PetscPartitionerList, currentType, name, sizeof(name), &flg));
  if (flg) PetscCall(PetscPartitionerSetType(part, name));
  PetscCall(PetscOptionsBool("-petscpartitioner_use_vertex_weights","Use vertex weights","",part->usevwgt,&part->usevwgt,NULL));
  if (part->ops->setfromoptions) PetscCall((*part->ops->setfromoptions)(PetscOptionsObject,part));
  PetscCall(PetscViewerDestroy(&part->viewer));
  PetscCall(PetscViewerDestroy(&part->viewerGraph));
  PetscCall(PetscOptionsGetViewer(((PetscObject) part)->comm, ((PetscObject) part)->options, ((PetscObject) part)->prefix, "-petscpartitioner_view", &part->viewer, NULL, NULL));
  PetscCall(PetscOptionsGetViewer(((PetscObject) part)->comm, ((PetscObject) part)->options, ((PetscObject) part)->prefix, "-petscpartitioner_view_graph", &part->viewerGraph, NULL, &part->viewGraph));
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) part));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerSetUp - Construct data structures for the PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to setup

  Level: developer

.seealso: `PetscPartitionerView()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerSetUp(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (part->ops->setup) PetscCall((*part->ops->setup)(part));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerReset - Resets data structures for the PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to reset

  Level: developer

.seealso: `PetscPartitionerSetUp()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerReset(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (part->ops->reset) PetscCall((*part->ops->reset)(part));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerDestroy - Destroys a PetscPartitioner object

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to destroy

  Level: developer

.seealso: `PetscPartitionerView()`
@*/
PetscErrorCode PetscPartitionerDestroy(PetscPartitioner *part)
{
  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part), PETSCPARTITIONER_CLASSID, 1);

  if (--((PetscObject)(*part))->refct > 0) {*part = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*part))->refct = 0;

  PetscCall(PetscPartitionerReset(*part));

  PetscCall(PetscViewerDestroy(&(*part)->viewer));
  PetscCall(PetscViewerDestroy(&(*part)->viewerGraph));
  if ((*part)->ops->destroy) PetscCall((*(*part)->ops->destroy)(*part));
  PetscCall(PetscHeaderDestroy(part));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerPartition - Partition a graph

  Collective on PetscPartitioner

  Input Parameters:
+ part    - The PetscPartitioner
. nparts  - Number of partitions
. numVertices - Number of vertices in the local part of the graph
. start - row pointers for the local part of the graph (CSR style)
. adjacency - adjacency list (CSR style)
. vertexSection - PetscSection describing the absolute weight of each local vertex (can be NULL)
- targetSection - PetscSection describing the absolute weight of each partition (can be NULL)

  Output Parameters:
+ partSection     - The PetscSection giving the division of points by partition
- partition       - The list of points by partition

  Options Database:
+ -petscpartitioner_view - View the partitioner information
- -petscpartitioner_view_graph - View the graph we are partitioning

  Notes:
    The chart of the vertexSection (if present) must contain [0,numVertices), with the number of dofs in the section specifying the absolute weight for each vertex.
    The chart of the targetSection (if present) must contain [0,nparts), with the number of dofs in the section specifying the absolute weight for each partition. This information must be the same across processes, PETSc does not check it.

  Level: developer

.seealso `PetscPartitionerCreate()`, `PetscPartitionerSetType()`, `PetscSectionCreate()`, `PetscSectionSetChart()`, `PetscSectionSetDof()`
@*/
PetscErrorCode PetscPartitionerPartition(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertexSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(part, nparts, 2);
  PetscCheck(nparts > 0,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_OUTOFRANGE, "Number of parts must be positive");
  PetscCheck(numVertices >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices must be non-negative");
  if (numVertices && !part->noGraph) {
    PetscValidIntPointer(start, 4);
    PetscValidIntPointer(start + numVertices, 4);
    if (start[numVertices]) PetscValidIntPointer(adjacency, 5);
  }
  if (vertexSection) {
    PetscInt s,e;

    PetscValidHeaderSpecific(vertexSection, PETSC_SECTION_CLASSID, 6);
    PetscCall(PetscSectionGetChart(vertexSection, &s, &e));
    PetscCheck(s <= 0 && e >= numVertices,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid vertexSection chart [%" PetscInt_FMT ",%" PetscInt_FMT ")",s,e);
  }
  if (targetSection) {
    PetscInt s,e;

    PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 7);
    PetscCall(PetscSectionGetChart(targetSection, &s, &e));
    PetscCheck(s <= 0 && e >= nparts,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid targetSection chart [%" PetscInt_FMT ",%" PetscInt_FMT ")",s,e);
  }
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 8);
  PetscValidPointer(partition, 9);

  PetscCall(PetscSectionReset(partSection));
  PetscCall(PetscSectionSetChart(partSection, 0, nparts));
  if (nparts == 1) { /* quick */
    PetscCall(PetscSectionSetDof(partSection, 0, numVertices));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)part),numVertices,0,1,partition));
  } else {
    PetscCheck(part->ops->partition,PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "PetscPartitioner %s has no partitioning method", ((PetscObject)part)->type_name);
    PetscCall((*part->ops->partition)(part, nparts, numVertices, start, adjacency, vertexSection, targetSection, partSection, partition));
  }
  PetscCall(PetscSectionSetUp(partSection));
  if (part->viewerGraph) {
    PetscViewer viewer = part->viewerGraph;
    PetscBool   isascii;
    PetscInt    v, i;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank));
    PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
    if (isascii) {
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]Nv: %" PetscInt_FMT "\n", rank, numVertices));
      for (v = 0; v < numVertices; ++v) {
        const PetscInt s = start[v];
        const PetscInt e = start[v+1];

        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]  ", rank));
        for (i = s; i < e; ++i) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT " ", adjacency[i]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%" PetscInt_FMT "-%" PetscInt_FMT ")\n", s, e));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  }
  if (part->viewer) PetscCall(PetscPartitionerView(part,part->viewer));
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerCreate - Creates an empty PetscPartitioner object. The type can then be set with PetscPartitionerSetType().

  Collective

  Input Parameter:
. comm - The communicator for the PetscPartitioner object

  Output Parameter:
. part - The PetscPartitioner object

  Level: beginner

.seealso: `PetscPartitionerSetType()`, `PetscPartitionerDestroy()`
@*/
PetscErrorCode PetscPartitionerCreate(MPI_Comm comm, PetscPartitioner *part)
{
  PetscPartitioner p;
  const char       *partitionerType = NULL;

  PetscFunctionBegin;
  PetscValidPointer(part, 2);
  *part = NULL;
  PetscCall(PetscPartitionerInitializePackage());

  PetscCall(PetscHeaderCreate(p, PETSCPARTITIONER_CLASSID, "PetscPartitioner", "Graph Partitioner", "PetscPartitioner", comm, PetscPartitionerDestroy, PetscPartitionerView));
  PetscCall(PetscPartitionerGetDefaultType(comm, &partitionerType));
  PetscCall(PetscPartitionerSetType(p, partitionerType));

  p->edgeCut = 0;
  p->balance = 0.0;
  p->usevwgt = PETSC_TRUE;

  *part = p;
  PetscFunctionReturn(0);
}
