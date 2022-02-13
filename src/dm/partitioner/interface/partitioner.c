#include <petsc/private/partitionerimpl.h>        /*I "petscpartitioner.h" I*/

/*@C
  PetscPartitionerSetType - Builds a particular PetscPartitioner

  Collective on PetscPartitioner

  Input Parameters:
+ part - The PetscPartitioner object
- name - The kind of partitioner

  Options Database Key:
. -petscpartitioner_type <type> - Sets the PetscPartitioner type; use -help for a list of available types

  Level: intermediate

.seealso: PetscPartitionerGetType(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerSetType(PetscPartitioner part, PetscPartitionerType name)
{
  PetscErrorCode (*r)(PetscPartitioner);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) part, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscPartitionerRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscPartitionerList, name, &r);CHKERRQ(ierr);
  PetscCheckFalse(!r,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscPartitioner type: %s", name);

  if (part->ops->destroy) {
    ierr = (*part->ops->destroy)(part);CHKERRQ(ierr);
  }
  part->noGraph = PETSC_FALSE;
  ierr = PetscMemzero(part->ops, sizeof(*part->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) part, name);CHKERRQ(ierr);
  ierr = (*r)(part);CHKERRQ(ierr);
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

.seealso: PetscPartitionerSetType(), PetscPartitionerCreate()
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
.seealso:  PetscPartitionerView(), PetscObjectViewFromOptions()
@*/
PetscErrorCode PetscPartitionerViewFromOptions(PetscPartitioner A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCPARTITIONER_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerView - Views a PetscPartitioner

  Collective on PetscPartitioner

  Input Parameters:
+ part - the PetscPartitioner object to view
- v    - the viewer

  Level: developer

.seealso: PetscPartitionerDestroy()
@*/
PetscErrorCode PetscPartitionerView(PetscPartitioner part, PetscViewer v)
{
  PetscMPIInt    size;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) part), &v);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) part), &size);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPrintf(v, "Graph Partitioner: %d MPI Process%s\n", size, size > 1 ? "es" : "");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "  type: %s\n", ((PetscObject)part)->type_name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "  edge cut: %D\n", part->edgeCut);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "  balance: %.2g\n", part->balance);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "  use vertex weights: %d\n", part->usevwgt);CHKERRQ(ierr);
  }
  if (part->ops->view) {ierr = (*part->ops->view)(part, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerGetDefaultType(MPI_Comm comm, const char **defaultType)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
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

.seealso: PetscPartitionerView(), PetscPartitionerSetType(), PetscPartitionerPartition()
@*/
PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner part)
{
  const char    *currentType = NULL;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscObjectOptionsBegin((PetscObject) part);CHKERRQ(ierr);
  ierr = PetscPartitionerGetType(part, &currentType);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscpartitioner_type", "Graph partitioner", "PetscPartitionerSetType", PetscPartitionerList, currentType, name, sizeof(name), &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPartitionerSetType(part, name);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-petscpartitioner_use_vertex_weights","Use vertex weights","",part->usevwgt,&part->usevwgt,NULL);CHKERRQ(ierr);
  if (part->ops->setfromoptions) {
    ierr = (*part->ops->setfromoptions)(PetscOptionsObject,part);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&part->viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&part->viewerGraph);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(((PetscObject) part)->comm, ((PetscObject) part)->options, ((PetscObject) part)->prefix, "-petscpartitioner_view", &part->viewer, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(((PetscObject) part)->comm, ((PetscObject) part)->options, ((PetscObject) part)->prefix, "-petscpartitioner_view_graph", &part->viewerGraph, NULL, &part->viewGraph);CHKERRQ(ierr);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) part);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerSetUp - Construct data structures for the PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to setup

  Level: developer

.seealso: PetscPartitionerView(), PetscPartitionerDestroy()
@*/
PetscErrorCode PetscPartitionerSetUp(PetscPartitioner part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (part->ops->setup) {ierr = (*part->ops->setup)(part);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerReset - Resets data structures for the PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to reset

  Level: developer

.seealso: PetscPartitionerSetUp(), PetscPartitionerDestroy()
@*/
PetscErrorCode PetscPartitionerReset(PetscPartitioner part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (part->ops->reset) {ierr = (*part->ops->reset)(part);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerDestroy - Destroys a PetscPartitioner object

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to destroy

  Level: developer

.seealso: PetscPartitionerView()
@*/
PetscErrorCode PetscPartitionerDestroy(PetscPartitioner *part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part), PETSCPARTITIONER_CLASSID, 1);

  if (--((PetscObject)(*part))->refct > 0) {*part = NULL; PetscFunctionReturn(0);}
  ((PetscObject) (*part))->refct = 0;

  ierr = PetscPartitionerReset(*part);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&(*part)->viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*part)->viewerGraph);CHKERRQ(ierr);
  if ((*part)->ops->destroy) {ierr = (*(*part)->ops->destroy)(*part);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(part);CHKERRQ(ierr);
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

.seealso PetscPartitionerCreate(), PetscSectionCreate(), PetscSectionSetChart(), PetscSectionSetDof()
@*/
PetscErrorCode PetscPartitionerPartition(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertexSection, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(part, nparts, 2);
  PetscCheckFalse(nparts <= 0,PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_OUTOFRANGE, "Number of parts must be positive");
  PetscCheckFalse(numVertices < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices must be non-negative");
  if (numVertices && !part->noGraph) {
    PetscValidIntPointer(start, 4);
    PetscValidIntPointer(start + numVertices, 4);
    if (start[numVertices]) PetscValidIntPointer(adjacency, 5);
  }
  if (vertexSection) {
    PetscInt s,e;

    PetscValidHeaderSpecific(vertexSection, PETSC_SECTION_CLASSID, 6);
    ierr = PetscSectionGetChart(vertexSection, &s, &e);CHKERRQ(ierr);
    PetscCheckFalse(s > 0 || e < numVertices,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid vertexSection chart [%D,%D)",s,e);
  }
  if (targetSection) {
    PetscInt s,e;

    PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 7);
    ierr = PetscSectionGetChart(targetSection, &s, &e);CHKERRQ(ierr);
    PetscCheckFalse(s > 0 || e < nparts,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid targetSection chart [%D,%D)",s,e);
  }
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 8);
  PetscValidPointer(partition, 9);

  ierr = PetscSectionReset(partSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  if (nparts == 1) { /* quick */
    ierr = PetscSectionSetDof(partSection, 0, numVertices);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)part),numVertices,0,1,partition);CHKERRQ(ierr);
  } else {
    PetscCheckFalse(!part->ops->partition,PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "PetscPartitioner %s has no partitioning method", ((PetscObject)part)->type_name);
    ierr = (*part->ops->partition)(part, nparts, numVertices, start, adjacency, vertexSection, targetSection, partSection, partition);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  if (part->viewerGraph) {
    PetscViewer viewer = part->viewerGraph;
    PetscBool   isascii;
    PetscInt    v, i;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
    if (isascii) {
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]Nv: %D\n", rank, numVertices);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        const PetscInt s = start[v];
        const PetscInt e = start[v+1];

        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]  ", rank);CHKERRQ(ierr);
        for (i = s; i < e; ++i) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%D ", adjacency[i]);CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D-%D)\n", s, e);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  }
  if (part->viewer) {
    ierr = PetscPartitionerView(part,part->viewer);CHKERRQ(ierr);
  }
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

.seealso: PetscPartitionerSetType(), PETSCPARTITIONERCHACO, PETSCPARTITIONERPARMETIS, PETSCPARTITIONERSHELL, PETSCPARTITIONERSIMPLE, PETSCPARTITIONERGATHER
@*/
PetscErrorCode PetscPartitionerCreate(MPI_Comm comm, PetscPartitioner *part)
{
  PetscPartitioner p;
  const char       *partitionerType = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(part, 2);
  *part = NULL;
  ierr = PetscPartitionerInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, PETSCPARTITIONER_CLASSID, "PetscPartitioner", "Graph Partitioner", "PetscPartitioner", comm, PetscPartitionerDestroy, PetscPartitionerView);CHKERRQ(ierr);
  ierr = PetscPartitionerGetDefaultType(comm, &partitionerType);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(p, partitionerType);CHKERRQ(ierr);

  p->edgeCut = 0;
  p->balance = 0.0;
  p->usevwgt = PETSC_TRUE;

  *part = p;
  PetscFunctionReturn(0);
}

