#include <petscdmnetwork.h> /*I  "petscdmnetwork.h"  I*/
#include <petscdraw.h>

/*@
  DMNetworkMonitorCreate - Creates a network monitor context

  Collective

  Input Parameters:
. network - network to monitor

  Output Parameters:
. monitorptr - Location to put network monitor context

  Level: intermediate

.seealso: DMNetworkMonitorDestroy(), DMNetworkMonitorAdd()
@*/
PetscErrorCode DMNetworkMonitorCreate(DM network,DMNetworkMonitor *monitorptr)
{
  DMNetworkMonitor monitor;
  MPI_Comm         comm;
  PetscMPIInt      size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)network,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Parallel DMNetworkMonitor is not supported yet");

  PetscCall(PetscMalloc1(1,&monitor));
  monitor->comm      = comm;
  monitor->network   = network;
  monitor->firstnode = NULL;

  *monitorptr = monitor;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkMonitorDestroy - Destroys a network monitor and all associated viewers

  Collective on monitor

  Input Parameters:
. monitor - monitor to destroy

  Level: intermediate

.seealso: DMNetworkMonitorCreate, DMNetworkMonitorAdd
@*/
PetscErrorCode DMNetworkMonitorDestroy(DMNetworkMonitor *monitor)
{
  PetscFunctionBegin;
  while ((*monitor)->firstnode) {
    PetscCall(DMNetworkMonitorPop(*monitor));
  }

  PetscCall(PetscFree(*monitor));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkMonitorPop - Removes the most recently added viewer

  Collective on monitor

  Input Parameters:
. monitor - the monitor

  Level: intermediate

.seealso: DMNetworkMonitorCreate(), DMNetworkMonitorDestroy()
@*/
PetscErrorCode DMNetworkMonitorPop(DMNetworkMonitor monitor)
{
  DMNetworkMonitorList node;

  PetscFunctionBegin;
  if (monitor->firstnode) {
    /* Update links */
    node = monitor->firstnode;
    monitor->firstnode = node->next;

    /* Free list node */
    PetscCall(PetscViewerDestroy(&(node->viewer)));
    PetscCall(VecDestroy(&(node->v)));
    PetscCall(PetscFree(node));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkMonitorAdd - Adds a new viewer to monitor

  Collective on monitor

  Input Parameters:
+ monitor - the monitor
. name - name of viewer
. element - vertex / edge number
. nodes - number of nodes
. start - variable starting offset
. blocksize - variable blocksize
. xmin - xmin (or PETSC_DECIDE) for viewer
. xmax - xmax (or PETSC_DECIDE) for viewer
. ymin - ymin for viewer
. ymax - ymax for viewer
- hold - determines if plot limits should be held

  Level: intermediate

  Notes:
  This is written to be independent of the semantics associated to the variables
  at a given network vertex / edge.

  Precisely, the parameters nodes, start and blocksize allow you to select a general
  strided subarray of the variables to monitor.

.seealso: DMNetworkMonitorCreate(), DMNetworkMonitorDestroy()
@*/
PetscErrorCode DMNetworkMonitorAdd(DMNetworkMonitor monitor,const char *name,PetscInt element,PetscInt nodes,PetscInt start,PetscInt blocksize,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscBool hold)
{
  PetscDrawLG          drawlg;
  PetscDrawAxis        axis;
  PetscMPIInt          rank, size;
  DMNetworkMonitorList node;
  char                 titleBuffer[64];
  PetscInt             vStart,vEnd,eStart,eEnd;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(monitor->comm, &rank));
  PetscCallMPI(MPI_Comm_size(monitor->comm, &size));

  PetscCall(DMNetworkGetVertexRange(monitor->network, &vStart, &vEnd));
  PetscCall(DMNetworkGetEdgeRange(monitor->network, &eStart, &eEnd));

  /* Make window title */
  if (vStart <= element && element < vEnd) {
    PetscCall(PetscSNPrintf(titleBuffer, 64, "%s @ vertex %d [%d / %d]", name, element - vStart, rank, size-1));
  } else if (eStart <= element && element < eEnd) {
    PetscCall(PetscSNPrintf(titleBuffer, 64, "%s @ edge %d [%d / %d]", name, element - eStart, rank, size-1));
  } else {
    /* vertex / edge is not on local machine, so skip! */
    PetscFunctionReturn(0);
  }

  PetscCall(PetscMalloc1(1, &node));

  /* Setup viewer. */
  PetscCall(PetscViewerDrawOpen(monitor->comm, NULL, titleBuffer, PETSC_DECIDE, PETSC_DECIDE, PETSC_DRAW_QUARTER_SIZE, PETSC_DRAW_QUARTER_SIZE, &(node->viewer)));
  PetscCall(PetscViewerPushFormat(node->viewer, PETSC_VIEWER_DRAW_LG_XRANGE));
  PetscCall(PetscViewerDrawGetDrawLG(node->viewer, 0, &drawlg));
  PetscCall(PetscDrawLGGetAxis(drawlg, &axis));
  if (xmin != PETSC_DECIDE && xmax != PETSC_DECIDE) {
    PetscCall(PetscDrawAxisSetLimits(axis, xmin, xmax, ymin, ymax));
  } else {
    PetscCall(PetscDrawAxisSetLimits(axis, 0, nodes-1, ymin, ymax));
  }
  PetscCall(PetscDrawAxisSetHoldLimits(axis, hold));

  /* Setup vector storage for drawing. */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, nodes, &(node->v)));

  node->element   = element;
  node->nodes     = nodes;
  node->start     = start;
  node->blocksize = blocksize;

  node->next         = monitor->firstnode;
  monitor->firstnode = node;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkMonitorView - Monitor function for TSMonitorSet.

  Collectiveon DMNetworkMonitor

  Input Parameters:
+ monitor - DMNetworkMonitor object
- x - TS solution vector

  Level: intermediate

.seealso: DMNetworkMonitorCreate(), DMNetworkMonitorDestroy(), DMNetworkMonitorAdd()
@*/

PetscErrorCode DMNetworkMonitorView(DMNetworkMonitor monitor,Vec x)
{
  PetscInt            varoffset,i,start;
  const PetscScalar   *xx;
  PetscScalar         *vv;
  DMNetworkMonitorList node;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &xx));
  for (node = monitor->firstnode; node; node = node->next) {
    PetscCall(DMNetworkGetGlobalVecOffset(monitor->network, node->element, ALL_COMPONENTS, &varoffset));
    PetscCall(VecGetArray(node->v, &vv));
    start = varoffset + node->start;
    for (i = 0; i < node->nodes; i++) {
      vv[i] = xx[start+i*node->blocksize];
    }
    PetscCall(VecRestoreArray(node->v, &vv));
    PetscCall(VecView(node->v, node->viewer));
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscFunctionReturn(0);
}
