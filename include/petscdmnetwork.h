/*
  DMNetwork, for parallel unstructured network problems.
*/
#pragma once

#include <petscdmplex.h>
#include <petscviewer.h>

/* MANSEC = DM */
/* SUBMANSEC = DMNetwork */

#define ALL_COMPONENTS -1

/*MC
   DMNetworkComponentGenericDataType - The integer-sized datatype used by `DMNETWORK` to store user-registered component data on the network

   Level: developer

   Note:
   `DMNetworkComponentGenericDataType` is a typedef for `PetscInt`. The type is needed so that the buffer holding the component data can be communicated with `PetscSF` during `DMNetwork` distribution. User code obtains a pointer to the buffer with `DMNetworkGetComponent()` and must cast it to the appropriate user-defined component struct.

.seealso: [](ch_network), `DM`, `DMNETWORK`, `DMNetworkRegisterComponent()`, `DMNetworkGetComponent()`, `DMNetworkAddComponent()`
M*/
typedef PetscInt DMNetworkComponentGenericDataType;

PETSC_EXTERN PetscErrorCode DMNetworkCreate(MPI_Comm, DM *);
PETSC_EXTERN PetscErrorCode DMNetworkSetNumSubNetworks(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMNetworkGetNumSubNetworks(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkLayoutSetUp(DM);
PETSC_EXTERN PetscErrorCode DMNetworkRegisterComponent(DM, const char *, size_t, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetVertexRange(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetEdgeRange(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetNumEdges(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetNumVertices(DM, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode DMNetworkAddComponent(DM, PetscInt, PetscInt, void *, PetscInt);
PETSC_EXTERN PetscErrorCode DMNetworkGetComponent(DM, PetscInt, PetscInt, PetscInt *, void *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkFinalizeComponents(DM);
PETSC_EXTERN PetscErrorCode DMNetworkGetNumComponents(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetLocalVecOffset(DM, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetGlobalVecOffset(DM, PetscInt, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMNetworkGetEdgeOffset(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetVertexOffset(DM, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMNetworkAssembleGraphStructures(DM);
PETSC_EXTERN PetscErrorCode DMNetworkSetVertexLocalToGlobalOrdering(DM);
PETSC_EXTERN PetscErrorCode DMNetworkGetVertexLocalToGlobalOrdering(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSFGetSubSF(PetscSF, ISLocalToGlobalMapping, PetscSF *);
PETSC_EXTERN PetscErrorCode DMNetworkDistribute(DM *, PetscInt);
PETSC_EXTERN PetscErrorCode DMNetworkGetSupportingEdges(DM, PetscInt, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMNetworkGetConnectedVertices(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMNetworkIsGhostVertex(DM, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMNetworkIsSharedVertex(DM, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMNetworkEdgeSetMatrix(DM, PetscInt, Mat[]);
PETSC_EXTERN PetscErrorCode DMNetworkVertexSetMatrix(DM, PetscInt, Mat[]);
PETSC_EXTERN PetscErrorCode DMNetworkHasJacobian(DM, PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode DMNetworkGetPlex(DM, DM *);
PETSC_EXTERN PetscErrorCode DMNetworkGetGlobalEdgeIndex(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetGlobalVertexIndex(DM, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMNetworkAddSubnetwork(DM, const char *, PetscInt, PetscInt[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMNetworkGetSubnetwork(DM, PetscInt, PetscInt *, PetscInt *, const PetscInt **, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMNetworkAddSharedVertices(DM, PetscInt, PetscInt, PetscInt, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode DMNetworkGetSharedVertices(DM, PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMNetworkSharedVertexGetInfo(DM, PetscInt, PetscInt *, PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMNetworkCreateIS(DM, PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscInt *[], IS *);
PETSC_EXTERN PetscErrorCode DMNetworkCreateLocalIS(DM, PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscInt *[], IS *);

/*S
  DMNetworkMonitorList - Linked-list node held by a `DMNetworkMonitor`; each node records a `PetscViewer` and the subset of a global `Vec` that should be plotted for one network element

  Level: developer

.seealso: `DM`, `DMNETWORK`, `DMNetworkMonitor`, `DMNetworkMonitorAdd()`, `DMNetworkMonitorView()`
S*/
typedef struct _n_DMNetworkMonitorList *DMNetworkMonitorList;
struct _n_DMNetworkMonitorList {
  PetscViewer          viewer;
  Vec                  v;
  PetscInt             element;
  PetscInt             nodes;
  PetscInt             start;
  PetscInt             blocksize;
  DMNetworkMonitorList next;
};

/*S
  DMNetworkMonitor - Lightweight collection of `DMNetworkMonitorList` nodes used to drive per-element visualization of a `DMNETWORK` solution across time integration

  Level: developer

.seealso: `DM`, `DMNETWORK`, `DMNetworkMonitorCreate()`, `DMNetworkMonitorDestroy()`, `DMNetworkMonitorAdd()`, `DMNetworkMonitorView()`
S*/
typedef struct _n_DMNetworkMonitor *DMNetworkMonitor;
struct _n_DMNetworkMonitor {
  MPI_Comm             comm;
  DM                   network;
  DMNetworkMonitorList firstnode;
};

PETSC_EXTERN PetscErrorCode DMNetworkMonitorCreate(DM, DMNetworkMonitor *);
PETSC_EXTERN PetscErrorCode DMNetworkMonitorDestroy(DMNetworkMonitor *);
PETSC_EXTERN PetscErrorCode DMNetworkMonitorPop(DMNetworkMonitor);
PETSC_EXTERN PetscErrorCode DMNetworkMonitorAdd(DMNetworkMonitor, const char *, PetscInt, PetscInt, PetscInt, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, PetscBool);
PETSC_EXTERN PetscErrorCode DMNetworkMonitorView(DMNetworkMonitor, Vec);

PETSC_EXTERN PetscErrorCode DMNetworkViewSetShowRanks(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMNetworkViewSetViewRanks(DM, IS);
PETSC_EXTERN PetscErrorCode DMNetworkViewSetShowGlobal(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMNetworkViewSetShowVertices(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMNetworkViewSetShowNumbering(DM, PetscBool);
