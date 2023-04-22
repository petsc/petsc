#define PETSCDM_DLL
#include <petsc/private/dmnetworkimpl.h> /*I   "petscdmnetwork.h"   I*/
#include <petsc/private/vecimpl.h>

PetscErrorCode DMSetFromOptions_Network(DM dm, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMNetwork Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* External function declarations here */
extern PetscErrorCode DMCreateMatrix_Network(DM, Mat *);
extern PetscErrorCode DMDestroy_Network(DM);
extern PetscErrorCode DMView_Network(DM, PetscViewer);
extern PetscErrorCode DMGlobalToLocalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMGlobalToLocalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMSetUp_Network(DM);
extern PetscErrorCode DMClone_Network(DM, DM *);
extern PetscErrorCode DMCreateCoordinateDM_Network(DM, DM *);

static PetscErrorCode VecArrayPrint_private(PetscViewer viewer, PetscInt n, const PetscScalar *xv)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscImaginaryPart(xv[i]) > 0.0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "    %g + %g i\n", (double)PetscRealPart(xv[i]), (double)PetscImaginaryPart(xv[i])));
    } else if (PetscImaginaryPart(xv[i]) < 0.0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "    %g - %g i\n", (double)PetscRealPart(xv[i]), -(double)PetscImaginaryPart(xv[i])));
    } else PetscCall(PetscViewerASCIIPrintf(viewer, "    %g\n", (double)PetscRealPart(xv[i])));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "    %g\n", (double)xv[i]));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Network_Seq(DM networkdm, Vec X, PetscViewer viewer)
{
  PetscInt           e, v, Start, End, offset, nvar, id;
  const PetscScalar *xv;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &xv));

  /* iterate over edges */
  PetscCall(DMNetworkGetEdgeRange(networkdm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(networkdm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &offset));
    PetscCall(DMNetworkGetGlobalEdgeIndex(networkdm, e, &id));

    PetscCall(PetscViewerASCIIPrintf(viewer, "  Edge %" PetscInt_FMT ":\n", id));
    PetscCall(VecArrayPrint_private(viewer, nvar, xv + offset));
  }

  /* iterate over vertices */
  PetscCall(DMNetworkGetVertexRange(networkdm, &Start, &End));
  for (v = Start; v < End; v++) {
    PetscCall(DMNetworkGetComponent(networkdm, v, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm, v, ALL_COMPONENTS, &offset));
    PetscCall(DMNetworkGetGlobalVertexIndex(networkdm, v, &id));

    PetscCall(PetscViewerASCIIPrintf(viewer, "  Vertex %" PetscInt_FMT ":\n", id));
    PetscCall(VecArrayPrint_private(viewer, nvar, xv + offset));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecRestoreArrayRead(X, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Network_MPI(DM networkdm, Vec X, PetscViewer viewer)
{
  PetscInt           i, e, v, eStart, eEnd, vStart, vEnd, offset, nvar, len_loc, len, k;
  const PetscScalar *xv;
  MPI_Comm           comm;
  PetscMPIInt        size, rank, tag = ((PetscObject)viewer)->tag;
  Vec                localX;
  PetscBool          ghostvtex;
  PetscScalar       *values;
  PetscInt           j, ne, nv, id;
  MPI_Status         status;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)networkdm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMGetLocalVector(networkdm, &localX));
  PetscCall(DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX));
  PetscCall(VecGetArrayRead(localX, &xv));

  PetscCall(VecGetLocalSize(localX, &len_loc));

  PetscCall(DMNetworkGetEdgeRange(networkdm, &eStart, &eEnd));
  PetscCall(DMNetworkGetVertexRange(networkdm, &vStart, &vEnd));
  len_loc += 2 * (1 + eEnd - eStart + vEnd - vStart);

  /* values = [nedges, nvertices; id, nvar, xedge; ...; id, nvars, xvertex;...], to be sent to proc[0] */
  PetscCall(MPIU_Allreduce(&len_loc, &len, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscCalloc1(len, &values));

  if (rank == 0) PetscCall(PetscViewerASCIIPrintf(viewer, "Process [%d]\n", rank));

  /* iterate over edges */
  k = 2;
  for (e = eStart; e < eEnd; e++) {
    PetscCall(DMNetworkGetComponent(networkdm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm, e, ALL_COMPONENTS, &offset));
    PetscCall(DMNetworkGetGlobalEdgeIndex(networkdm, e, &id));

    if (rank == 0) { /* print its own entries */
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Edge %" PetscInt_FMT ":\n", id));
      PetscCall(VecArrayPrint_private(viewer, nvar, xv + offset));
    } else {
      values[0] += 1; /* number of edges */
      values[k++] = id;
      values[k++] = nvar;
      for (i = offset; i < offset + nvar; i++) values[k++] = xv[i];
    }
  }

  /* iterate over vertices */
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm, v, &ghostvtex));
    if (ghostvtex) continue;
    PetscCall(DMNetworkGetComponent(networkdm, v, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm, v, ALL_COMPONENTS, &offset));
    PetscCall(DMNetworkGetGlobalVertexIndex(networkdm, v, &id));

    if (rank == 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Vertex %" PetscInt_FMT ":\n", id));
      PetscCall(VecArrayPrint_private(viewer, nvar, xv + offset));
    } else {
      values[1] += 1; /* number of vertices */
      values[k++] = id;
      values[k++] = nvar;
      for (i = offset; i < offset + nvar; i++) values[k++] = xv[i];
    }
  }

  if (rank == 0) {
    /* proc[0] receives and prints messages */
    for (j = 1; j < size; j++) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Process [%" PetscInt_FMT "]\n", j));

      PetscCallMPI(MPI_Recv(values, (PetscMPIInt)len, MPIU_SCALAR, j, tag, comm, &status));

      ne = (PetscInt)PetscAbsScalar(values[0]);
      nv = (PetscInt)PetscAbsScalar(values[1]);

      /* print received edges */
      k = 2;
      for (i = 0; i < ne; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Edge %" PetscInt_FMT ":\n", id));
        PetscCall(VecArrayPrint_private(viewer, nvar, values + k));
        k += nvar;
      }

      /* print received vertices */
      for (i = 0; i < nv; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Vertex %" PetscInt_FMT ":\n", id));
        PetscCall(VecArrayPrint_private(viewer, nvar, values + k));
        k += nvar;
      }
    }
  } else {
    /* sends values to proc[0] */
    PetscCallMPI(MPI_Send((void *)values, k, MPIU_SCALAR, 0, tag, comm));
  }

  PetscCall(PetscFree(values));
  PetscCall(VecRestoreArrayRead(localX, &xv));
  PetscCall(DMRestoreLocalVector(networkdm, &localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

PetscErrorCode VecView_Network(Vec v, PetscViewer viewer)
{
  DM        dm;
  PetscBool isseq;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)v, VECSEQ, &isseq));

  /* Use VecView_Network if the viewer is ASCII; use VecView_Seq/MPI for other viewer formats */
  if (iascii) {
    if (isseq) PetscCall(VecView_Network_Seq(dm, v, viewer));
    else PetscCall(VecView_Network_MPI(dm, v, viewer));
  } else {
    if (isseq) PetscCall(VecView_Seq(v, viewer));
    else PetscCall(VecView_MPI(v, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateGlobalVector_Network(DM dm, Vec *vec)
{
  DM_Network *network = (DM_Network *)dm->data;

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(network->plex, vec));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecView_Network));
  PetscCall(VecSetDM(*vec, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateLocalVector_Network(DM dm, Vec *vec)
{
  DM_Network *network = (DM_Network *)dm->data;

  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector(network->plex, vec));
  PetscCall(VecSetDM(*vec, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMNetworkInitializeToDefault_NonShared(DM dm)
{
  DM_Network *network = (DM_Network *)dm->data;

  PetscFunctionBegin;
  network->Je                 = NULL;
  network->Jv                 = NULL;
  network->Jvptr              = NULL;
  network->userEdgeJacobian   = PETSC_FALSE;
  network->userVertexJacobian = PETSC_FALSE;

  network->vertex.DofSection       = NULL;
  network->vertex.GlobalDofSection = NULL;
  network->vertex.mapping          = NULL;
  network->vertex.sf               = NULL;

  network->edge.DofSection       = NULL;
  network->edge.GlobalDofSection = NULL;
  network->edge.mapping          = NULL;
  network->edge.sf               = NULL;

  network->DataSection      = NULL;
  network->DofSection       = NULL;
  network->GlobalDofSection = NULL;
  network->componentsetup   = PETSC_FALSE;

  network->plex = NULL;

  network->component  = NULL;
  network->ncomponent = 0;

  network->header             = NULL;
  network->cvalue             = NULL;
  network->componentdataarray = NULL;

  network->max_comps_registered = DMNETWORK_MAX_COMP_REGISTERED_DEFAULT; /* return to default */
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* Default values for the parameters in DMNetwork */
PetscErrorCode DMNetworkInitializeToDefault(DM dm)
{
  DM_Network          *network     = (DM_Network *)dm->data;
  DMNetworkCloneShared cloneshared = network->cloneshared;

  PetscFunctionBegin;
  PetscCall(DMNetworkInitializeToDefault_NonShared(dm));
  /* Default values for shared data */
  cloneshared->refct            = 1;
  cloneshared->NVertices        = 0;
  cloneshared->NEdges           = 0;
  cloneshared->nVertices        = 0;
  cloneshared->nEdges           = 0;
  cloneshared->nsubnet          = 0;
  cloneshared->pStart           = -1;
  cloneshared->pEnd             = -1;
  cloneshared->vStart           = -1;
  cloneshared->vEnd             = -1;
  cloneshared->eStart           = -1;
  cloneshared->eEnd             = -1;
  cloneshared->vltog            = NULL;
  cloneshared->distributecalled = PETSC_FALSE;

  cloneshared->subnet     = NULL;
  cloneshared->subnetvtx  = NULL;
  cloneshared->subnetedge = NULL;
  cloneshared->svtx       = NULL;
  cloneshared->nsvtx      = 0;
  cloneshared->Nsvtx      = 0;
  cloneshared->svertices  = NULL;
  cloneshared->sedgelist  = NULL;
  cloneshared->svtable    = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMInitialize_Network(DM dm)
{
  PetscFunctionBegin;
  PetscCall(DMSetDimension(dm, 1));
  dm->ops->view                    = DMView_Network;
  dm->ops->setfromoptions          = DMSetFromOptions_Network;
  dm->ops->clone                   = DMClone_Network;
  dm->ops->setup                   = DMSetUp_Network;
  dm->ops->createglobalvector      = DMCreateGlobalVector_Network;
  dm->ops->createlocalvector       = DMCreateLocalVector_Network;
  dm->ops->getlocaltoglobalmapping = NULL;
  dm->ops->createfieldis           = NULL;
  dm->ops->createcoordinatedm      = DMCreateCoordinateDM_Network;
  dm->ops->getcoloring             = NULL;
  dm->ops->creatematrix            = DMCreateMatrix_Network;
  dm->ops->createinterpolation     = NULL;
  dm->ops->createinjection         = NULL;
  dm->ops->refine                  = NULL;
  dm->ops->coarsen                 = NULL;
  dm->ops->refinehierarchy         = NULL;
  dm->ops->coarsenhierarchy        = NULL;
  dm->ops->globaltolocalbegin      = DMGlobalToLocalBegin_Network;
  dm->ops->globaltolocalend        = DMGlobalToLocalEnd_Network;
  dm->ops->localtoglobalbegin      = DMLocalToGlobalBegin_Network;
  dm->ops->localtoglobalend        = DMLocalToGlobalEnd_Network;
  dm->ops->destroy                 = DMDestroy_Network;
  dm->ops->createsubdm             = NULL;
  dm->ops->locatepoints            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
  copies over the subnetid and index portions of the DMNetworkComponentHeader from original dm to the newdm
*/
static PetscErrorCode DMNetworkCopyHeaderTopological(DM dm, DM newdm)
{
  DM_Network *network = (DM_Network *)dm->data, *newnetwork = (DM_Network *)newdm->data;
  PetscInt    p, i, np, index, subnetid;

  PetscFunctionBegin;
  np = network->cloneshared->pEnd - network->cloneshared->pStart;
  PetscCall(PetscCalloc2(np, &newnetwork->header, np, &newnetwork->cvalue));
  for (i = 0; i < np; i++) {
    p = i + network->cloneshared->pStart;
    PetscCall(DMNetworkGetSubnetID(dm, p, &subnetid));
    PetscCall(DMNetworkGetIndex(dm, p, &index));
    newnetwork->header[i].index        = index;
    newnetwork->header[i].subnetid     = subnetid;
    newnetwork->header[i].size         = NULL;
    newnetwork->header[i].key          = NULL;
    newnetwork->header[i].offset       = NULL;
    newnetwork->header[i].nvar         = NULL;
    newnetwork->header[i].offsetvarrel = NULL;
    newnetwork->header[i].ndata        = 0;
    newnetwork->header[i].maxcomps     = DMNETWORK_MAX_COMP_AT_POINT_DEFAULT;
    newnetwork->header[i].hsize        = sizeof(struct _p_DMNetworkComponentHeader) / sizeof(sizeof(DMNetworkComponentGenericDataType));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMClone_Network(DM dm, DM *newdm)
{
  DM_Network *network = (DM_Network *)dm->data, *newnetwork = NULL;

  PetscFunctionBegin;
  network->cloneshared->refct++;
  PetscCall(PetscNew(&newnetwork));
  (*newdm)->data = newnetwork;
  PetscCall(DMNetworkInitializeToDefault_NonShared(*newdm));
  newnetwork->cloneshared = network->cloneshared; /* Share all data that can be cloneshared */

  PetscCheck(network->plex, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "Must call DMNetworkLayoutSetUp() first");
  PetscCall(DMClone(network->plex, &newnetwork->plex));
  PetscCall(DMNetworkCopyHeaderTopological(dm, *newdm));
  PetscCall(DMNetworkInitializeNonTopological(*newdm)); /* initialize all non-topological data to the state after DMNetworkLayoutSetUp as been called */
  PetscCall(PetscObjectChangeTypeName((PetscObject)*newdm, DMNETWORK));
  PetscCall(DMInitialize_Network(*newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Developer Note: Be aware that the plex inside of the network does not have a coordinate plex.
*/
PetscErrorCode DMCreateCoordinateDM_Network(DM dm, DM *cdm)
{
  DM_Network *newnetwork = NULL;
  PetscInt    Nf;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(DMClone(dm, cdm));
  newnetwork = (DM_Network *)(*cdm)->data;
  PetscCall(DMGetNumFields(newnetwork->plex, &Nf));
  PetscCall(DMSetNumFields(*cdm, Nf)); /* consistency with the coordinate plex */
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*cdm, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*cdm, "cdm_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMNETWORK = "network" - A DM object that encapsulates an unstructured network. The implementation is based on the DM object
                          DMPlex that manages unstructured grids. Distributed networks use a non-overlapping partitioning of
                          the edges. In the local representation, Vecs contain all unknowns in the interior and shared boundary.
                          This is specified by a PetscSection object. Ownership in the global representation is determined by
                          ownership of the underlying DMPlex points. This is specified by another PetscSection object.

  Level: intermediate

.seealso: `DMType`, `DMNetworkCreate()`, `DMCreate()`, `DMSetType()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Network(DM dm)
{
  DM_Network *network;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscNew(&network));
  PetscCall(PetscNew(&network->cloneshared));
  dm->data = network;

  PetscCall(DMNetworkInitializeToDefault(dm));
  PetscCall(DMInitialize_Network(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMNetworkCreate - Creates a DMNetwork object, which encapsulates an unstructured network.

  Collective

  Input Parameter:
. comm - The communicator for the DMNetwork object

  Output Parameter:
. network  - The DMNetwork object

  Level: beginner

@*/
PetscErrorCode DMNetworkCreate(MPI_Comm comm, DM *network)
{
  PetscFunctionBegin;
  PetscValidPointer(network, 2);
  PetscCall(DMCreate(comm, network));
  PetscCall(DMSetType(*network, DMNETWORK));
  PetscFunctionReturn(PETSC_SUCCESS);
}
