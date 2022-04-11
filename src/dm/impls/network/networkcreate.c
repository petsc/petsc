#define PETSCDM_DLL
#include <petsc/private/dmnetworkimpl.h>    /*I   "petscdmnetwork.h"   I*/
#include <petsc/private/vecimpl.h>

PetscErrorCode  DMSetFromOptions_Network(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscOptionsHeadBegin(PetscOptionsObject,"DMNetwork Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateMatrix_Network(DM, Mat*);
extern PetscErrorCode DMDestroy_Network(DM);
extern PetscErrorCode DMView_Network(DM, PetscViewer);
extern PetscErrorCode DMGlobalToLocalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMGlobalToLocalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMSetUp_Network(DM);
extern PetscErrorCode DMClone_Network(DM, DM*);

static PetscErrorCode VecArrayPrint_private(PetscViewer viewer,PetscInt n,const PetscScalar *xv)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscImaginaryPart(xv[i]) > 0.0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"    %g + %g i\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i])));
    } else if (PetscImaginaryPart(xv[i]) < 0.0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"    %g - %g i\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i])));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"    %g\n",(double)PetscRealPart(xv[i])));
    }
#else
    PetscCall(PetscViewerASCIIPrintf(viewer,"    %g\n",(double)xv[i]));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Network_Seq(DM networkdm,Vec X,PetscViewer viewer)
{
  PetscInt          e,v,Start,End,offset,nvar,id;
  const PetscScalar *xv;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X,&xv));

  /* iterate over edges */
  PetscCall(DMNetworkGetEdgeRange(networkdm,&Start,&End));
  for (e=Start; e<End; e++) {
    PetscCall(DMNetworkGetComponent(networkdm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&offset));
    PetscCall(DMNetworkGetGlobalEdgeIndex(networkdm,e,&id));

    PetscCall(PetscViewerASCIIPrintf(viewer,"  Edge %" PetscInt_FMT ":\n",id));
    PetscCall(VecArrayPrint_private(viewer,nvar,xv+offset));
  }

  /* iterate over vertices */
  PetscCall(DMNetworkGetVertexRange(networkdm,&Start,&End));
  for (v=Start; v<End; v++) {
    PetscCall(DMNetworkGetComponent(networkdm,v,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&offset));
    PetscCall(DMNetworkGetGlobalVertexIndex(networkdm,v,&id));

    PetscCall(PetscViewerASCIIPrintf(viewer,"  Vertex %" PetscInt_FMT ":\n",id));
    PetscCall(VecArrayPrint_private(viewer,nvar,xv+offset));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(VecRestoreArrayRead(X,&xv));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Network_MPI(DM networkdm,Vec X,PetscViewer viewer)
{
  PetscInt          i,e,v,eStart,eEnd,vStart,vEnd,offset,nvar,len_loc,len,k;
  const PetscScalar *xv;
  MPI_Comm          comm;
  PetscMPIInt       size,rank,tag = ((PetscObject)viewer)->tag;
  Vec               localX;
  PetscBool         ghostvtex;
  PetscScalar       *values;
  PetscInt          j,ne,nv,id;
  MPI_Status        status;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)networkdm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(DMGetLocalVector(networkdm,&localX));
  PetscCall(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));
  PetscCall(VecGetArrayRead(localX,&xv));

  PetscCall(VecGetLocalSize(localX,&len_loc));

  PetscCall(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  PetscCall(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
  len_loc += 2*(1 + eEnd-eStart + vEnd-vStart);

  /* values = [nedges, nvertices; id, nvar, xedge; ...; id, nvars, xvertex;...], to be sent to proc[0] */
  PetscCallMPI(MPI_Allreduce(&len_loc,&len,1,MPIU_INT,MPI_MAX,comm));
  PetscCall(PetscCalloc1(len,&values));

  if (rank == 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank));
  }

  /* iterate over edges */
  k = 2;
  for (e=eStart; e<eEnd; e++) {
    PetscCall(DMNetworkGetComponent(networkdm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&offset));
    PetscCall(DMNetworkGetGlobalEdgeIndex(networkdm,e,&id));

    if (rank == 0) { /* print its own entries */
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Edge %" PetscInt_FMT ":\n",id));
      PetscCall(VecArrayPrint_private(viewer,nvar,xv+offset));
    } else {
      values[0]  += 1; /* number of edges */
      values[k++] = id;
      values[k++] = nvar;
      for (i=offset; i< offset+nvar; i++) values[k++] = xv[i];
    }
  }

  /* iterate over vertices */
  for (v=vStart; v<vEnd; v++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm,v,&ghostvtex));
    if (ghostvtex) continue;
    PetscCall(DMNetworkGetComponent(networkdm,v,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&offset));
    PetscCall(DMNetworkGetGlobalVertexIndex(networkdm,v,&id));

    if (rank == 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Vertex %" PetscInt_FMT ":\n",id));
      PetscCall(VecArrayPrint_private(viewer,nvar,xv+offset));
    } else {
      values[1]  += 1; /* number of vertices */
      values[k++] = id;
      values[k++] = nvar;
      for (i=offset; i< offset+nvar; i++) values[k++] = xv[i];
    }
  }

  if (rank == 0) {
    /* proc[0] receives and prints messages */
    for (j=1; j<size; j++) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Process [%d]\n",j));

      PetscCallMPI(MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,comm,&status));

      ne = (PetscInt)PetscAbsScalar(values[0]);
      nv = (PetscInt)PetscAbsScalar(values[1]);

      /* print received edges */
      k = 2;
      for (i=0; i<ne; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Edge %" PetscInt_FMT ":\n",id));
        PetscCall(VecArrayPrint_private(viewer,nvar,values+k));
        k   += nvar;
      }

      /* print received vertices */
      for (i=0; i<nv; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Vertex %" PetscInt_FMT ":\n",id));
        PetscCall(VecArrayPrint_private(viewer,nvar,values+k));
        k   += nvar;
      }
    }
  } else {
    /* sends values to proc[0] */
    PetscCallMPI(MPI_Send((void*)values,k,MPIU_SCALAR,0,tag,comm));
  }

  PetscCall(PetscFree(values));
  PetscCall(VecRestoreArrayRead(localX,&xv));
  PetscCall(DMRestoreLocalVector(networkdm,&localX));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec,PetscViewer);

PetscErrorCode VecView_Network(Vec v,PetscViewer viewer)
{
  DM             dm;
  PetscBool      isseq;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v,&dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v),PETSC_ERR_ARG_WRONG,"Vector not generated from a DM");
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)v,VECSEQ,&isseq));

  /* Use VecView_Network if the viewer is ASCII; use VecView_Seq/MPI for other viewer formats */
  if (iascii) {
    if (isseq) {
      PetscCall(VecView_Network_Seq(dm,v,viewer));
    } else {
      PetscCall(VecView_Network_MPI(dm,v,viewer));
    }
  } else {
    if (isseq) {
      PetscCall(VecView_Seq(v,viewer));
    } else {
      PetscCall(VecView_MPI(v,viewer));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Network(DM dm,Vec *vec)
{
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(network->plex,vec));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Network));
  PetscCall(VecSetDM(*vec,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Network(DM dm,Vec *vec)
{
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector(network->plex,vec));
  PetscCall(VecSetDM(*vec,dm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMInitialize_Network(DM dm)
{
  PetscFunctionBegin;
  PetscCall(DMSetDimension(dm,1));
  dm->ops->view                            = DMView_Network;
  dm->ops->setfromoptions                  = DMSetFromOptions_Network;
  dm->ops->clone                           = DMClone_Network;
  dm->ops->setup                           = DMSetUp_Network;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Network;
  dm->ops->createlocalvector               = DMCreateLocalVector_Network;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = DMCreateMatrix_Network;
  dm->ops->createinterpolation             = NULL;
  dm->ops->createinjection                 = NULL;
  dm->ops->refine                          = NULL;
  dm->ops->coarsen                         = NULL;
  dm->ops->refinehierarchy                 = NULL;
  dm->ops->coarsenhierarchy                = NULL;
  dm->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Network;
  dm->ops->globaltolocalend                = DMGlobalToLocalEnd_Network;
  dm->ops->localtoglobalbegin              = DMLocalToGlobalBegin_Network;
  dm->ops->localtoglobalend                = DMLocalToGlobalEnd_Network;
  dm->ops->destroy                         = DMDestroy_Network;
  dm->ops->createsubdm                     = NULL;
  dm->ops->locatepoints                    = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMClone_Network(DM dm, DM *newdm)
{
  DM_Network     *network = (DM_Network *) dm->data;

  PetscFunctionBegin;
  network->refct++;
  (*newdm)->data = network;
  PetscCall(PetscObjectChangeTypeName((PetscObject) *newdm, DMNETWORK));
  PetscCall(DMInitialize_Network(*newdm));
  PetscFunctionReturn(0);
}

/*MC
  DMNETWORK = "network" - A DM object that encapsulates an unstructured network. The implementation is based on the DM object
                          DMPlex that manages unstructured grids. Distributed networks use a non-overlapping partitioning of
                          the edges. In the local representation, Vecs contain all unknowns in the interior and shared boundary.
                          This is specified by a PetscSection object. Ownership in the global representation is determined by
                          ownership of the underlying DMPlex points. This is specified by another PetscSection object.

  Level: intermediate

.seealso: DMType, DMNetworkCreate(), DMCreate(), DMSetType()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Network(DM dm)
{
  DM_Network     *network;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscNewLog(dm,&network));
  dm->data = network;

  network->refct     = 1;
  network->NVertices = 0;
  network->NEdges    = 0;
  network->nVertices = 0;
  network->nEdges    = 0;
  network->nsubnet   = 0;

  network->max_comps_registered = 20;
  network->component            = NULL;
  network->header               = NULL;
  network->cvalue               = NULL;

  PetscCall(DMInitialize_Network(dm));
  PetscFunctionReturn(0);
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
  PetscValidPointer(network,2);
  PetscCall(DMCreate(comm, network));
  PetscCall(DMSetType(*network, DMNETWORK));
  PetscFunctionReturn(0);
}
