#define PETSCDM_DLL
#include <petsc/private/dmnetworkimpl.h>    /*I   "petscdmnetwork.h"   I*/
#include <petsc/private/vecimpl.h>

PetscErrorCode  DMSetFromOptions_Network(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMNetwork Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
    if (PetscImaginaryPart(xv[i]) > 0.0) {
      ierr = PetscViewerASCIIPrintf(viewer,"    %g + %g i\n",(double)PetscRealPart(xv[i]),(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
    } else if (PetscImaginaryPart(xv[i]) < 0.0) {
      ierr = PetscViewerASCIIPrintf(viewer,"    %g - %g i\n",(double)PetscRealPart(xv[i]),-(double)PetscImaginaryPart(xv[i]));CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"    %g\n",(double)PetscRealPart(xv[i]));CHKERRQ(ierr);
    }
#else
    ierr = PetscViewerASCIIPrintf(viewer,"    %g\n",(double)xv[i]);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Network_Seq(DM networkdm,Vec X,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          e,v,Start,End,offset,nvar,id;
  const PetscScalar *xv;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&xv);CHKERRQ(ierr);

  /* iterate over edges */
  ierr = DMNetworkGetEdgeRange(networkdm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetComponent(networkdm,e,ALL_COMPONENTS,NULL,NULL,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalEdgeIndex(networkdm,e,&id);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  Edge %D:\n",id);CHKERRQ(ierr);
    ierr = VecArrayPrint_private(viewer,nvar,xv+offset);CHKERRQ(ierr);
  }

  /* iterate over vertices */
  ierr = DMNetworkGetVertexRange(networkdm,&Start,&End);CHKERRQ(ierr);
  for (v=Start; v<End; v++) {
    ierr = DMNetworkGetComponent(networkdm,v,ALL_COMPONENTS,NULL,NULL,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,v,&id);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  Vertex %D:\n",id);CHKERRQ(ierr);
    ierr = VecArrayPrint_private(viewer,nvar,xv+offset);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&xv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Network_MPI(DM networkdm,Vec X,PetscViewer viewer)
{
  PetscErrorCode    ierr;
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
  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localX,&xv);CHKERRQ(ierr);

  ierr = VecGetLocalSize(localX,&len_loc);CHKERRQ(ierr);

  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
  len_loc += 2*(1 + eEnd-eStart + vEnd-vStart);

  /* values = [nedges, nvertices; id, nvar, xedge; ...; id, nvars, xvertex;...], to be sent to proc[0] */
  ierr = MPI_Allreduce(&len_loc,&len,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
  ierr = PetscCalloc1(len,&values);CHKERRQ(ierr);

  if (rank == 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Process [%d]\n",rank);CHKERRQ(ierr);
  }

  /* iterate over edges */
  k = 2;
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponent(networkdm,e,ALL_COMPONENTS,NULL,NULL,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalEdgeIndex(networkdm,e,&id);CHKERRQ(ierr);

    if (rank == 0) { /* print its own entries */
      ierr = PetscViewerASCIIPrintf(viewer,"  Edge %D:\n",id);CHKERRQ(ierr);
      ierr = VecArrayPrint_private(viewer,nvar,xv+offset);CHKERRQ(ierr);
    } else {
      values[0]  += 1; /* number of edges */
      values[k++] = id;
      values[k++] = nvar;
      for (i=offset; i< offset+nvar; i++) values[k++] = xv[i];
    }
  }

  /* iterate over vertices */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;
    ierr = DMNetworkGetComponent(networkdm,v,ALL_COMPONENTS,NULL,NULL,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,v,&id);CHKERRQ(ierr);

    if (rank == 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Vertex %D:\n",id);CHKERRQ(ierr);
      ierr = VecArrayPrint_private(viewer,nvar,xv+offset);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"Process [%d]\n",j);CHKERRQ(ierr);

      ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,comm,&status);CHKERRMPI(ierr);

      ne = (PetscInt)PetscAbsScalar(values[0]);
      nv = (PetscInt)PetscAbsScalar(values[1]);

      /* print received edges */
      k = 2;
      for (i=0; i<ne; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        ierr = PetscViewerASCIIPrintf(viewer,"  Edge %D:\n",id);CHKERRQ(ierr);
        ierr = VecArrayPrint_private(viewer,nvar,values+k);CHKERRQ(ierr);
        k   += nvar;
      }

      /* print received vertices */
      for (i=0; i<nv; i++) {
        id   = (PetscInt)PetscAbsScalar(values[k++]);
        nvar = (PetscInt)PetscAbsScalar(values[k++]);
        ierr = PetscViewerASCIIPrintf(viewer,"  Vertex %D:\n",id);CHKERRQ(ierr);
        ierr = VecArrayPrint_private(viewer,nvar,values+k);CHKERRQ(ierr);
        k   += nvar;
      }
    }
  } else {
    /* sends values to proc[0] */
    ierr = MPI_Send((void*)values,k,MPIU_SCALAR,0,tag,comm);CHKERRMPI(ierr);
  }

  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localX,&xv);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec,PetscViewer);

PetscErrorCode VecView_Network(Vec v,PetscViewer viewer)
{
  DM             dm;
  PetscErrorCode ierr;
  PetscBool      isseq;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_ARG_WRONG,"Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)v,VECSEQ,&isseq);CHKERRQ(ierr);

  /* Use VecView_Network if the viewer is ASCII; use VecView_Seq/MPI for other viewer formats */
  if (iascii) {
    if (isseq) {
      ierr = VecView_Network_Seq(dm,v,viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView_Network_MPI(dm,v,viewer);CHKERRQ(ierr);
    }
  } else {
    if (isseq) {
      ierr = VecView_Seq(v,viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView_MPI(v,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Network(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(network->plex,vec);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void)) VecView_Network);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Network(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(network->plex,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMInitialize_Network(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetDimension(dm,1);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  network->refct++;
  (*newdm)->data = network;
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, DMNETWORK);CHKERRQ(ierr);
  ierr = DMInitialize_Network(*newdm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&network);CHKERRQ(ierr);
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

  ierr = DMInitialize_Network(dm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(network,2);
  ierr = DMCreate(comm, network);CHKERRQ(ierr);
  ierr = DMSetType(*network, DMNETWORK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
