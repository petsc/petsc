#include <petsc/private/dmnetworkimpl.h>  /*I  "petscdmnetwork.h"  I*/

/*@
  DMNetworkGetPlex - Gets the Plex DM associated with this network DM

  Not collective

  Input Parameters:
+ netdm - the dm object
- plexmdm - the plex dm object

  Level: Advanced

.seealso: DMNetworkCreate()
@*/
PetscErrorCode DMNetworkGetPlex(DM netdm, DM *plexdm)
{
  DM_Network *network = (DM_Network*) netdm->data;

  PetscFunctionBegin;
  *plexdm = network->plex;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetSizes - Gets the the number of subnetworks and coupling subnetworks

  Collective on dm

  Input Parameters:
+ dm - the dm object
. Nsubnet - global number of subnetworks
- NsubnetCouple - global number of coupling subnetworks

  Level: beginner

.seealso: DMNetworkCreate()
@*/
PetscErrorCode DMNetworkGetSizes(DM netdm, PetscInt *Nsubnet, PetscInt *Ncsubnet)
{
  DM_Network *network = (DM_Network*) netdm->data;

  PetscFunctionBegin;
  *Nsubnet = network->nsubnet;
  *Ncsubnet = network->ncsubnet;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetSizes - Sets the number of subnetworks, local and global vertices and edges for each subnetwork.

  Collective on dm

  Input Parameters:
+ dm - the dm object
. Nsubnet - global number of subnetworks
. nV - number of local vertices for each subnetwork
. nE - number of local edges for each subnetwork
. NsubnetCouple - global number of coupling subnetworks
- nec - number of local edges for each coupling subnetwork

   You cannot change the sizes once they have been set. nV, nE are arrays of length Nsubnet, and nec is array of length NsubnetCouple.

   Level: beginner

.seealso: DMNetworkCreate()
@*/
PetscErrorCode DMNetworkSetSizes(DM dm,PetscInt Nsubnet,PetscInt nV[], PetscInt nE[],PetscInt NsubnetCouple,PetscInt nec[])
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*) dm->data;
  PetscInt       a[2],b[2],i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (Nsubnet <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of subnetworks %D cannot be less than 1",Nsubnet);
  if (NsubnetCouple < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of coupling subnetworks %D cannot be less than 0",NsubnetCouple);

  PetscValidLogicalCollectiveInt(dm,Nsubnet,2);
  if (NsubnetCouple) PetscValidLogicalCollectiveInt(dm,NsubnetCouple,5);
  if (network->nsubnet != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Network sizes alread set, cannot resize the network");

  if (!nV || !nE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Local vertex size or edge size must be provided");

  network->nsubnet  = Nsubnet + NsubnetCouple;
  network->ncsubnet = NsubnetCouple;
  ierr = PetscCalloc1(Nsubnet+NsubnetCouple,&network->subnet);CHKERRQ(ierr);

  /* ----------------------------------------------------------
   p=v or e; P=V or E
   subnet[0].pStart   = 0
   subnet[i+1].pStart = subnet[i].pEnd = subnet[i].pStart + (nE[i] or NV[i])
   ----------------------------------------------------------------------- */
  for (i=0; i < Nsubnet; i++) {
    /* Get global number of vertices and edges for subnet[i] */
    a[0] = nV[i]; a[1] = nE[i]; /* local number of vertices (excluding ghost) and edges */
    ierr = MPIU_Allreduce(a,b,2,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    network->subnet[i].Nvtx = b[0];
    network->subnet[i].Nedge = b[1];

    network->subnet[i].nvtx   = nV[i]; /* local nvtx, without ghost */

    /* global subnet[].vStart and vEnd, used by DMNetworkLayoutSetUp() */
    network->subnet[i].vStart = network->NVertices;
    network->subnet[i].vEnd   = network->subnet[i].vStart + network->subnet[i].Nvtx;

    network->nVertices += nV[i];
    network->NVertices += network->subnet[i].Nvtx;

    network->subnet[i].nedge  = nE[i];
    network->subnet[i].eStart = network->nEdges;
    network->subnet[i].eEnd   = network->subnet[i].eStart + nE[i];
    network->nEdges += nE[i];
    network->NEdges += network->subnet[i].Nedge;
  }

  /* coupling subnetwork */
  for (; i < Nsubnet+NsubnetCouple; i++) {
    /* Get global number of coupling edges for subnet[i] */
    ierr = MPIU_Allreduce(nec+(i-Nsubnet),&network->subnet[i].Nedge,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);

    network->subnet[i].nvtx   = 0; /* We design coupling subnetwork such that it does not have its own vertices */
    network->subnet[i].vStart = network->nVertices;
    network->subnet[i].vEnd   = network->subnet[i].vStart;

    network->subnet[i].nedge  = nec[i-Nsubnet];
    network->subnet[i].eStart = network->nEdges;
    network->subnet[i].eEnd = network->subnet[i].eStart + nec[i-Nsubnet];
    network->nEdges += nec[i-Nsubnet];
    network->NEdges += network->subnet[i].Nedge;
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetEdgeList - Sets the list of local edges (vertex connectivity) for the network

  Logically collective on dm

  Input Parameters:
+ dm - the dm object
. edgelist - list of edges for each subnetwork
- edgelistCouple - list of edges for each coupling subnetwork

  Notes:
  There is no copy involved in this operation, only the pointer is referenced. The edgelist should
  not be destroyed before the call to DMNetworkLayoutSetUp

  Level: beginner

  Example usage:
  Consider the following 2 separate networks and a coupling network:

.vb
 network 0: v0 -> v1 -> v2 -> v3
 network 1: v1 -> v2 -> v0
 coupling network: network 1: v2 -> network 0: v0
.ve

 The resulting input
   edgelist[0] = [0 1 | 1 2 | 2 3];
   edgelist[1] = [1 2 | 2 0]
   edgelistCouple[0] = [(network)1 (v)2 (network)0 (v)0].

.seealso: DMNetworkCreate, DMNetworkSetSizes
@*/
PetscErrorCode DMNetworkSetEdgeList(DM dm,PetscInt *edgelist[],PetscInt *edgelistCouple[])
{
  DM_Network *network = (DM_Network*) dm->data;
  PetscInt   i;

  PetscFunctionBegin;
  for (i=0; i < (network->nsubnet-network->ncsubnet); i++) network->subnet[i].edgelist = edgelist[i];
  if (network->ncsubnet) {
    PetscInt j = 0;
    if (!edgelistCouple) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Must provide edgelist_couple");
    while (i < network->nsubnet) network->subnet[i++].edgelist = edgelistCouple[j++];
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkLayoutSetUp - Sets up the bare layout (graph) for the network

  Collective on dm

  Input Parameters:
. DM - the dmnetwork object

  Notes:
  This routine should be called after the network sizes and edgelists have been provided. It creates
  the bare layout of the network and sets up the network to begin insertion of components.

  All the components should be registered before calling this routine.

  Level: beginner

.seealso: DMNetworkSetSizes, DMNetworkSetEdgeList
@*/
PetscErrorCode DMNetworkLayoutSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       numCorners=2,dim = 1; /* One dimensional network */
  PetscInt       i,j,ctr,nsubnet,*eowners,np,*edges,*subnetvtx,vStart;
  PetscInt       k,netid,vid, *vidxlTog,*edgelist_couple=NULL;
  const PetscInt *cone;
  MPI_Comm       comm;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* Create the local edgelist for the network by concatenating local input edgelists of the subnetworks */
  ierr = PetscCalloc1(2*network->nEdges,&edges);CHKERRQ(ierr);
  nsubnet = network->nsubnet - network->ncsubnet;
  ctr = 0;
  for (i=0; i < nsubnet; i++) {
    for (j = 0; j < network->subnet[i].nedge; j++) {
      edges[2*ctr]   = network->subnet[i].vStart + network->subnet[i].edgelist[2*j];
      edges[2*ctr+1] = network->subnet[i].vStart + network->subnet[i].edgelist[2*j+1];
      ctr++;
    }
  }

  /* Append local coupling edgelists of the subnetworks */
  i       = nsubnet; /* netid of coupling subnet */
  nsubnet = network->nsubnet;
  while (i < nsubnet) {
    edgelist_couple = network->subnet[i].edgelist;

    k = 0;
    for (j = 0; j < network->subnet[i].nedge; j++) {
      netid = edgelist_couple[k]; vid = edgelist_couple[k+1];
      edges[2*ctr] = network->subnet[netid].vStart + vid; k += 2;

      netid = edgelist_couple[k]; vid = edgelist_couple[k+1];
      edges[2*ctr+1] = network->subnet[netid].vStart + vid; k+=2;
      ctr++;
    }
    i++;
  }
  /*
  if (rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] edgelist:\n",rank);
    for(i=0; i < network->nEdges; i++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%D %D]",edges[2*i],edges[2*i+1]);CHKERRQ(ierr);
      printf("\n");
    }
  }
   */

  /* Create network->plex */
  ierr = DMCreate(comm,&network->plex);CHKERRQ(ierr);
  ierr = DMSetType(network->plex,DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(network->plex,dim);CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMPlexBuildFromCellList(network->plex,network->nEdges,network->nVertices,numCorners,edges);CHKERRQ(ierr);
  } else {
    ierr = DMPlexBuildFromCellListParallel(network->plex,network->nEdges,network->nVertices,network->NVertices,numCorners,edges,NULL);CHKERRQ(ierr);
  }

  ierr = DMPlexGetChart(network->plex,&network->pStart,&network->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,0,&network->eStart,&network->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,1,&network->vStart,&network->vEnd);CHKERRQ(ierr);
  vStart = network->vStart;

  ierr = PetscSectionCreate(comm,&network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm,&network->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DataSection,network->pStart,network->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DofSection,network->pStart,network->pEnd);CHKERRQ(ierr);

  network->dataheadersize = sizeof(struct _p_DMNetworkComponentHeader)/sizeof(DMNetworkComponentGenericDataType);
  np = network->pEnd - network->pStart;
  ierr = PetscCalloc2(np,&network->header,np,&network->cvalue);CHKERRQ(ierr);

  /* Create vidxlTog: maps local vertex index to global index */
  np = network->vEnd - vStart;
  ierr = PetscMalloc2(np,&vidxlTog,size+1,&eowners);CHKERRQ(ierr);
  ctr = 0;
  for (i=network->eStart; i<network->eEnd; i++) {
    ierr = DMNetworkGetConnectedVertices(dm,i,&cone);CHKERRQ(ierr);
    vidxlTog[cone[0] - vStart] = edges[2*ctr];
    vidxlTog[cone[1] - vStart] = edges[2*ctr+1];
    ctr++;
  }
  ierr = PetscFree(edges);CHKERRQ(ierr);

  /* Create vertices and edges array for the subnetworks */
  for (j=0; j < network->nsubnet; j++) {
    ierr = PetscCalloc1(network->subnet[j].nedge,&network->subnet[j].edges);CHKERRQ(ierr);

    /* Temporarily setting nvtx and nedge to 0 so we can use them as counters in the below for loop.
       These get updated when the vertices and edges are added. */
    network->subnet[j].nvtx  = 0;
    network->subnet[j].nedge = 0;
  }
  ierr = PetscCalloc1(np,&network->subnetvtx);CHKERRQ(ierr);


  /* Get edge ownership */
  np = network->eEnd - network->eStart;
  ierr = MPI_Allgather(&np,1,MPIU_INT,eowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  eowners[0] = 0;
  for (i=2; i<=size; i++) eowners[i] += eowners[i-1];

  i = 0; j = 0;
  while (i < np) { /* local edges, including coupling edges */
    network->header[i].index = i + eowners[rank];   /* Global edge index */

    if (j < network->nsubnet && i < network->subnet[j].eEnd) {
      network->header[i].subnetid = j; /* Subnetwork id */
      network->subnet[j].edges[network->subnet[j].nedge++] = i;

      network->header[i].ndata = 0;
      ierr = PetscSectionAddDof(network->DataSection,i,network->dataheadersize);CHKERRQ(ierr);
      network->header[i].offset[0] = 0;
      network->header[i].offsetvarrel[0] = 0;
      i++;
    }
    if (i >= network->subnet[j].eEnd) j++;
  }

  /* Count network->subnet[*].nvtx */
  for (i=vStart; i<network->vEnd; i++) { /* local vertices, including ghosts */
    k = vidxlTog[i-vStart];
    for (j=0; j < network->nsubnet; j++) {
      if (network->subnet[j].vStart <= k && k < network->subnet[j].vEnd) {
        network->subnet[j].nvtx++;
        break;
      }
    }
  }

  /* Set network->subnet[*].vertices on array network->subnetvtx */
  subnetvtx = network->subnetvtx;
  for (j=0; j<network->nsubnet; j++) {
    network->subnet[j].vertices = subnetvtx;
    subnetvtx                  += network->subnet[j].nvtx;
    network->subnet[j].nvtx = 0;
  }

  /* Set vertex array for the subnetworks */
  for (i=vStart; i<network->vEnd; i++) { /* local vertices, including ghosts */
    network->header[i].index = vidxlTog[i-vStart]; /*  Global vertex index */

    k = vidxlTog[i-vStart];
    for (j=0; j < network->nsubnet; j++) {
      if (network->subnet[j].vStart <= k && k < network->subnet[j].vEnd) {
        network->header[i].subnetid = j;
        network->subnet[j].vertices[network->subnet[j].nvtx++] = i;
        break;
      }
    }

    network->header[i].ndata = 0;
    ierr = PetscSectionAddDof(network->DataSection,i,network->dataheadersize);CHKERRQ(ierr);
    network->header[i].offset[0] = 0;
    network->header[i].offsetvarrel[0] = 0;
  }

  ierr = PetscFree2(vidxlTog,eowners);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSubnetworkInfo - Returns the info for the subnetwork

  Input Parameters:
+ dm - the DM object
- id   - the ID (integer) of the subnetwork

  Output Parameters:
+ nv    - number of vertices (local)
. ne    - number of edges (local)
. vtx   - local vertices for this subnetwork
- edge  - local edges for this subnetwork

  Notes:
  Cannot call this routine before DMNetworkLayoutSetup()

  Level: intermediate

.seealso: DMNetworkLayoutSetUp, DMNetworkCreate
@*/
PetscErrorCode DMNetworkGetSubnetworkInfo(DM dm,PetscInt id,PetscInt *nv, PetscInt *ne,const PetscInt **vtx, const PetscInt **edge)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (id >= network->nsubnet) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subnet ID %D exceeds the num of subnets %D",id,network->nsubnet);
  *nv   = network->subnet[id].nvtx;
  *ne   = network->subnet[id].nedge;
  *vtx  = network->subnet[id].vertices;
  *edge = network->subnet[id].edges;
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSubnetworkCoupleInfo - Returns the info for the coupling subnetwork

  Input Parameters:
+ dm - the DM object
- id   - the ID (integer) of the coupling subnetwork

  Output Parameters:
+ ne - number of edges (local)
- edge  - local edges for this coupling subnetwork

  Notes:
  Cannot call this routine before DMNetworkLayoutSetup()

  Level: intermediate

.seealso: DMNetworkGetSubnetworkInfo, DMNetworkLayoutSetUp, DMNetworkCreate
@*/
PetscErrorCode DMNetworkGetSubnetworkCoupleInfo(DM dm,PetscInt id,PetscInt *ne,const PetscInt **edge)
{
  DM_Network *net = (DM_Network*)dm->data;
  PetscInt   id1;

  PetscFunctionBegin;
  if (net->ncsubnet) {
    if (id >= net->ncsubnet) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subnet ID %D exceeds the num of coupling subnets %D",id,net->ncsubnet);

    id1   = id + net->nsubnet - net->ncsubnet;
    *ne   = net->subnet[id1].nedge;
    *edge = net->subnet[id1].edges;
  } else {
    *ne   = 0;
    *edge = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkRegisterComponent - Registers the network component

  Logically collective on dm

  Input Parameters:
+ dm   - the network object
. name - the component name
- size - the storage size in bytes for this component data

   Output Parameters:
.   key - an integer key that defines the component

   Notes
   This routine should be called by all processors before calling DMNetworkLayoutSetup().

   Level: beginner

.seealso: DMNetworkLayoutSetUp, DMNetworkCreate
@*/
PetscErrorCode DMNetworkRegisterComponent(DM dm,const char *name,size_t size,PetscInt *key)
{
  PetscErrorCode        ierr;
  DM_Network            *network = (DM_Network*) dm->data;
  DMNetworkComponent    *component=&network->component[network->ncomponent];
  PetscBool             flg=PETSC_FALSE;
  PetscInt              i;

  PetscFunctionBegin;
  for (i=0; i < network->ncomponent; i++) {
    ierr = PetscStrcmp(component->name,name,&flg);CHKERRQ(ierr);
    if (flg) {
      *key = i;
      PetscFunctionReturn(0);
    }
  }
  if(network->ncomponent == MAX_COMPONENTS) {
    SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Number of components registered exceeds the max %D",MAX_COMPONENTS);
  }

  ierr = PetscStrcpy(component->name,name);CHKERRQ(ierr);
  component->size = size/sizeof(DMNetworkComponentGenericDataType);
  *key = network->ncomponent;
  network->ncomponent++;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVertexRange - Get the bounds [start, end) for the vertices.

  Not Collective

  Input Parameters:
. dm - The DMNetwork object

  Output Parameters:
+ vStart - The first vertex point
- vEnd   - One beyond the last vertex point

  Level: beginner

.seealso: DMNetworkGetEdgeRange
@*/
PetscErrorCode DMNetworkGetVertexRange(DM dm,PetscInt *vStart,PetscInt *vEnd)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (vStart) *vStart = network->vStart;
  if (vEnd) *vEnd = network->vEnd;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetEdgeRange - Get the bounds [start, end) for the edges.

  Not Collective

  Input Parameters:
. dm - The DMNetwork object

  Output Parameters:
+ eStart - The first edge point
- eEnd   - One beyond the last edge point

  Level: beginner

.seealso: DMNetworkGetVertexRange
@*/
PetscErrorCode DMNetworkGetEdgeRange(DM dm,PetscInt *eStart,PetscInt *eEnd)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (eStart) *eStart = network->eStart;
  if (eEnd) *eEnd = network->eEnd;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalEdgeIndex - Get the user global numbering for the edge.

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
- p  - edge point

  Output Parameters:
. index - user global numbering for the edge

  Level: intermediate

.seealso: DMNetworkGetGlobalVertexIndex
@*/
PetscErrorCode DMNetworkGetGlobalEdgeIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscErrorCode    ierr;
  DM_Network        *network = (DM_Network*)dm->data;
  PetscInt          offsetp;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() first");
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetp);
  *index = header->index;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalVertexIndex - Get the user global numbering for the vertex.

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
- p  - vertex point

  Output Parameters:
. index - user global numbering for the vertex

  Level: intermediate

.seealso: DMNetworkGetGlobalEdgeIndex
@*/
PetscErrorCode DMNetworkGetGlobalVertexIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscErrorCode    ierr;
  DM_Network        *network = (DM_Network*)dm->data;
  PetscInt          offsetp;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() first");
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetp);
  *index = header->index;
  PetscFunctionReturn(0);
}

/*
  DMNetworkGetComponentKeyOffset - Gets the type along with the offset for indexing the
                                    component value from the component data array

  Not Collective

  Input Parameters:
+ dm      - The DMNetwork object
. p       - vertex/edge point
- compnum - component number

  Output Parameters:
+ compkey - the key obtained when registering the component
- offset  - offset into the component data array associated with the vertex/edge point

  Notes:
  Typical usage:

  DMNetworkGetComponentDataArray(dm, &arr);
  DMNetworkGetVertex/EdgeRange(dm,&Start,&End);
  Loop over vertices or edges
    DMNetworkGetNumComponents(dm,v,&numcomps);
    Loop over numcomps
      DMNetworkGetComponentKeyOffset(dm,v,compnum,&key,&offset);
      compdata = (UserCompDataType)(arr+offset);

  Level: intermediate

.seealso: DMNetworkGetNumComponents, DMNetworkGetComponentDataArray,
*/
PetscErrorCode DMNetworkGetComponentKeyOffset(DM dm,PetscInt p, PetscInt compnum, PetscInt *compkey, PetscInt *offset)
{
  PetscErrorCode           ierr;
  PetscInt                 offsetp;
  DMNetworkComponentHeader header;
  DM_Network               *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetp);
  if (compkey) *compkey = header->key[compnum];
  if (offset) *offset  = offsetp+network->dataheadersize+header->offset[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetComponent - Returns the network component and its key

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
. p  - edge or vertex point
- compnum - component number

  Output Parameters:
+ compkey - the key set for this computing during registration
- component - the component data

  Notes:
  Typical usage:

  DMNetworkGetVertex/EdgeRange(dm,&Start,&End);
  Loop over vertices or edges
    DMNetworkGetNumComponents(dm,v,&numcomps);
    Loop over numcomps
      DMNetworkGetComponent(dm,v,compnum,&key,&component);

  Level: beginner

.seealso: DMNetworkGetNumComponents, DMNetworkGetVariableOffset
@*/
PetscErrorCode DMNetworkGetComponent(DM dm, PetscInt p, PetscInt compnum, PetscInt *key, void **component)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetd = 0;

  PetscFunctionBegin;
  ierr = DMNetworkGetComponentKeyOffset(dm,p,compnum,key,&offsetd);CHKERRQ(ierr);
  *component = network->componentdataarray+offsetd;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAddComponent - Adds a network component at the given point (vertex/edge)

  Not Collective

  Input Parameters:
+ dm           - The DMNetwork object
. p            - vertex/edge point
. componentkey - component key returned while registering the component
- compvalue    - pointer to the data structure for the component

  Level: beginner

.seealso: DMNetworkGetVertexRange, DMNetworkGetEdgeRange, DMNetworkRegisterComponent
@*/
PetscErrorCode DMNetworkAddComponent(DM dm, PetscInt p,PetscInt componentkey,void* compvalue)
{
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponent       *component = &network->component[componentkey];
  DMNetworkComponentHeader header = &network->header[p];
  DMNetworkComponentValue  cvalue = &network->cvalue[p];
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (header->ndata == MAX_DATA_AT_POINT) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Number of components at a point exceeds the max %D",MAX_DATA_AT_POINT);

  header->size[header->ndata] = component->size;
  ierr = PetscSectionAddDof(network->DataSection,p,component->size);CHKERRQ(ierr);
  header->key[header->ndata] = componentkey;
  if (header->ndata != 0) header->offset[header->ndata] = header->offset[header->ndata-1] + header->size[header->ndata-1];
  header->nvar[header->ndata] = 0;

  cvalue->data[header->ndata] = (void*)compvalue;
  header->ndata++;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetComponentNumVariables - Sets the number of variables for a component

  Not Collective

  Input Parameters:
+ dm           - The DMNetwork object
. p            - vertex/edge point
. compnum      - component number (First component added = 0, second = 1, ...)
- nvar         - number of variables for the component

  Level: beginner

.seealso: DMNetworkAddComponent(), DMNetworkGetNumComponents(),DMNetworkRegisterComponent()
@*/
PetscErrorCode DMNetworkSetComponentNumVariables(DM dm, PetscInt p,PetscInt compnum,PetscInt nvar)
{
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponentHeader header = &network->header[p];
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = DMNetworkAddNumVariables(dm,p,nvar);CHKERRQ(ierr);
  header->nvar[compnum] = nvar;
  if (compnum != 0) header->offsetvarrel[compnum] = header->offsetvarrel[compnum-1] + header->nvar[compnum-1];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetNumComponents - Get the number of components at a vertex/edge

  Not Collective

  Input Parameters:
+ dm - The DMNetwork object
- p  - vertex/edge point

  Output Parameters:
. numcomponents - Number of components at the vertex/edge

  Level: beginner

.seealso: DMNetworkRegisterComponent, DMNetworkAddComponent
@*/
PetscErrorCode DMNetworkGetNumComponents(DM dm,PetscInt p,PetscInt *numcomponents)
{
  PetscErrorCode ierr;
  PetscInt       offset;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->DataSection,p,&offset);CHKERRQ(ierr);
  *numcomponents = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVariableOffset - Get the offset for accessing the variable associated with the given vertex/edge from the local vector.

  Not Collective

  Input Parameters:
+ dm     - The DMNetwork object
- p      - the edge/vertex point

  Output Parameters:
. offset - the offset

  Level: beginner

.seealso: DMNetworkGetVariableGlobalOffset, DMGetLocalVector
@*/
PetscErrorCode DMNetworkGetVariableOffset(DM dm,PetscInt p,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->plex->localSection,p,offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVariableGlobalOffset - Get the global offset for the variable associated with the given vertex/edge from the global vector.

  Not Collective

  Input Parameters:
+ dm      - The DMNetwork object
- p       - the edge/vertex point

  Output Parameters:
. offsetg - the offset

  Level: beginner

.seealso: DMNetworkGetVariableOffset, DMGetLocalVector
@*/
PetscErrorCode DMNetworkGetVariableGlobalOffset(DM dm,PetscInt p,PetscInt *offsetg)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->plex->globalSection,p,offsetg);CHKERRQ(ierr);
  if (*offsetg < 0) *offsetg = -(*offsetg + 1); /* Convert to actual global offset for ghost vertex */
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetComponentVariableOffset - Get the offset for accessing the variable associated with a component for the given vertex/edge from the local vector.

  Not Collective

  Input Parameters:
+ dm     - The DMNetwork object
. p      - the edge/vertex point
- compnum - component number

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: DMNetworkGetVariableGlobalOffset(), DMGetLocalVector(), DMNetworkSetComponentNumVariables()
@*/
PetscErrorCode DMNetworkGetComponentVariableOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = DMNetworkGetVariableOffset(dm,p,&offsetp);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetd);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetd);
  *offset = offsetp + header->offsetvarrel[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetComponentVariableGlobalOffset - Get the global offset for accessing the variable associated with a component for the given vertex/edge from the local vector.

  Not Collective

  Input Parameters:
+ dm     - The DMNetwork object
. p      - the edge/vertex point
- compnum - component number

  Output Parameters:
. offsetg - the global offset

  Level: intermediate

.seealso: DMNetworkGetVariableGlobalOffset(), DMNetworkGetComponentVariableOffset(), DMGetLocalVector(), DMNetworkSetComponentNumVariables()
@*/
PetscErrorCode DMNetworkGetComponentVariableGlobalOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offsetg)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = DMNetworkGetVariableGlobalOffset(dm,p,&offsetp);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetd);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetd);
  *offsetg = offsetp + header->offsetvarrel[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetEdgeOffset - Get the offset for accessing the variable associated with the given edge from the local subvector.

  Not Collective

  Input Parameters:
+ dm     - The DMNetwork object
- p      - the edge point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: DMNetworkGetVariableGlobalOffset, DMGetLocalVector
@*/
PetscErrorCode DMNetworkGetEdgeOffset(DM dm,PetscInt p,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;

  ierr = PetscSectionGetOffset(network->edge.DofSection,p,offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVertexOffset - Get the offset for accessing the variable associated with the given vertex from the local subvector.

  Not Collective

  Input Parameters:
+ dm     - The DMNetwork object
- p      - the vertex point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: DMNetworkGetVariableGlobalOffset, DMGetLocalVector
@*/
PetscErrorCode DMNetworkGetVertexOffset(DM dm,PetscInt p,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;

  p -= network->vStart;

  ierr = PetscSectionGetOffset(network->vertex.DofSection,p,offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*@
  DMNetworkAddNumVariables - Add number of variables associated with a given point.

  Not Collective

  Input Parameters:
+ dm   - The DMNetworkObject
. p    - the vertex/edge point
- nvar - number of additional variables

  Level: beginner

.seealso: DMNetworkSetNumVariables
@*/
PetscErrorCode DMNetworkAddNumVariables(DM dm,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionAddDof(network->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetNumVariables - Gets number of variables for a vertex/edge point.

  Not Collective

  Input Parameters:
+ dm   - The DMNetworkObject
- p    - the vertex/edge point

  Output Parameters:
. nvar - number of variables

  Level: beginner

.seealso: DMNetworkAddNumVariables, DMNetworkSddNumVariables
@*/
PetscErrorCode DMNetworkGetNumVariables(DM dm,PetscInt p,PetscInt *nvar)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetDof(network->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetNumVariables - Sets number of variables for a vertex/edge point.

  Not Collective

  Input Parameters:
+ dm   - The DMNetworkObject
. p    - the vertex/edge point
- nvar - number of variables

  Level: beginner

.seealso: DMNetworkAddNumVariables
@*/
PetscErrorCode DMNetworkSetNumVariables(DM dm,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionSetDof(network->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Sets up the array that holds the data for all components and its associated section. This
   function is called during DMSetUp() */
PetscErrorCode DMNetworkComponentSetUp(DM dm)
{
  PetscErrorCode           ierr;
  DM_Network               *network = (DM_Network*)dm->data;
  PetscInt                 arr_size,p,offset,offsetp,ncomp,i;
  DMNetworkComponentHeader header;
  DMNetworkComponentValue  cvalue;
  DMNetworkComponentGenericDataType *componentdataarray;

  PetscFunctionBegin;
  ierr = PetscSectionSetUp(network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(network->DataSection,&arr_size);CHKERRQ(ierr);
  ierr = PetscMalloc1(arr_size,&network->componentdataarray);CHKERRQ(ierr);
  componentdataarray = network->componentdataarray;
  for (p = network->pStart; p < network->pEnd; p++) {
    ierr = PetscSectionGetOffset(network->DataSection,p,&offsetp);CHKERRQ(ierr);
    /* Copy header */
    header = &network->header[p];
    ierr = PetscMemcpy(componentdataarray+offsetp,header,network->dataheadersize*sizeof(DMNetworkComponentGenericDataType));CHKERRQ(ierr);
    /* Copy data */
    cvalue = &network->cvalue[p];
    ncomp = header->ndata;
    for (i = 0; i < ncomp; i++) {
      offset = offsetp + network->dataheadersize + header->offset[i];
      ierr = PetscMemcpy(componentdataarray+offset,cvalue->data[i],header->size[i]*sizeof(DMNetworkComponentGenericDataType));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Sets up the section for dofs. This routine is called during DMSetUp() */
PetscErrorCode DMNetworkVariablesSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionSetUp(network->DofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMNetworkGetComponentDataArray - Returns the component data array

  Not Collective

  Input Parameters:
. dm - The DMNetwork Object

  Output Parameters:
. componentdataarray - array that holds data for all components

  Level: intermediate

.seealso: DMNetworkGetComponentKeyOffset, DMNetworkGetNumComponents
*/
PetscErrorCode DMNetworkGetComponentDataArray(DM dm,DMNetworkComponentGenericDataType **componentdataarray)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  *componentdataarray = network->componentdataarray;
  PetscFunctionReturn(0);
}

/* Get a subsection from a range of points */
PetscErrorCode DMNetworkGetSubSection_private(PetscSection master, PetscInt pstart, PetscInt pend,PetscSection *subsection)
{
  PetscErrorCode ierr;
  PetscInt       i, nvar;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)master), subsection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subsection, 0, pend - pstart);CHKERRQ(ierr);
  for (i = pstart; i < pend; i++) {
    ierr = PetscSectionGetDof(master,i,&nvar);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*subsection, i - pstart, nvar);CHKERRQ(ierr);
  }

  ierr = PetscSectionSetUp(*subsection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create a submap of points with a GlobalToLocal structure */
PetscErrorCode DMNetworkSetSubMap_private(PetscInt pstart, PetscInt pend, ISLocalToGlobalMapping *map)
{
  PetscErrorCode ierr;
  PetscInt       i, *subpoints;

  PetscFunctionBegin;
  /* Create index sets to map from "points" to "subpoints" */
  ierr = PetscMalloc1(pend - pstart, &subpoints);CHKERRQ(ierr);
  for (i = pstart; i < pend; i++) {
    subpoints[i - pstart] = i;
  }
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,pend-pstart,subpoints,PETSC_COPY_VALUES,map);CHKERRQ(ierr);
  ierr = PetscFree(subpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAssembleGraphStructures - Assembles vertex and edge data structures. Must be called after DMNetworkDistribute.

  Collective

  Input Parameters:
. dm   - The DMNetworkObject

  Note: the routine will create alternative orderings for the vertices and edges. Assume global network points are:

  points = [0 1 2 3 4 5 6]

  where edges = [0,1,2,3] and vertices = [4,5,6]. The new orderings will be specific to the subset (i.e vertices = [0,1,2] <- [4,5,6]).

  With this new ordering a local PetscSection, global PetscSection and PetscSF will be created specific to the subset.

  Level: intermediate

@*/
PetscErrorCode DMNetworkAssembleGraphStructures(DM dm)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  /* Create maps for vertices and edges */
  ierr = DMNetworkSetSubMap_private(network->vStart,network->vEnd,&network->vertex.mapping);CHKERRQ(ierr);
  ierr = DMNetworkSetSubMap_private(network->eStart,network->eEnd,&network->edge.mapping);CHKERRQ(ierr);

  /* Create local sub-sections */
  ierr = DMNetworkGetSubSection_private(network->DofSection,network->vStart,network->vEnd,&network->vertex.DofSection);CHKERRQ(ierr);
  ierr = DMNetworkGetSubSection_private(network->DofSection,network->eStart,network->eEnd,&network->edge.DofSection);CHKERRQ(ierr);

  if (size > 1) {
    ierr = PetscSFGetSubSF(network->plex->sf, network->vertex.mapping, &network->vertex.sf);CHKERRQ(ierr);

    ierr = PetscSectionCreateGlobalSection(network->vertex.DofSection, network->vertex.sf, PETSC_FALSE, PETSC_FALSE, &network->vertex.GlobalDofSection);CHKERRQ(ierr);
    ierr = PetscSFGetSubSF(network->plex->sf, network->edge.mapping, &network->edge.sf);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(network->edge.DofSection, network->edge.sf, PETSC_FALSE, PETSC_FALSE, &network->edge.GlobalDofSection);CHKERRQ(ierr);
  } else {
    /* create structures for vertex */
    ierr = PetscSectionClone(network->vertex.DofSection,&network->vertex.GlobalDofSection);CHKERRQ(ierr);
    /* create structures for edge */
    ierr = PetscSectionClone(network->edge.DofSection,&network->edge.GlobalDofSection);CHKERRQ(ierr);
  }

  /* Add viewers */
  ierr = PetscObjectSetName((PetscObject)network->edge.GlobalDofSection,"Global edge dof section");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)network->vertex.GlobalDofSection,"Global vertex dof section");CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(network->edge.GlobalDofSection, NULL, "-edge_global_section_view");CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(network->vertex.GlobalDofSection, NULL, "-vertex_global_section_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkDistribute - Distributes the network and moves associated component data.

  Collective

  Input Parameter:
+ DM - the DMNetwork object
- overlap - The overlap of partitions, 0 is the default

  Notes:
  Distributes the network with <overlap>-overlapping partitioning of the edges.

  Level: intermediate

.seealso: DMNetworkCreate
@*/
PetscErrorCode DMNetworkDistribute(DM *dm,PetscInt overlap)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  DM_Network     *oldDMnetwork = (DM_Network*)((*dm)->data);
  DM_Network     *newDMnetwork;
  PetscSF        pointsf=NULL;
  DM             newDM;
  PetscInt       j,e,v,offset,*subnetvtx;
  PetscPartitioner         part;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)*dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  ierr = DMNetworkCreate(PetscObjectComm((PetscObject)*dm),&newDM);CHKERRQ(ierr);
  newDMnetwork = (DM_Network*)newDM->data;
  newDMnetwork->dataheadersize = sizeof(struct _p_DMNetworkComponentHeader)/sizeof(DMNetworkComponentGenericDataType);

  /* Enable runtime options for petscpartitioner */
  ierr = DMPlexGetPartitioner(oldDMnetwork->plex,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

  /* Distribute plex dm and dof section */
  ierr = DMPlexDistribute(oldDMnetwork->plex,overlap,&pointsf,&newDMnetwork->plex);CHKERRQ(ierr);

  /* Distribute dof section */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)*dm),&newDMnetwork->DofSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointsf,oldDMnetwork->DofSection,NULL,newDMnetwork->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)*dm),&newDMnetwork->DataSection);CHKERRQ(ierr);

  /* Distribute data and associated section */
  ierr = DMPlexDistributeData(newDMnetwork->plex,pointsf,oldDMnetwork->DataSection,MPIU_INT,(void*)oldDMnetwork->componentdataarray,newDMnetwork->DataSection,(void**)&newDMnetwork->componentdataarray);CHKERRQ(ierr);

  ierr = PetscSectionGetChart(newDMnetwork->DataSection,&newDMnetwork->pStart,&newDMnetwork->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMnetwork->plex,0, &newDMnetwork->eStart,&newDMnetwork->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMnetwork->plex,1,&newDMnetwork->vStart,&newDMnetwork->vEnd);CHKERRQ(ierr);
  newDMnetwork->nEdges    = newDMnetwork->eEnd - newDMnetwork->eStart;
  newDMnetwork->nVertices = newDMnetwork->vEnd - newDMnetwork->vStart;
  newDMnetwork->NVertices = oldDMnetwork->NVertices;
  newDMnetwork->NEdges    = oldDMnetwork->NEdges;

  /* Set Dof section as the section for dm */
  ierr = DMSetLocalSection(newDMnetwork->plex,newDMnetwork->DofSection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(newDMnetwork->plex,&newDMnetwork->GlobalDofSection);CHKERRQ(ierr);

  /* Set up subnetwork info in the newDM */
  newDMnetwork->nsubnet  = oldDMnetwork->nsubnet;
  newDMnetwork->ncsubnet = oldDMnetwork->ncsubnet;
  ierr = PetscCalloc1(newDMnetwork->nsubnet,&newDMnetwork->subnet);CHKERRQ(ierr);
  /* Copy over the global number of vertices and edges in each subnetwork. Note that these are already
     calculated in DMNetworkLayoutSetUp()
  */
  for(j=0; j < newDMnetwork->nsubnet; j++) {
    newDMnetwork->subnet[j].Nvtx  = oldDMnetwork->subnet[j].Nvtx;
    newDMnetwork->subnet[j].Nedge = oldDMnetwork->subnet[j].Nedge;
  }

  for (e = newDMnetwork->eStart; e < newDMnetwork->eEnd; e++ ) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    newDMnetwork->subnet[header->subnetid].nedge++;
  }

  for (v = newDMnetwork->vStart; v < newDMnetwork->vEnd; v++ ) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    newDMnetwork->subnet[header->subnetid].nvtx++;
  }

  /* Now create the vertices and edge arrays for the subnetworks */
  ierr = PetscCalloc1(newDMnetwork->vEnd-newDMnetwork->vStart,&newDMnetwork->subnetvtx);CHKERRQ(ierr);
  subnetvtx = newDMnetwork->subnetvtx;

  for (j=0; j<newDMnetwork->nsubnet; j++) {
    ierr = PetscCalloc1(newDMnetwork->subnet[j].nedge,&newDMnetwork->subnet[j].edges);CHKERRQ(ierr);
    newDMnetwork->subnet[j].vertices = subnetvtx;
    subnetvtx                       += newDMnetwork->subnet[j].nvtx;

    /* Temporarily setting nvtx and nedge to 0 so we can use them as counters in the below for loop.
       These get updated when the vertices and edges are added. */
    newDMnetwork->subnet[j].nvtx = newDMnetwork->subnet[j].nedge = 0;
  }

  /* Set the vertices and edges in each subnetwork */
  for (e = newDMnetwork->eStart; e < newDMnetwork->eEnd; e++ ) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    newDMnetwork->subnet[header->subnetid].edges[newDMnetwork->subnet[header->subnetid].nedge++] = e;
  }

  for (v = newDMnetwork->vStart; v < newDMnetwork->vEnd; v++ ) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    newDMnetwork->subnet[header->subnetid].vertices[newDMnetwork->subnet[header->subnetid].nvtx++] = v;
  }

  newDM->setupcalled = (*dm)->setupcalled;
  newDMnetwork->distributecalled = PETSC_TRUE;

  /* Destroy point SF */
  ierr = PetscSFDestroy(&pointsf);CHKERRQ(ierr);

  ierr = DMDestroy(dm);CHKERRQ(ierr);
  *dm  = newDM;
  PetscFunctionReturn(0);
}

/*@C
  PetscSFGetSubSF - Returns an SF for a specific subset of points. Leaves are re-numbered to reflect the new ordering.

  Input Parameters:
+ masterSF - the original SF structure
- map      - a ISLocalToGlobal mapping that contains the subset of points

  Output Parameters:
. subSF    - a subset of the masterSF for the desired subset.
@*/
PetscErrorCode PetscSFGetSubSF(PetscSF mastersf, ISLocalToGlobalMapping map, PetscSF *subSF) {

  PetscErrorCode        ierr;
  PetscInt              nroots, nleaves, *ilocal_sub;
  PetscInt              i, *ilocal_map, nroots_sub, nleaves_sub = 0;
  PetscInt              *local_points, *remote_points;
  PetscSFNode           *iremote_sub;
  const PetscInt        *ilocal;
  const PetscSFNode     *iremote;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(mastersf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);

  /* Look for leaves that pertain to the subset of points. Get the local ordering */
  ierr = PetscMalloc1(nleaves,&ilocal_map);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(map,IS_GTOLM_MASK,nleaves,ilocal,NULL,ilocal_map);CHKERRQ(ierr);
  for (i = 0; i < nleaves; i++) {
    if (ilocal_map[i] != -1) nleaves_sub += 1;
  }
  /* Re-number ilocal with subset numbering. Need information from roots */
  ierr = PetscMalloc2(nroots,&local_points,nroots,&remote_points);CHKERRQ(ierr);
  for (i = 0; i < nroots; i++) local_points[i] = i;
  ierr = ISGlobalToLocalMappingApply(map,IS_GTOLM_MASK,nroots,local_points,NULL,local_points);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(mastersf, MPIU_INT, local_points, remote_points);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(mastersf, MPIU_INT, local_points, remote_points);CHKERRQ(ierr);
  /* Fill up graph using local (that is, local to the subset) numbering. */
  ierr = PetscMalloc1(nleaves_sub,&ilocal_sub);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves_sub,&iremote_sub);CHKERRQ(ierr);
  nleaves_sub = 0;
  for (i = 0; i < nleaves; i++) {
    if (ilocal_map[i] != -1) {
      ilocal_sub[nleaves_sub] = ilocal_map[i];
      iremote_sub[nleaves_sub].rank = iremote[i].rank;
      iremote_sub[nleaves_sub].index = remote_points[ilocal[i]];
      nleaves_sub += 1;
    }
  }
  ierr = PetscFree2(local_points,remote_points);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetSize(map,&nroots_sub);CHKERRQ(ierr);

  /* Create new subSF */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,subSF);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*subSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*subSF,nroots_sub,nleaves_sub,ilocal_sub,PETSC_OWN_POINTER,iremote_sub,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(ilocal_map);CHKERRQ(ierr);
  ierr = PetscFree(iremote_sub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSupportingEdges - Return the supporting edges for this vertex point

  Not Collective

  Input Parameters:
+ dm - The DMNetwork object
- p  - the vertex point

  Output Parameters:
+ nedges - number of edges connected to this vertex point
- edges  - List of edge points

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMNetworkCreate, DMNetworkGetConnectedVertices
@*/
PetscErrorCode DMNetworkGetSupportingEdges(DM dm,PetscInt vertex,PetscInt *nedges,const PetscInt *edges[])
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMPlexGetSupportSize(network->plex,vertex,nedges);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(network->plex,vertex,edges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetConnectedVertices - Return the connected vertices for this edge point

  Not Collective

  Input Parameters:
+ dm - The DMNetwork object
- p  - the edge point

  Output Parameters:
. vertices  - vertices connected to this edge

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMNetworkCreate, DMNetworkGetSupportingEdges
@*/
PetscErrorCode DMNetworkGetConnectedVertices(DM dm,PetscInt edge,const PetscInt *vertices[])
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMPlexGetCone(network->plex,edge,vertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkIsGhostVertex - Returns TRUE if the vertex is a ghost vertex

  Not Collective

  Input Parameters:
+ dm - The DMNetwork object
- p  - the vertex point

  Output Parameter:
. isghost - TRUE if the vertex is a ghost point

  Level: beginner

.seealso: DMNetworkCreate, DMNetworkGetConnectedVertices, DMNetworkGetVertexRange
@*/
PetscErrorCode DMNetworkIsGhostVertex(DM dm,PetscInt p,PetscBool *isghost)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() first");
  *isghost = PETSC_FALSE;
  ierr = DMGetGlobalSection(network->plex,&sectiong);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(sectiong,p,&offsetg);CHKERRQ(ierr);
  if (offsetg < 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUp_Network(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network=(DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMNetworkComponentSetUp(dm);CHKERRQ(ierr);
  ierr = DMNetworkVariablesSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetLocalSection(network->plex,network->DofSection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(network->plex,&network->GlobalDofSection);CHKERRQ(ierr);

  dm->setupcalled = PETSC_TRUE;
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMNetworkHasJacobian - Sets global flag for using user's sub Jacobian matrices
                            -- replaced by DMNetworkSetOption(network,userjacobian,PETSC_TURE)?

    Collective

    Input Parameters:
+   dm - The DMNetwork object
.   eflg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE) if user provides Jacobian for edges
-   vflg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE) if user provides Jacobian for vertices

    Level: intermediate

@*/
PetscErrorCode DMNetworkHasJacobian(DM dm,PetscBool eflg,PetscBool vflg)
{
  DM_Network     *network=(DM_Network*)dm->data;
  PetscErrorCode ierr;
  PetscInt       nVertices = network->nVertices;

  PetscFunctionBegin;
  network->userEdgeJacobian   = eflg;
  network->userVertexJacobian = vflg;

  if (eflg && !network->Je) {
    ierr = PetscCalloc1(3*network->nEdges,&network->Je);CHKERRQ(ierr);
  }

  if (vflg && !network->Jv && nVertices) {
    PetscInt       i,*vptr,nedges,vStart=network->vStart;
    PetscInt       nedges_total;
    const PetscInt *edges;

    /* count nvertex_total */
    nedges_total = 0;
    ierr = PetscMalloc1(nVertices+1,&vptr);CHKERRQ(ierr);

    vptr[0] = 0;
    for (i=0; i<nVertices; i++) {
      ierr = DMNetworkGetSupportingEdges(dm,i+vStart,&nedges,&edges);CHKERRQ(ierr);
      nedges_total += nedges;
      vptr[i+1] = vptr[i] + 2*nedges + 1;
    }

    ierr = PetscCalloc1(2*nedges_total+nVertices,&network->Jv);CHKERRQ(ierr);
    network->Jvptr = vptr;
  }
  PetscFunctionReturn(0);
}

/*@
    DMNetworkEdgeSetMatrix - Sets user-provided Jacobian matrices for this edge to the network

    Not Collective

    Input Parameters:
+   dm - The DMNetwork object
.   p  - the edge point
-   J - array (size = 3) of Jacobian submatrices for this edge point:
        J[0]: this edge
        J[1] and J[2]: connected vertices, obtained by calling DMNetworkGetConnectedVertices()

    Level: advanced

.seealso: DMNetworkVertexSetMatrix
@*/
PetscErrorCode DMNetworkEdgeSetMatrix(DM dm,PetscInt p,Mat J[])
{
  DM_Network     *network=(DM_Network*)dm->data;

  PetscFunctionBegin;
  if (!network->Je) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"Must call DMNetworkHasJacobian() collectively before calling DMNetworkEdgeSetMatrix");

  if (J) {
    network->Je[3*p]   = J[0];
    network->Je[3*p+1] = J[1];
    network->Je[3*p+2] = J[2];
  }
  PetscFunctionReturn(0);
}

/*@
    DMNetworkVertexSetMatrix - Sets user-provided Jacobian matrix for this vertex to the network

    Not Collective

    Input Parameters:
+   dm - The DMNetwork object
.   p  - the vertex point
-   J - array of Jacobian (size = 2*(num of supporting edges) + 1) submatrices for this vertex point:
        J[0]:       this vertex
        J[1+2*i]:   i-th supporting edge
        J[1+2*i+1]: i-th connected vertex

    Level: advanced

.seealso: DMNetworkEdgeSetMatrix
@*/
PetscErrorCode DMNetworkVertexSetMatrix(DM dm,PetscInt p,Mat J[])
{
  PetscErrorCode ierr;
  DM_Network     *network=(DM_Network*)dm->data;
  PetscInt       i,*vptr,nedges,vStart=network->vStart;
  const PetscInt *edges;

  PetscFunctionBegin;
  if (!network->Jv) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"Must call DMNetworkHasJacobian() collectively before calling DMNetworkVertexSetMatrix");

  if (J) {
    vptr = network->Jvptr;
    network->Jv[vptr[p-vStart]] = J[0]; /* Set Jacobian for this vertex */

    /* Set Jacobian for each supporting edge and connected vertex */
    ierr = DMNetworkGetSupportingEdges(dm,p,&nedges,&edges);CHKERRQ(ierr);
    for (i=1; i<=2*nedges; i++) network->Jv[vptr[p-vStart]+i] = J[i];
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetPreallocationDenseblock_private(PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscErrorCode ierr;
  PetscInt       j;
  PetscScalar    val=(PetscScalar)ncols;

  PetscFunctionBegin;
  if (!ghost) {
    for (j=0; j<nrows; j++) {
      ierr = VecSetValues(vdnz,1,&rows[j],&val,ADD_VALUES);CHKERRQ(ierr);
    }
  } else {
    for (j=0; j<nrows; j++) {
      ierr = VecSetValues(vonz,1,&rows[j],&val,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetPreallocationUserblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscErrorCode ierr;
  PetscInt       j,ncols_u;
  PetscScalar    val;

  PetscFunctionBegin;
  if (!ghost) {
    for (j=0; j<nrows; j++) {
      ierr = MatGetRow(Ju,j,&ncols_u,NULL,NULL);CHKERRQ(ierr);
      val = (PetscScalar)ncols_u;
      ierr = VecSetValues(vdnz,1,&rows[j],&val,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Ju,j,&ncols_u,NULL,NULL);CHKERRQ(ierr);
    }
  } else {
    for (j=0; j<nrows; j++) {
      ierr = MatGetRow(Ju,j,&ncols_u,NULL,NULL);CHKERRQ(ierr);
      val = (PetscScalar)ncols_u;
      ierr = VecSetValues(vonz,1,&rows[j],&val,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Ju,j,&ncols_u,NULL,NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetPreallocationblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Ju) {
    ierr = MatSetPreallocationUserblock_private(Ju,nrows,rows,ncols,ghost,vdnz,vonz);CHKERRQ(ierr);
  } else {
    ierr = MatSetPreallocationDenseblock_private(nrows,rows,ncols,ghost,vdnz,vonz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetDenseblock_private(PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       j,*cols;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  ierr = PetscCalloc2(ncols,&cols,nrows*ncols,&zeros);CHKERRQ(ierr);
  for (j=0; j<ncols; j++) cols[j] = j+ cstart;
  ierr = MatSetValues(*J,nrows,rows,ncols,cols,zeros,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree2(cols,zeros);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetUserblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       j,M,N,row,col,ncols_u;
  const PetscInt *cols;
  PetscScalar    zero=0.0;

  PetscFunctionBegin;
  ierr = MatGetSize(Ju,&M,&N);CHKERRQ(ierr);
  if (nrows != M || ncols != N) SETERRQ4(PetscObjectComm((PetscObject)Ju),PETSC_ERR_USER,"%D by %D must equal %D by %D",nrows,ncols,M,N);

  for (row=0; row<nrows; row++) {
    ierr = MatGetRow(Ju,row,&ncols_u,&cols,NULL);CHKERRQ(ierr);
    for (j=0; j<ncols_u; j++) {
      col = cols[j] + cstart;
      ierr = MatSetValues(*J,1,&rows[row],1,&col,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(Ju,row,&ncols_u,&cols,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Ju) {
    ierr = MatSetUserblock_private(Ju,nrows,rows,ncols,cstart,J);CHKERRQ(ierr);
  } else {
    ierr = MatSetDenseblock_private(nrows,rows,ncols,cstart,J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Creates a GlobalToLocal mapping with a Local and Global section. This is akin to the routine DMGetLocalToGlobalMapping but without the need of providing a dm.
*/
PetscErrorCode CreateSubGlobalToLocalMapping_private(PetscSection globalsec, PetscSection localsec, ISLocalToGlobalMapping *ltog)
{
  PetscErrorCode ierr;
  PetscInt       i,size,dof;
  PetscInt       *glob2loc;

  PetscFunctionBegin;
  ierr = PetscSectionGetStorageSize(localsec,&size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&glob2loc);CHKERRQ(ierr);

  for (i = 0; i < size; i++) {
    ierr = PetscSectionGetOffset(globalsec,i,&dof);CHKERRQ(ierr);
    dof = (dof >= 0) ? dof : -(dof + 1);
    glob2loc[i] = dof;
  }

  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,size,glob2loc,PETSC_OWN_POINTER,ltog);CHKERRQ(ierr);
#if 0
  ierr = PetscIntView(size,glob2loc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#include <petsc/private/matimpl.h>

PetscErrorCode DMCreateMatrix_Network_Nest(DM dm,Mat *J)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscMPIInt    rank, size;
  PetscInt       eDof,vDof;
  Mat            j11,j12,j21,j22,bA[2][2];
  MPI_Comm       comm;
  ISLocalToGlobalMapping eISMap,vISMap;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = PetscSectionGetConstrainedStorageSize(network->edge.GlobalDofSection,&eDof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(network->vertex.GlobalDofSection,&vDof);CHKERRQ(ierr);

  ierr = MatCreate(comm, &j11);CHKERRQ(ierr);
  ierr = MatSetSizes(j11, eDof, eDof, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(j11, MATMPIAIJ);CHKERRQ(ierr);

  ierr = MatCreate(comm, &j12);CHKERRQ(ierr);
  ierr = MatSetSizes(j12, eDof, vDof, PETSC_DETERMINE ,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(j12, MATMPIAIJ);CHKERRQ(ierr);

  ierr = MatCreate(comm, &j21);CHKERRQ(ierr);
  ierr = MatSetSizes(j21, vDof, eDof, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(j21, MATMPIAIJ);CHKERRQ(ierr);

  ierr = MatCreate(comm, &j22);CHKERRQ(ierr);
  ierr = MatSetSizes(j22, vDof, vDof, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(j22, MATMPIAIJ);CHKERRQ(ierr);

  bA[0][0] = j11;
  bA[0][1] = j12;
  bA[1][0] = j21;
  bA[1][1] = j22;

  ierr = CreateSubGlobalToLocalMapping_private(network->edge.GlobalDofSection,network->edge.DofSection,&eISMap);CHKERRQ(ierr);
  ierr = CreateSubGlobalToLocalMapping_private(network->vertex.GlobalDofSection,network->vertex.DofSection,&vISMap);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(j11,eISMap,eISMap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(j12,eISMap,vISMap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(j21,vISMap,eISMap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(j22,vISMap,vISMap);CHKERRQ(ierr);

  ierr = MatSetUp(j11);CHKERRQ(ierr);
  ierr = MatSetUp(j12);CHKERRQ(ierr);
  ierr = MatSetUp(j21);CHKERRQ(ierr);
  ierr = MatSetUp(j22);CHKERRQ(ierr);

  ierr = MatCreateNest(comm,2,NULL,2,NULL,&bA[0][0],J);CHKERRQ(ierr);
  ierr = MatSetUp(*J);CHKERRQ(ierr);
  ierr = MatNestSetVecType(*J,VECNEST);CHKERRQ(ierr);
  ierr = MatDestroy(&j11);CHKERRQ(ierr);
  ierr = MatDestroy(&j12);CHKERRQ(ierr);
  ierr = MatDestroy(&j21);CHKERRQ(ierr);
  ierr = MatDestroy(&j22);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(*J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  /* Free structures */
  ierr = ISLocalToGlobalMappingDestroy(&eISMap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&vISMap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Network(DM dm,Mat *J)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       eStart,eEnd,vStart,vEnd,rstart,nrows,*rows,localSize;
  PetscInt       cstart,ncols,j,e,v;
  PetscBool      ghost,ghost_vc,ghost2,isNest;
  Mat            Juser;
  PetscSection   sectionGlobal;
  PetscInt       nedges,*vptr=NULL,vc,*rows_v; /* suppress maybe-uninitialized warning */
  const PetscInt *edges,*cone;
  MPI_Comm       comm;
  MatType        mtype;
  Vec            vd_nz,vo_nz;
  PetscInt       *dnnz,*onnz;
  PetscScalar    *vdnz,*vonz;

  PetscFunctionBegin;
  mtype = dm->mattype;
  ierr = PetscStrcmp(mtype,MATNEST,&isNest);CHKERRQ(ierr);
  if (isNest) {
    ierr = DMCreateMatrix_Network_Nest(dm,J);CHKERRQ(ierr);
    ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!network->userEdgeJacobian && !network->userVertexJacobian) {
    /* user does not provide Jacobian blocks */
    ierr = DMCreateMatrix_Plex(network->plex,J);CHKERRQ(ierr);
    ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MatCreate(PetscObjectComm((PetscObject)dm),J);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(network->plex,&sectionGlobal);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(sectionGlobal,&localSize);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);

  ierr = MatSetType(*J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);

  /* (1) Set matrix preallocation */
  /*------------------------------*/
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&vd_nz);CHKERRQ(ierr);
  ierr = VecSetSizes(vd_nz,localSize,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vd_nz);CHKERRQ(ierr);
  ierr = VecSet(vd_nz,0.0);CHKERRQ(ierr);
  ierr = VecDuplicate(vd_nz,&vo_nz);CHKERRQ(ierr);

  /* Set preallocation for edges */
  /*-----------------------------*/
  ierr = DMNetworkGetEdgeRange(dm,&eStart,&eEnd);CHKERRQ(ierr);

  ierr = PetscMalloc1(localSize,&rows);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    /* Get row indices */
    ierr = DMNetworkGetVariableGlobalOffset(dm,e,&rstart);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(dm,e,&nrows);CHKERRQ(ierr);
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set preallocation for conntected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,e,&cone);CHKERRQ(ierr);
      for (v=0; v<2; v++) {
        ierr = DMNetworkGetNumVariables(dm,cone[v],&ncols);CHKERRQ(ierr);

        if (network->Je) {
          Juser = network->Je[3*e+1+v]; /* Jacobian(e,v) */
        } else Juser = NULL;
        ierr = DMNetworkIsGhostVertex(dm,cone[v],&ghost);CHKERRQ(ierr);
        ierr = MatSetPreallocationblock_private(Juser,nrows,rows,ncols,ghost,vd_nz,vo_nz);CHKERRQ(ierr);
      }

      /* Set preallocation for edge self */
      cstart = rstart;
      if (network->Je) {
        Juser = network->Je[3*e]; /* Jacobian(e,e) */
      } else Juser = NULL;
      ierr = MatSetPreallocationblock_private(Juser,nrows,rows,nrows,PETSC_FALSE,vd_nz,vo_nz);CHKERRQ(ierr);
    }
  }

  /* Set preallocation for vertices */
  /*--------------------------------*/
  ierr = DMNetworkGetVertexRange(dm,&vStart,&vEnd);CHKERRQ(ierr);
  if (vEnd - vStart) vptr = network->Jvptr;

  for (v=vStart; v<vEnd; v++) {
    /* Get row indices */
    ierr = DMNetworkGetVariableGlobalOffset(dm,v,&rstart);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(dm,v,&nrows);CHKERRQ(ierr);
    if (!nrows) continue;

    ierr = DMNetworkIsGhostVertex(dm,v,&ghost);CHKERRQ(ierr);
    if (ghost) {
      ierr = PetscMalloc1(nrows,&rows_v);CHKERRQ(ierr);
    } else {
      rows_v = rows;
    }

    for (j=0; j<nrows; j++) rows_v[j] = j + rstart;

    /* Get supporting edges and connected vertices */
    ierr = DMNetworkGetSupportingEdges(dm,v,&nedges,&edges);CHKERRQ(ierr);

    for (e=0; e<nedges; e++) {
      /* Supporting edges */
      ierr = DMNetworkGetVariableGlobalOffset(dm,edges[e],&cstart);CHKERRQ(ierr);
      ierr = DMNetworkGetNumVariables(dm,edges[e],&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      ierr = MatSetPreallocationblock_private(Juser,nrows,rows_v,ncols,ghost,vd_nz,vo_nz);CHKERRQ(ierr);

      /* Connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,edges[e],&cone);CHKERRQ(ierr);
      vc = (v == cone[0]) ? cone[1]:cone[0];
      ierr = DMNetworkIsGhostVertex(dm,vc,&ghost_vc);CHKERRQ(ierr);

      ierr = DMNetworkGetNumVariables(dm,vc,&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+2]; /* Jacobian(v,vc) */
      } else Juser = NULL;
      if (ghost_vc||ghost) {
        ghost2 = PETSC_TRUE;
      } else {
        ghost2 = PETSC_FALSE;
      }
      ierr = MatSetPreallocationblock_private(Juser,nrows,rows_v,ncols,ghost2,vd_nz,vo_nz);CHKERRQ(ierr);
    }

    /* Set preallocation for vertex self */
    ierr = DMNetworkIsGhostVertex(dm,v,&ghost);CHKERRQ(ierr);
    if (!ghost) {
      ierr = DMNetworkGetVariableGlobalOffset(dm,v,&cstart);CHKERRQ(ierr);
      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]]; /* Jacobian(v,v) */
      } else Juser = NULL;
      ierr = MatSetPreallocationblock_private(Juser,nrows,rows_v,nrows,PETSC_FALSE,vd_nz,vo_nz);CHKERRQ(ierr);
    }
    if (ghost) {
      ierr = PetscFree(rows_v);CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(vd_nz);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vo_nz);CHKERRQ(ierr);

  ierr = PetscMalloc2(localSize,&dnnz,localSize,&onnz);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(vd_nz);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vo_nz);CHKERRQ(ierr);

  ierr = VecGetArray(vd_nz,&vdnz);CHKERRQ(ierr);
  ierr = VecGetArray(vo_nz,&vonz);CHKERRQ(ierr);
  for (j=0; j<localSize; j++) {
    dnnz[j] = (PetscInt)PetscRealPart(vdnz[j]);
    onnz[j] = (PetscInt)PetscRealPart(vonz[j]);
  }
  ierr = VecRestoreArray(vd_nz,&vdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(vo_nz,&vonz);CHKERRQ(ierr);
  ierr = VecDestroy(&vd_nz);CHKERRQ(ierr);
  ierr = VecDestroy(&vo_nz);CHKERRQ(ierr);

  ierr = MatSeqAIJSetPreallocation(*J,0,dnnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,0,dnnz,0,onnz);CHKERRQ(ierr);
  ierr = MatSetOption(*J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);

  /* (2) Set matrix entries for edges */
  /*----------------------------------*/
  for (e=eStart; e<eEnd; e++) {
    /* Get row indices */
    ierr = DMNetworkGetVariableGlobalOffset(dm,e,&rstart);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(dm,e,&nrows);CHKERRQ(ierr);
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set matrix entries for conntected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,e,&cone);CHKERRQ(ierr);
      for (v=0; v<2; v++) {
        ierr = DMNetworkGetVariableGlobalOffset(dm,cone[v],&cstart);CHKERRQ(ierr);
        ierr = DMNetworkGetNumVariables(dm,cone[v],&ncols);CHKERRQ(ierr);

        if (network->Je) {
          Juser = network->Je[3*e+1+v]; /* Jacobian(e,v) */
        } else Juser = NULL;
        ierr = MatSetblock_private(Juser,nrows,rows,ncols,cstart,J);CHKERRQ(ierr);
      }

      /* Set matrix entries for edge self */
      cstart = rstart;
      if (network->Je) {
        Juser = network->Je[3*e]; /* Jacobian(e,e) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows,nrows,cstart,J);CHKERRQ(ierr);
    }
  }

  /* Set matrix entries for vertices */
  /*---------------------------------*/
  for (v=vStart; v<vEnd; v++) {
    /* Get row indices */
    ierr = DMNetworkGetVariableGlobalOffset(dm,v,&rstart);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(dm,v,&nrows);CHKERRQ(ierr);
    if (!nrows) continue;

    ierr = DMNetworkIsGhostVertex(dm,v,&ghost);CHKERRQ(ierr);
    if (ghost) {
      ierr = PetscMalloc1(nrows,&rows_v);CHKERRQ(ierr);
    } else {
      rows_v = rows;
    }
    for (j=0; j<nrows; j++) rows_v[j] = j + rstart;

    /* Get supporting edges and connected vertices */
    ierr = DMNetworkGetSupportingEdges(dm,v,&nedges,&edges);CHKERRQ(ierr);

    for (e=0; e<nedges; e++) {
      /* Supporting edges */
      ierr = DMNetworkGetVariableGlobalOffset(dm,edges[e],&cstart);CHKERRQ(ierr);
      ierr = DMNetworkGetNumVariables(dm,edges[e],&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J);CHKERRQ(ierr);

      /* Connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,edges[e],&cone);CHKERRQ(ierr);
      vc = (v == cone[0]) ? cone[1]:cone[0];

      ierr = DMNetworkGetVariableGlobalOffset(dm,vc,&cstart);CHKERRQ(ierr);
      ierr = DMNetworkGetNumVariables(dm,vc,&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+2]; /* Jacobian(v,vc) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J);CHKERRQ(ierr);
    }

    /* Set matrix entries for vertex self */
    if (!ghost) {
      ierr = DMNetworkGetVariableGlobalOffset(dm,v,&cstart);CHKERRQ(ierr);
      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]]; /* Jacobian(v,v) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows_v,nrows,cstart,J);CHKERRQ(ierr);
    }
    if (ghost) {
      ierr = PetscFree(rows_v);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Network(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       j;

  PetscFunctionBegin;
  if (--network->refct > 0) PetscFunctionReturn(0);
  if (network->Je) {
    ierr = PetscFree(network->Je);CHKERRQ(ierr);
  }
  if (network->Jv) {
    ierr = PetscFree(network->Jvptr);CHKERRQ(ierr);
    ierr = PetscFree(network->Jv);CHKERRQ(ierr);
  }

  ierr = ISLocalToGlobalMappingDestroy(&network->vertex.mapping);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->vertex.DofSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->vertex.GlobalDofSection);CHKERRQ(ierr);
  if (network->vltog) {
    ierr = PetscFree(network->vltog);CHKERRQ(ierr);
  }
  if (network->vertex.sf) {
    ierr = PetscSFDestroy(&network->vertex.sf);CHKERRQ(ierr);
  }
  /* edge */
  ierr = ISLocalToGlobalMappingDestroy(&network->edge.mapping);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->edge.DofSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->edge.GlobalDofSection);CHKERRQ(ierr);
  if (network->edge.sf) {
    ierr = PetscSFDestroy(&network->edge.sf);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&network->plex);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&network->DofSection);CHKERRQ(ierr);

  for(j=0; j<network->nsubnet; j++) {
    ierr = PetscFree(network->subnet[j].edges);CHKERRQ(ierr);
  }
  ierr = PetscFree(network->subnetvtx);CHKERRQ(ierr);

  ierr = PetscFree(network->subnet);CHKERRQ(ierr);
  ierr = PetscFree(network->componentdataarray);CHKERRQ(ierr);
  ierr = PetscFree2(network->header,network->cvalue);CHKERRQ(ierr);
  ierr = PetscFree(network);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Network(DM dm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscBool      iascii;
  PetscMPIInt    rank;
  PetscInt       p,nsubnet;

  PetscFunctionBegin;
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() first");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  PetscValidHeaderSpecific(dm,DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    const PetscInt    *cone,*vtx,*edges;
    PetscInt          vfrom,vto,i,j,nv,ne;

    nsubnet = network->nsubnet - network->ncsubnet; /* num of subnetworks */
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  [%d] nsubnet: %D; nsubnetCouple: %D; nEdges: %D; nVertices: %D\n",rank,nsubnet,network->ncsubnet,network->nEdges,network->nVertices);CHKERRQ(ierr);

    for (i=0; i<nsubnet; i++) {
      ierr = DMNetworkGetSubnetworkInfo(dm,i,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
      if (ne) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "     Subnet %D: nEdges %D, nVertices %D\n",i,ne,nv);CHKERRQ(ierr);
        for (j=0; j<ne; j++) {
          p = edges[j];
          ierr = DMNetworkGetConnectedVertices(dm,p,&cone);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[0],&vfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[1],&vto);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalEdgeIndex(dm,edges[j],&p);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "       edge %D: %D----> %D\n",p,vfrom,vto);CHKERRQ(ierr);
        }
      }
    }
    /* Coupling subnets */
    nsubnet = network->nsubnet;
    for (; i<nsubnet; i++) {
      ierr = DMNetworkGetSubnetworkInfo(dm,i,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
      if (ne) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "     Subnet %D (couple): nEdges %D, nVertices %D\n",i,ne,nv);CHKERRQ(ierr);
        for (j=0; j<ne; j++) {
          p = edges[j];
          ierr = DMNetworkGetConnectedVertices(dm,p,&cone);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[0],&vfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[1],&vto);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalEdgeIndex(dm,edges[j],&p);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "       edge %D: %D----> %D\n",p,vfrom,vto);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMNetwork writing", ((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalBegin_Network(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(network->plex,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalEnd_Network(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalEnd(network->plex,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalBegin_Network(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalBegin(network->plex,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalEnd_Network(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalEnd(network->plex,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVertexLocalToGlobalOrdering - Get vertex global index

  Not collective

  Input Parameters:
+ dm - the dm object
- vloc - local vertex ordering, start from 0

  Output Parameters:
.  vg  - global vertex ordering, start from 0

  Level: advanced

.seealso: DMNetworkSetVertexLocalToGlobalOrdering()
@*/
PetscErrorCode DMNetworkGetVertexLocalToGlobalOrdering(DM dm,PetscInt vloc,PetscInt *vg)
{
  DM_Network  *network = (DM_Network*)dm->data;
  PetscInt    *vltog = network->vltog;

  PetscFunctionBegin;
  if (!vltog) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Must call DMNetworkSetVertexLocalToGlobalOrdering() first");
  *vg = vltog[vloc];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetVertexLocalToGlobalOrdering - Create and setup vertex local to global map

  Collective

  Input Parameters:
. dm - the dm object

  Level: advanced

.seealso: DMNetworkGetGlobalVertexIndex()
@*/
PetscErrorCode DMNetworkSetVertexLocalToGlobalOrdering(DM dm)
{
  PetscErrorCode    ierr;
  DM_Network        *network=(DM_Network*)dm->data;
  MPI_Comm          comm;
  PetscMPIInt       rank,size,*displs,*recvcounts,remoterank;
  PetscBool         ghost;
  PetscInt          *vltog,nroots,nleaves,i,*vrange,k,N,lidx;
  const PetscSFNode *iremote;
  PetscSF           vsf;
  Vec               Vleaves,Vleaves_seq;
  VecScatter        ctx;
  PetscScalar       *varr,val;
  const PetscScalar *varr_read;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (size == 1) {
    nroots = network->vEnd - network->vStart;
    ierr = PetscMalloc1(nroots, &vltog);CHKERRQ(ierr);
    for (i=0; i<nroots; i++) vltog[i] = i;
    network->vltog = vltog;
    PetscFunctionReturn(0);
  }

  if (!network->distributecalled) SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE,"Must call DMNetworkDistribute() first");
  if (network->vltog) {
    ierr = PetscFree(network->vltog);CHKERRQ(ierr);
  }

  ierr = DMNetworkSetSubMap_private(network->vStart,network->vEnd,&network->vertex.mapping);CHKERRQ(ierr);
  ierr = PetscSFGetSubSF(network->plex->sf, network->vertex.mapping, &network->vertex.sf);CHKERRQ(ierr);
  vsf = network->vertex.sf;

  ierr = PetscMalloc3(size+1,&vrange,size+1,&displs,size,&recvcounts);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(vsf,&nroots,&nleaves,NULL,&iremote);CHKERRQ(ierr);

  for (i=0; i<size; i++) { displs[i] = i; recvcounts[i] = 1;}

  i         = nroots - nleaves; /* local number of vertices, excluding ghosts */
  vrange[0] = 0;
  ierr = MPI_Allgatherv(&i,1,MPIU_INT,vrange+1,recvcounts,displs,MPIU_INT,comm);CHKERRQ(ierr);
  for (i=2; i<size+1; i++) {vrange[i] += vrange[i-1];}

  ierr = PetscMalloc1(nroots, &vltog);CHKERRQ(ierr);
  network->vltog = vltog;

  /* Set vltog for non-ghost vertices */
  k = 0;
  for (i=0; i<nroots; i++) {
    ierr = DMNetworkIsGhostVertex(dm,i+network->vStart,&ghost);CHKERRQ(ierr);
    if (ghost) continue;
    vltog[i] = vrange[rank] + k++;
  }
  ierr = PetscFree3(vrange,displs,recvcounts);CHKERRQ(ierr);

  /* Set vltog for ghost vertices */
  /* (a) create parallel Vleaves and sequential Vleaves_seq to convert local iremote[*].index to global index */
  ierr = VecCreate(comm,&Vleaves);CHKERRQ(ierr);
  ierr = VecSetSizes(Vleaves,2*nleaves,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vleaves);CHKERRQ(ierr);
  ierr = VecGetArray(Vleaves,&varr);CHKERRQ(ierr);
  for (i=0; i<nleaves; i++) {
    varr[2*i]   = (PetscScalar)(iremote[i].rank);  /* rank of remote process */
    varr[2*i+1] = (PetscScalar)(iremote[i].index); /* local index in remote process */
  }
  ierr = VecRestoreArray(Vleaves,&varr);CHKERRQ(ierr);

  /* (b) scatter local info to remote processes via VecScatter() */
  ierr = VecScatterCreateToAll(Vleaves,&ctx,&Vleaves_seq);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,Vleaves,Vleaves_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,Vleaves,Vleaves_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* (c) convert local indices to global indices in parallel vector Vleaves */
  ierr = VecGetSize(Vleaves_seq,&N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vleaves_seq,&varr_read);CHKERRQ(ierr);
  for (i=0; i<N; i+=2) {
    remoterank = (PetscMPIInt)PetscRealPart(varr_read[i]);
    if (remoterank == rank) {
      k = i+1; /* row number */
      lidx = (PetscInt)PetscRealPart(varr_read[i+1]);
      val  = (PetscScalar)vltog[lidx]; /* global index for non-ghost vertex computed above */
      ierr = VecSetValues(Vleaves,1,&k,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(Vleaves_seq,&varr_read);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Vleaves);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Vleaves);CHKERRQ(ierr);

  /* (d) Set vltog for ghost vertices by copying local values of Vleaves */
  ierr = VecGetArrayRead(Vleaves,&varr_read);CHKERRQ(ierr);
  k = 0;
  for (i=0; i<nroots; i++) {
    ierr = DMNetworkIsGhostVertex(dm,i+network->vStart,&ghost);CHKERRQ(ierr);
    if (!ghost) continue;
    vltog[i] = (PetscInt)PetscRealPart(varr_read[2*k+1]); k++;
  }
  ierr = VecRestoreArrayRead(Vleaves,&varr_read);CHKERRQ(ierr);

  ierr = VecDestroy(&Vleaves);CHKERRQ(ierr);
  ierr = VecDestroy(&Vleaves_seq);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
