#include <petsc/private/dmnetworkimpl.h>  /*I  "petscdmnetwork.h"  I*/

PetscLogEvent DMNetwork_LayoutSetUp;
PetscLogEvent DMNetwork_SetUpNetwork;
PetscLogEvent DMNetwork_Distribute;

/*
  Creates the component header and value objects for a network point
*/
static PetscErrorCode SetUpNetworkHeaderComponentValue(DM dm,DMNetworkComponentHeader header,DMNetworkComponentValue cvalue)
{
  PetscFunctionBegin;
  /* Allocate arrays for component information */
  PetscCall(PetscCalloc5(header->maxcomps,&header->size,header->maxcomps,&header->key,header->maxcomps,&header->offset,header->maxcomps,&header->nvar,header->maxcomps,&header->offsetvarrel));
  PetscCall(PetscCalloc1(header->maxcomps,&cvalue->data));

  /* The size of the header is the size of struct _p_DMNetworkComponentHeader. Since the struct contains PetscInt pointers we cannot use sizeof(struct). So, we need to explicitly calculate the size.
   If the data header struct changes then this header size calculation needs to be updated. */
  header->hsize = sizeof(struct _p_DMNetworkComponentHeader) + 5*header->maxcomps*sizeof(PetscInt);
  header->hsize /= sizeof(DMNetworkComponentGenericDataType);
  PetscFunctionReturn(0);
}

PetscErrorCode DMNetworkInitializeHeaderComponentData(DM dm)
{
  DM_Network *network = (DM_Network*)dm->data;
  PetscInt np,p,defaultnumcomp = DMNETWORK_MAX_COMP_AT_POINT_DEFAULT;

  PetscFunctionBegin;
  np = network->cloneshared->pEnd - network->cloneshared->pStart;
  if (network->header)
  for (p=0; p < np; p++) {
    network->header[p].maxcomps = defaultnumcomp;
    PetscCall(SetUpNetworkHeaderComponentValue(dm,&network->header[p],&network->cvalue[p]));
  }

  PetscFunctionReturn(0);
}
/*@
  DMNetworkGetPlex - Gets the Plex DM associated with this network DM

  Not collective

  Input Parameters:
. dm - the dm object

  Output Parameters:
. plexdm - the plex dm object

  Level: Advanced

.seealso: `DMNetworkCreate()`
@*/
PetscErrorCode DMNetworkGetPlex(DM dm,DM *plexdm)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  *plexdm = network->plex;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetNumSubNetworks - Gets the the number of subnetworks

  Not collective

  Input Parameter:
. dm - the dm object

  Output Parameters:
+ nsubnet - local number of subnetworks
- Nsubnet - global number of subnetworks

  Level: beginner

.seealso: `DMNetworkCreate()`, `DMNetworkSetNumSubNetworks()`
@*/
PetscErrorCode DMNetworkGetNumSubNetworks(DM dm,PetscInt *nsubnet,PetscInt *Nsubnet)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (nsubnet) *nsubnet = network->cloneshared->nsubnet;
  if (Nsubnet) *Nsubnet = network->cloneshared->Nsubnet;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetNumSubNetworks - Sets the number of subnetworks

  Collective on dm

  Input Parameters:
+ dm - the dm object
. nsubnet - local number of subnetworks
- Nsubnet - global number of subnetworks

   Level: beginner

.seealso: `DMNetworkCreate()`, `DMNetworkGetNumSubNetworks()`
@*/
PetscErrorCode DMNetworkSetNumSubNetworks(DM dm,PetscInt nsubnet,PetscInt Nsubnet)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCheck(network->cloneshared->Nsubnet == 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Network sizes alread set, cannot resize the network");

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(dm,nsubnet,2);
  PetscValidLogicalCollectiveInt(dm,Nsubnet,3);

  if (Nsubnet == PETSC_DECIDE) {
    PetscCheck(nsubnet >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of local subnetworks %" PetscInt_FMT " cannot be less than 0",nsubnet);
    PetscCall(MPIU_Allreduce(&nsubnet,&Nsubnet,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm)));
  }
  PetscCheck(Nsubnet >= 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Number of global subnetworks %" PetscInt_FMT " cannot be less than 1",Nsubnet);

  network->cloneshared->Nsubnet  = Nsubnet;
  network->cloneshared->nsubnet  = 0;       /* initial value; will be determind by DMNetworkAddSubnetwork() */
  PetscCall(PetscCalloc1(Nsubnet,&network->cloneshared->subnet));

  /* num of shared vertices */
  network->cloneshared->nsvtx = 0;
  network->cloneshared->Nsvtx = 0;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAddSubnetwork - Add a subnetwork

  Collective on dm

  Input Parameters:
+ dm - the dm object
. name - name of the subnetwork
. ne - number of local edges of this subnetwork
- edgelist - list of edges for this subnetwork, this is a one dimensional array with pairs of entries being the two vertices (in global numbering of the vertices) of each edge,
$            [first vertex of first edge, second vertex of first edge, first vertex of second edge, second vertex of second edge, etc]

  Output Parameters:
. netnum - global index of the subnetwork

  Notes:
  There is no copy involved in this operation, only the pointer is referenced. The edgelist should
  not be destroyed before the call to DMNetworkLayoutSetUp()

  A network can comprise of a single subnetwork OR multiple subnetworks. For a single subnetwork, the subnetwork can be read either in serial or parallel. For a multiple subnetworks,
  each subnetwork topology needs to be set on a unique rank and the communicator size needs to be at least equal to the number of subnetworks.

  Level: beginner

  Example usage:
  Consider the following networks:
  1) A sigle subnetwork:
.vb
 network 0:
 rank[0]:
   v0 --> v2; v1 --> v2
 rank[1]:
   v3 --> v5; v4 --> v5
.ve

 The resulting input
 network 0:
 rank[0]:
   ne = 2
   edgelist = [0 2 | 1 2]
 rank[1]:
   ne = 2
   edgelist = [3 5 | 4 5]

  2) Two subnetworks:
.vb
 subnetwork 0:
 rank[0]:
   v0 --> v2; v2 --> v1; v1 --> v3;
 subnetwork 1:
 rank[1]:
   v0 --> v3; v3 --> v2; v2 --> v1;
.ve

 The resulting input
 subnetwork 0:
 rank[0]:
   ne = 3
   edgelist = [0 2 | 2 1 | 1 3]
 rank[1]:
   ne = 0
   edgelist = NULL

 subnetwork 1:
 rank[0]:
   ne = 0
   edgelist = NULL
 rank[1]:
   edgelist = [0 3 | 3 2 | 2 1]

.seealso: `DMNetworkCreate()`, `DMNetworkSetNumSubnetworks()`
@*/
PetscErrorCode DMNetworkAddSubnetwork(DM dm,const char* name,PetscInt ne,PetscInt edgelist[],PetscInt *netnum)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,Nedge,j,Nvtx,nvtx,nvtx_min=-1,nvtx_max=0;
  PetscBT        table;

  PetscFunctionBegin;
  for (i=0; i<ne; i++) {
    PetscCheck(edgelist[2*i] != edgelist[2*i+1],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Edge %" PetscInt_FMT " has the same vertex %" PetscInt_FMT " at each endpoint",i,edgelist[2*i]);
  }

  i = 0;
  if (ne) nvtx_min = nvtx_max = edgelist[0];
  for (j=0; j<ne; j++) {
    nvtx_min = PetscMin(nvtx_min, edgelist[i]);
    nvtx_max = PetscMax(nvtx_max, edgelist[i]);
    i++;
    nvtx_min = PetscMin(nvtx_min, edgelist[i]);
    nvtx_max = PetscMax(nvtx_max, edgelist[i]);
    i++;
  }
  Nvtx = nvtx_max - nvtx_min + 1; /* approximated total local nvtx for this subnet */

  /* Get exact local nvtx for this subnet: counting local values between nvtx_min and nvtx_max */
  PetscCall(PetscBTCreate(Nvtx,&table));
  PetscCall(PetscBTMemzero(Nvtx,table));
  i = 0;
  for (j=0; j<ne; j++) {
    PetscCall(PetscBTSet(table,edgelist[i++]-nvtx_min));
    PetscCall(PetscBTSet(table,edgelist[i++]-nvtx_min));
  }
  nvtx = 0;
  for (j=0; j<Nvtx; j++) {
    if (PetscBTLookup(table,j)) nvtx++;
  }

  /* Get global total Nvtx = max(edgelist[])+1 for this subnet */
  PetscCall(MPIU_Allreduce(&nvtx_max,&Nvtx,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm)));
  Nvtx++;
  PetscCall(PetscBTDestroy(&table));

  /* Get global total Nedge for this subnet */
  PetscCall(MPIU_Allreduce(&ne,&Nedge,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm)));

  i = network->cloneshared->nsubnet;
  if (name) {
    PetscCall(PetscStrcpy(network->cloneshared->subnet[i].name,name));
  }
  network->cloneshared->subnet[i].nvtx     = nvtx; /* include ghost vertices */
  network->cloneshared->subnet[i].nedge    = ne;
  network->cloneshared->subnet[i].edgelist = edgelist;
  network->cloneshared->subnet[i].Nvtx     = Nvtx;
  network->cloneshared->subnet[i].Nedge    = Nedge;

  /* ----------------------------------------------------------
   p=v or e;
   subnet[0].pStart   = 0
   subnet[i+1].pStart = subnet[i].pEnd = subnet[i].pStart + (nE[i] or NV[i])
   ----------------------------------------------------------------------- */
  /* GLOBAL subnet[].vStart and vEnd, used by DMNetworkLayoutSetUp() */
  network->cloneshared->subnet[i].vStart = network->cloneshared->NVertices;
  network->cloneshared->subnet[i].vEnd   = network->cloneshared->subnet[i].vStart + network->cloneshared->subnet[i].Nvtx; /* global vEnd of subnet[i] */

  network->cloneshared->nVertices += nvtx; /* include ghost vertices */
  network->cloneshared->NVertices += network->cloneshared->subnet[i].Nvtx;

  /* LOCAL subnet[].eStart and eEnd, used by DMNetworkLayoutSetUp() */
  network->cloneshared->subnet[i].eStart = network->cloneshared->nEdges;
  network->cloneshared->subnet[i].eEnd   = network->cloneshared->subnet[i].eStart + ne;
  network->cloneshared->nEdges += ne;
  network->cloneshared->NEdges += network->cloneshared->subnet[i].Nedge;

  PetscCall(PetscStrcpy(network->cloneshared->subnet[i].name,name));
  if (netnum) *netnum = network->cloneshared->nsubnet;
  network->cloneshared->nsubnet++;
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkSharedVertexGetInfo - Get info of a shared vertex struct, see petsc/private/dmnetworkimpl.h

  Not collective

  Input Parameters:
+ dm - the DM object
- v - vertex point

  Output Parameters:
+ gidx - global number of this shared vertex in the internal dmplex
. n - number of subnetworks that share this vertex
- sv - array of size n: sv[2*i,2*i+1]=(net[i], idx[i]), i=0,...,n-1

  Level: intermediate

.seealso: `DMNetworkGetSharedVertices()`
@*/
PetscErrorCode DMNetworkSharedVertexGetInfo(DM dm,PetscInt v,PetscInt *gidx,PetscInt *n,const PetscInt **sv)
{
  DM_Network     *network = (DM_Network*)dm->data;
  SVtx           *svtx = network->cloneshared->svtx;
  PetscInt       i,gidx_tmp;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetGlobalVertexIndex(dm,v,&gidx_tmp));
  PetscCall(PetscTableFind(network->cloneshared->svtable,gidx_tmp+1,&i));
  PetscCheck(i > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"input vertex is not a shared vertex");

  i--;
  if (gidx) *gidx = gidx_tmp;
  if (n)    *n    = svtx[i].n;
  if (sv)   *sv   = svtx[i].sv;
  PetscFunctionReturn(0);
}

/*
  VtxGetInfo - Get info of an input vertex=(net,idx)

  Input Parameters:
+ Nsvtx - global num of shared vertices
. svtx - array of shared vertices (global)
- (net,idx) - subnet number and local index for a vertex

  Output Parameters:
+ gidx - global index of (net,idx)
. svtype - see petsc/private/dmnetworkimpl.h
- svtx_idx - ordering in the svtx array
*/
static inline PetscErrorCode VtxGetInfo(PetscInt Nsvtx,SVtx *svtx,PetscInt net,PetscInt idx,PetscInt *gidx,SVtxType *svtype,PetscInt *svtx_idx)
{
  PetscInt i,j,*svto,g_idx;
  SVtxType vtype;

  PetscFunctionBegin;
  if (!Nsvtx) PetscFunctionReturn(0);

  g_idx = -1;
  vtype = SVNONE;

  for (i=0; i<Nsvtx; i++) {
    if (net == svtx[i].sv[0] && idx == svtx[i].sv[1]) {
      g_idx = svtx[i].gidx;
      vtype = SVFROM;
    } else { /* loop over svtx[i].n */
      for (j=1; j<svtx[i].n; j++) {
        svto = svtx[i].sv + 2*j;
        if (net == svto[0] && idx == svto[1]) {
          /* input vertex net.idx is a shared to_vertex, output its global index and its svtype */
          g_idx = svtx[i].gidx; /* output gidx for to_vertex */
          vtype = SVTO;
        }
      }
    }
    if (vtype != SVNONE) break;
  }
  if (gidx)     *gidx     = g_idx;
  if (svtype)   *svtype   = vtype;
  if (svtx_idx) *svtx_idx = i;
  PetscFunctionReturn(0);
}

/*
  TableAddSVtx - Add a new shared vertice from sedgelist[k] to a ctable svta

  Input:  network, sedgelist, k, svta
  Output: svta, tdata, ta2sv
*/
static inline PetscErrorCode TableAddSVtx(DM_Network *network,PetscInt *sedgelist,PetscInt k,PetscTable svta,PetscInt* tdata,PetscInt *ta2sv)
{
  PetscInt       net,idx,gidx;

  PetscFunctionBegin;
  net = sedgelist[k];
  idx = sedgelist[k+1];
  gidx = network->cloneshared->subnet[net].vStart + idx;
  PetscCall(PetscTableAdd(svta,gidx+1,*tdata+1,INSERT_VALUES));

  ta2sv[*tdata] = k; /* maps tdata to index of sedgelist */
  (*tdata)++;
  PetscFunctionReturn(0);
}

/*
  SharedVtxCreate - Create an array of global shared vertices. See SVtx and SVtxType in dmnetworkimpl.h

  Input:  dm, Nsedgelist, sedgelist

  Note: Output svtx is organized as
        sv(net[0],idx[0]) --> sv(net[1],idx[1])
                          --> sv(net[1],idx[1])
                          ...
                          --> sv(net[n-1],idx[n-1])
        and net[0] < net[1] < ... < net[n-1]
        where sv[0] has SVFROM type, sv[i], i>0, has SVTO type.
 */
static PetscErrorCode SharedVtxCreate(DM dm,PetscInt Nsedgelist,PetscInt *sedgelist)
{
  SVtx               *svtx = NULL;
  PetscInt           *sv,k,j,nsv,*tdata,**ta2sv;
  PetscTable         *svtas;
  PetscInt           gidx,net,idx,i,nta,ita,idx_from,idx_to,n;
  DM_Network         *network = (DM_Network*)dm->data;
  PetscTablePosition ppos;

  PetscFunctionBegin;
  /* (1) Crete an array of ctables svtas to map (net,idx) -> gidx; a svtas[] for a shared/merged vertex */
  PetscCall(PetscCalloc3(Nsedgelist,&svtas,Nsedgelist,&tdata,2*Nsedgelist,&ta2sv));

  k   = 0;   /* sedgelist vertex counter j = 4*k */
  nta = 0;   /* num of svta tables created */

  /* for j=0 */
  PetscCall(PetscTableCreate(2*Nsedgelist,network->cloneshared->NVertices+1,&svtas[nta]));
  PetscCall(PetscMalloc1(2*Nsedgelist,&ta2sv[nta]));

  PetscCall(TableAddSVtx(network,sedgelist,k,svtas[nta],&tdata[nta],ta2sv[nta]));
  PetscCall(TableAddSVtx(network,sedgelist,k+2,svtas[nta],&tdata[nta],ta2sv[nta]));
  nta++; k += 4;

  for (j = 1; j < Nsedgelist; j++) { /* j: sedgelist counter */
    for (ita = 0; ita < nta; ita++) {
      /* vfrom */
      net = sedgelist[k]; idx = sedgelist[k+1];
      gidx = network->cloneshared->subnet[net].vStart + idx; /* global index of the vertex net.idx before merging shared vertices */
      PetscCall(PetscTableFind(svtas[ita],gidx+1,&idx_from));

      /* vto */
      net = sedgelist[k+2]; idx = sedgelist[k+3];
      gidx = network->cloneshared->subnet[net].vStart + idx;
      PetscCall(PetscTableFind(svtas[ita],gidx+1,&idx_to));

      if (idx_from || idx_to) { /* vfrom or vto is on table svtas[ita] */
        idx_from--; idx_to--;
        if (idx_from < 0) { /* vto is on svtas[ita] */
          PetscCall(TableAddSVtx(network,sedgelist,k,svtas[ita],&tdata[ita],ta2sv[ita]));
          break;
        } else if (idx_to < 0) {
          PetscCall(TableAddSVtx(network,sedgelist,k+2,svtas[ita],&tdata[ita],ta2sv[ita]));
          break;
        }
      }
    }

    if (ita == nta) {
      PetscCall(PetscTableCreate(2*Nsedgelist,network->cloneshared->NVertices+1,&svtas[nta]));
      PetscCall(PetscMalloc1(2*Nsedgelist, &ta2sv[nta]));

      PetscCall(TableAddSVtx(network,sedgelist,k,svtas[nta],&tdata[nta],ta2sv[nta]));
      PetscCall(TableAddSVtx(network,sedgelist,k+2,svtas[nta],&tdata[nta],ta2sv[nta]));
      nta++;
    }
    k += 4;
  }

  /* (2) Create svtable for query shared vertices using gidx */
  PetscCall(PetscTableCreate(nta,network->cloneshared->NVertices+1,&network->cloneshared->svtable));

  /* (3) Construct svtx from svtas
     svtx: array of SVtx: sv[0]=(net[0],idx[0]) to vertices sv[k], k=1,...,n-1;
     net[k], k=0, ...,n-1, are in ascending order */
  PetscCall(PetscMalloc1(nta,&svtx));
  for (nsv = 0; nsv < nta; nsv++) {
    /* for a single svtx, put shared vertices in ascending order of gidx */
    PetscCall(PetscTableGetCount(svtas[nsv],&n));
    PetscCall(PetscCalloc1(2*n,&sv));
    svtx[nsv].sv   = sv;
    svtx[nsv].n    = n;
    svtx[nsv].gidx = network->cloneshared->NVertices; /* initialization */

    PetscCall(PetscTableGetHeadPosition(svtas[nsv],&ppos));
    for (k=0; k<n; k++) { /* gidx is sorted in ascending order */
      PetscCall(PetscTableGetNext(svtas[nsv],&ppos,&gidx,&i));
      gidx--; i--;

      if (svtx[nsv].gidx > gidx) svtx[nsv].gidx = gidx; /*svtx[nsv].gidx = min(gidx) */

      j = ta2sv[nsv][i]; /* maps i to index of sedgelist */
      sv[2*k]   = sedgelist[j];   /* subnet number */
      sv[2*k+1] = sedgelist[j+1]; /* index on the subnet */
    }

    /* Setup svtable for query shared vertices */
    PetscCall(PetscTableAdd(network->cloneshared->svtable,svtx[nsv].gidx+1,nsv+1,INSERT_VALUES));
  }

  for (j=0; j<nta; j++) {
    PetscCall(PetscTableDestroy(&svtas[j]));
    PetscCall(PetscFree(ta2sv[j]));
  }
  PetscCall(PetscFree3(svtas,tdata,ta2sv));

  network->cloneshared->Nsvtx = nta;
  network->cloneshared->svtx  = svtx;
  PetscFunctionReturn(0);
}

/*
  GetEdgelist_Coupling - Get an integrated edgelist for dmplex from user-provided subnet[].edgelist when subnets are coupled by shared vertices

  Input Parameters:
. dm - the dmnetwork object

   Output Parameters:
+  edges - the integrated edgelist for dmplex
-  nmerged_ptr - num of vertices being merged
*/
static PetscErrorCode GetEdgelist_Coupling(DM dm,PetscInt *edges,PetscInt *nmerged_ptr)
{
  MPI_Comm       comm;
  PetscMPIInt    size,rank,*recvcounts=NULL,*displs=NULL;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,j,ctr,np;
  PetscInt       *vidxlTog,Nsv,Nsubnet=network->cloneshared->Nsubnet;
  PetscInt       *sedgelist=network->cloneshared->sedgelist;
  PetscInt       net,idx,gidx,nmerged,*vrange,gidx_from,net_from,sv_idx;
  SVtxType       svtype = SVNONE;
  SVtx           *svtx;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  /* (1) Create global svtx[] from sedgelist */
  /* --------------------------------------- */
  PetscCall(SharedVtxCreate(dm,network->cloneshared->Nsvtx,sedgelist));
  Nsv  = network->cloneshared->Nsvtx;
  svtx = network->cloneshared->svtx;

  /* (2) Merge shared vto vertices to their vfrom vertex with same global vetex index (gidx) */
  /* --------------------------------------------------------------------------------------- */
  /* (2.1) compute vrage[rank]: global index of 1st local vertex in proc[rank] */
  PetscCall(PetscMalloc4(size+1,&vrange,size,&displs,size,&recvcounts,network->cloneshared->nVertices,&vidxlTog));
  for (i=0; i<size; i++) {displs[i] = i; recvcounts[i] = 1;}

  vrange[0] = 0;
  PetscCallMPI(MPI_Allgatherv(&network->cloneshared->nVertices,1,MPIU_INT,vrange+1,recvcounts,displs,MPIU_INT,comm));
  for (i=2; i<size+1; i++) vrange[i] += vrange[i-1];

  /* (2.2) Create vidxlTog: maps UN-MERGED local vertex index i to global index gidx (plex, excluding ghost vertices) */
  i = 0; gidx = 0;
  nmerged        = 0; /* local num of merged vertices */
  network->cloneshared->nsvtx = 0; /* local num of SVtx structs, including ghosts */
  for (net=0; net<Nsubnet; net++) {
    for (idx=0; idx<network->cloneshared->subnet[net].Nvtx; idx++) { /* Note: global subnet[net].Nvtx */
      PetscCall(VtxGetInfo(Nsv,svtx,net,idx,&gidx_from,&svtype,&sv_idx));
      if (svtype == SVTO) {
        if (network->cloneshared->subnet[net].nvtx) {/* this proc owns sv_to */
          net_from = svtx[sv_idx].sv[0]; /* subnet number of its shared vertex */
          if (network->cloneshared->subnet[net_from].nvtx == 0) {
            /* this proc does not own v_from, thus a ghost local vertex */
            network->cloneshared->nsvtx++;
          }
          vidxlTog[i++] = gidx_from; /* gidx before merging! Bug??? */
          nmerged++; /* a shared vertex -- merged */
        }
      } else {
        if (svtype == SVFROM && network->cloneshared->subnet[net].nvtx) {
          /* this proc owns this v_from, a new local shared vertex */
          network->cloneshared->nsvtx++;
        }
        if (network->cloneshared->subnet[net].nvtx) vidxlTog[i++] = gidx;
        gidx++;
      }
    }
  }
#if defined(PETSC_USE_DEBUG)
  PetscCheck(i == network->cloneshared->nVertices,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"%" PetscInt_FMT " != %" PetscInt_FMT " nVertices",i,network->cloneshared->nVertices);
#endif

  /* (2.3) Shared vertices in the subnetworks are merged, update global NVertices: np = sum(local nmerged) */
  PetscCallMPI(MPI_Allreduce(&nmerged,&np,1,MPIU_INT,MPI_SUM,comm));
  network->cloneshared->NVertices -= np;

  ctr = 0;
  for (net=0; net<Nsubnet; net++) {
    for (j = 0; j < network->cloneshared->subnet[net].nedge; j++) {
      /* vfrom: */
      i = network->cloneshared->subnet[net].edgelist[2*j] + (network->cloneshared->subnet[net].vStart - vrange[rank]);
      edges[2*ctr] = vidxlTog[i];

      /* vto */
      i = network->cloneshared->subnet[net].edgelist[2*j+1] + (network->cloneshared->subnet[net].vStart - vrange[rank]);
      edges[2*ctr+1] = vidxlTog[i];
      ctr++;
    }
  }
  PetscCall(PetscFree4(vrange,displs,recvcounts,vidxlTog));
  PetscCall(PetscFree(sedgelist)); /* created in DMNetworkAddSharedVertices() */

  *nmerged_ptr = nmerged;
  PetscFunctionReturn(0);
}

PetscErrorCode DMNetworkInitializeNonTopological(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       p,pStart = network->cloneshared->pStart, pEnd = network->cloneshared->pEnd;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));

  PetscCall(PetscSectionCreate(comm,&network->DataSection));
  PetscCall(PetscSectionCreate(comm,&network->DofSection));
  PetscCall(PetscSectionSetChart(network->DataSection,pStart,pEnd));
  PetscCall(PetscSectionSetChart(network->DofSection,pStart,pEnd));

  PetscCall(DMNetworkInitializeHeaderComponentData(dm));

  for (p = 0; p <pEnd-pStart; p++) {
    network->header[p].ndata           = 0;
    network->header[p].offset[0]       = 0;
    network->header[p].offsetvarrel[0] = 0;
    PetscCall(PetscSectionAddDof(network->DataSection,p,network->header[p].hsize));
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkLayoutSetUp - Sets up the bare layout (graph) for the network

  Not Collective

  Input Parameters:
. dm - the dmnetwork object

  Notes:
  This routine should be called after the network sizes and edgelists have been provided. It creates
  the bare layout of the network and sets up the network to begin insertion of components.

  All the components should be registered before calling this routine.

  Level: beginner

.seealso: `DMNetworkSetNumSubNetworks()`, `DMNetworkAddSubnetwork()`
@*/
PetscErrorCode DMNetworkLayoutSetUp(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,j,ctr,Nsubnet=network->cloneshared->Nsubnet,np,*edges,*subnetvtx,*subnetedge,e,v,vfrom,vto,net,globaledgeoff;
  const PetscInt *cone;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscSection   sectiong;
  PetscInt       nmerged=0;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMNetwork_LayoutSetUp,dm,0,0,0));
  PetscCheck(network->cloneshared->nsubnet == Nsubnet,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Must call DMNetworkAddSubnetwork() %" PetscInt_FMT " times",Nsubnet);

  /* This implementation requires user input each subnet by a single processor when Nsubnet>1, thus subnet[net].nvtx=subnet[net].Nvtx when net>0 */
  for (net=1; net<Nsubnet; net++) {
    if (network->cloneshared->subnet[net].nvtx) PetscCheck(network->cloneshared->subnet[net].nvtx == network->cloneshared->subnet[net].Nvtx,PETSC_COMM_SELF,PETSC_ERR_SUP,"subnetwork %" PetscInt_FMT " local num of vertices %" PetscInt_FMT " != %" PetscInt_FMT " global num",net,network->cloneshared->subnet[net].nvtx,network->cloneshared->subnet[net].Nvtx);
  }

  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(MPI_Comm_size(comm,&size));

  /* Create LOCAL edgelist in global vertex ordering for the network by concatenating local input edgelists of the subnetworks */
  PetscCall(PetscCalloc1(2*network->cloneshared->nEdges,&edges));

  if (network->cloneshared->Nsvtx) { /* subnetworks are coupled via shared vertices */
    PetscCall(GetEdgelist_Coupling(dm,edges,&nmerged));
  } else { /* subnetworks are not coupled */
    /* Create a 0-size svtable for query shared vertices */
    PetscCall(PetscTableCreate(0,network->cloneshared->NVertices+1,&network->cloneshared->svtable));
    ctr = 0;
    for (i=0; i < Nsubnet; i++) {
      for (j = 0; j < network->cloneshared->subnet[i].nedge; j++) {
        edges[2*ctr]   = network->cloneshared->subnet[i].vStart + network->cloneshared->subnet[i].edgelist[2*j];
        edges[2*ctr+1] = network->cloneshared->subnet[i].vStart + network->cloneshared->subnet[i].edgelist[2*j+1];
        ctr++;
      }
    }
  }

  /* Create network->plex; One dimensional network, numCorners=2 */
  PetscCall(DMCreate(comm,&network->plex));
  PetscCall(DMSetType(network->plex,DMPLEX));
  PetscCall(DMSetDimension(network->plex,1));

  if (size == 1) PetscCall(DMPlexBuildFromCellList(network->plex,network->cloneshared->nEdges,PETSC_DECIDE,2,edges));
  else PetscCall(DMPlexBuildFromCellListParallel(network->plex,network->cloneshared->nEdges,PETSC_DECIDE,PETSC_DECIDE,2,edges,NULL, NULL));

  PetscCall(DMPlexGetChart(network->plex,&network->cloneshared->pStart,&network->cloneshared->pEnd));
  PetscCall(DMPlexGetHeightStratum(network->plex,0,&network->cloneshared->eStart,&network->cloneshared->eEnd));
  PetscCall(DMPlexGetHeightStratum(network->plex,1,&network->cloneshared->vStart,&network->cloneshared->vEnd));
  np = network->cloneshared->pEnd - network->cloneshared->pStart;
  PetscCall(PetscCalloc2(np,&network->header,np,&network->cvalue));

  /* Create edge and vertex arrays for the subnetworks
     This implementation assumes that DMNetwork reads
     (1) a single subnetwork in parallel; or
     (2) n subnetworks using n processors, one subnetwork/processor.
  */
  PetscCall(PetscCalloc2(network->cloneshared->nEdges,&subnetedge,network->cloneshared->nVertices+network->cloneshared->nsvtx,&subnetvtx)); /* Maps local edge/vertex to local subnetwork's edge/vertex */
  network->cloneshared->subnetedge = subnetedge;
  network->cloneshared->subnetvtx  = subnetvtx;
  for (j=0; j < Nsubnet; j++) {
    network->cloneshared->subnet[j].edges = subnetedge;
    subnetedge              += network->cloneshared->subnet[j].nedge;

    network->cloneshared->subnet[j].vertices = subnetvtx;
    subnetvtx                  += network->cloneshared->subnet[j].nvtx;
  }
  network->cloneshared->svertices = subnetvtx;

  /* Get edge ownership */
  np = network->cloneshared->eEnd - network->cloneshared->eStart;
  PetscCallMPI(MPI_Scan(&np,&globaledgeoff,1,MPIU_INT,MPI_SUM,comm));
  globaledgeoff -= np;

  /* Setup local edge and vertex arrays for subnetworks */
  e = 0;
  for (i=0; i < Nsubnet; i++) {
    ctr = 0;
    for (j = 0; j < network->cloneshared->subnet[i].nedge; j++) {
      /* edge e */
      network->header[e].index    = e + globaledgeoff;   /* Global edge index */
      network->header[e].subnetid = i;
      network->cloneshared->subnet[i].edges[j] = e;

      /* connected vertices */
      PetscCall(DMPlexGetCone(network->plex,e,&cone));

      /* vertex cone[0] */
      v = cone[0];
      network->header[v].index     = edges[2*e];  /* Global vertex index */
      network->header[v].subnetid  = i;           /* Subnetwork id */
      if (Nsubnet == 1) {
        network->cloneshared->subnet[i].vertices[v - network->cloneshared->vStart] = v; /* user's subnet[].idx = petsc's v */
      } else {
        vfrom = network->cloneshared->subnet[i].edgelist[2*ctr];     /* =subnet[i].idx, Global index! */
        network->cloneshared->subnet[i].vertices[vfrom] = v; /* user's subnet[].dix = petsc's v */
      }

      /* vertex cone[1] */
      v = cone[1];
      network->header[v].index    = edges[2*e+1];   /* Global vertex index */
      network->header[v].subnetid = i;              /* Subnetwork id */
      if (Nsubnet == 1) {
        network->cloneshared->subnet[i].vertices[v - network->cloneshared->vStart] = v; /* user's subnet[].idx = petsc's v */
      } else {
        vto = network->cloneshared->subnet[i].edgelist[2*ctr+1];     /* =subnet[i].idx, Global index! */
        network->cloneshared->subnet[i].vertices[vto] = v; /* user's subnet[].dix = petsc's v */
      }

      e++; ctr++;
    }
  }
  PetscCall(PetscFree(edges));

  /* Set local vertex array for the subnetworks */
  j = 0;
  for (v = network->cloneshared->vStart; v < network->cloneshared->vEnd; v++) {
    /* local shared vertex */
    PetscCall(PetscTableFind(network->cloneshared->svtable,network->header[v].index+1,&i));
    if (i) network->cloneshared->svertices[j++] = v;
  }

  /* Create a global section to be used by DMNetworkIsGhostVertex() which is a non-collective routine */
  /* see snes_tutorials_network-ex1_4 */
  PetscCall(DMGetGlobalSection(network->plex,&sectiong));
  /* Initialize non-topological data structures  */
  PetscCall(DMNetworkInitializeNonTopological(dm));
  PetscCall(PetscLogEventEnd(DMNetwork_LayoutSetUp,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSubnetwork - Returns the information about a requested subnetwork

  Not collective

  Input Parameters:
+ dm - the DM object
- netnum - the global index of the subnetwork

  Output Parameters:
+ nv - number of vertices (local)
. ne - number of edges (local)
. vtx - local vertices of the subnetwork
- edge - local edges of the subnetwork

  Notes:
    Cannot call this routine before DMNetworkLayoutSetup()

    The local vertices returned on each rank are determined by DMNetwork. The user does not have any control over what vertices are local.

  Level: intermediate

.seealso: `DMNetworkCreate()`, `DMNetworkAddSubnetwork()`, `DMNetworkLayoutSetUp()`
@*/
PetscErrorCode DMNetworkGetSubnetwork(DM dm,PetscInt netnum,PetscInt *nv,PetscInt *ne,const PetscInt **vtx,const PetscInt **edge)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCheck(netnum < network->cloneshared->Nsubnet,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subnet index %" PetscInt_FMT " exceeds the num of subnets %" PetscInt_FMT,netnum,network->cloneshared->Nsubnet);
  if (nv) *nv     = network->cloneshared->subnet[netnum].nvtx;
  if (ne) *ne     = network->cloneshared->subnet[netnum].nedge;
  if (vtx) *vtx   = network->cloneshared->subnet[netnum].vertices;
  if (edge) *edge = network->cloneshared->subnet[netnum].edges;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAddSharedVertices - Add shared vertices that connect two given subnetworks

  Collective on dm

  Input Parameters:
+ dm - the dm object
. anetnum - first subnetwork global numbering returned by DMNetworkAddSubnetwork()
. bnetnum - second subnetwork global numbering returned by DMNetworkAddSubnetwork()
. nsvtx - number of vertices that are shared by the two subnetworks
. asvtx - vertex index in the first subnetwork
- bsvtx - vertex index in the second subnetwork

  Level: beginner

.seealso: `DMNetworkCreate()`, `DMNetworkAddSubnetwork()`, `DMNetworkGetSharedVertices()`
@*/
PetscErrorCode DMNetworkAddSharedVertices(DM dm,PetscInt anetnum,PetscInt bnetnum,PetscInt nsvtx,PetscInt asvtx[],PetscInt bsvtx[])
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,nsubnet = network->cloneshared->Nsubnet,*sedgelist,Nsvtx=network->cloneshared->Nsvtx;

  PetscFunctionBegin;
  PetscCheck(anetnum != bnetnum,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Subnetworks must have different netnum");
  PetscCheck(anetnum >= 0 && bnetnum >= 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"netnum cannot be negative");
  if (!Nsvtx) {
    /* allocate network->sedgelist to hold at most 2*nsubnet pairs of shared vertices */
    PetscCall(PetscMalloc1(2*4*nsubnet,&network->cloneshared->sedgelist));
  }

  sedgelist = network->cloneshared->sedgelist;
  for (i=0; i<nsvtx; i++) {
    sedgelist[4*Nsvtx]   = anetnum; sedgelist[4*Nsvtx+1] = asvtx[i];
    sedgelist[4*Nsvtx+2] = bnetnum; sedgelist[4*Nsvtx+3] = bsvtx[i];
    Nsvtx++;
  }
  PetscCheck(Nsvtx <= 2*nsubnet,PETSC_COMM_SELF,PETSC_ERR_SUP,"allocate more space for coupling edgelist");
  network->cloneshared->Nsvtx = Nsvtx;
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSharedVertices - Returns the info for the shared vertices

  Not collective

  Input Parameter:
. dm - the DM object

  Output Parameters:
+ nsv - number of local shared vertices
- svtx - local shared vertices

  Notes:
  Cannot call this routine before DMNetworkLayoutSetup()

  Level: intermediate

.seealso: `DMNetworkGetSubnetwork()`, `DMNetworkLayoutSetUp()`, `DMNetworkAddSharedVertices()`
@*/
PetscErrorCode DMNetworkGetSharedVertices(DM dm,PetscInt *nsv,const PetscInt **svtx)
{
  DM_Network *net = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (nsv)  *nsv  = net->cloneshared->nsvtx;
  if (svtx) *svtx = net->cloneshared->svertices;
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkRegisterComponent - Registers the network component

  Logically collective on dm

  Input Parameters:
+ dm - the network object
. name - the component name
- size - the storage size in bytes for this component data

   Output Parameters:
.  key - an integer key that defines the component

   Notes
   This routine should be called by all processors before calling DMNetworkLayoutSetup().

   Level: beginner

.seealso: `DMNetworkCreate()`, `DMNetworkLayoutSetUp()`
@*/
PetscErrorCode DMNetworkRegisterComponent(DM dm,const char *name,size_t size,PetscInt *key)
{
  DM_Network            *network = (DM_Network*) dm->data;
  DMNetworkComponent    *component=NULL,*newcomponent=NULL;
  PetscBool             flg=PETSC_FALSE;
  PetscInt              i;

  PetscFunctionBegin;
  if (!network->component) {
    PetscCall(PetscCalloc1(network->max_comps_registered,&network->component));
  }

  for (i=0; i < network->ncomponent; i++) {
    PetscCall(PetscStrcmp(network->component[i].name,name,&flg));
    if (flg) {
      *key = i;
      PetscFunctionReturn(0);
    }
  }

  if (network->ncomponent == network->max_comps_registered) {
    /* Reached max allowed so resize component */
    network->max_comps_registered += 2;
    PetscCall(PetscCalloc1(network->max_comps_registered,&newcomponent));
    /* Copy over the previous component info */
    for (i=0; i < network->ncomponent; i++) {
      PetscCall(PetscStrcpy(newcomponent[i].name,network->component[i].name));
      newcomponent[i].size = network->component[i].size;
    }
    /* Free old one */
    PetscCall(PetscFree(network->component));
    /* Update pointer */
    network->component = newcomponent;
  }

  component = &network->component[network->ncomponent];

  PetscCall(PetscStrcpy(component->name,name));
  component->size = size/sizeof(DMNetworkComponentGenericDataType);
  *key = network->ncomponent;
  network->ncomponent++;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVertexRange - Get the bounds [start, end) for the local vertices

  Not Collective

  Input Parameter:
. dm - the DMNetwork object

  Output Parameters:
+ vStart - the first vertex point
- vEnd - one beyond the last vertex point

  Level: beginner

.seealso: `DMNetworkGetEdgeRange()`
@*/
PetscErrorCode DMNetworkGetVertexRange(DM dm,PetscInt *vStart,PetscInt *vEnd)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (vStart) *vStart = network->cloneshared->vStart;
  if (vEnd) *vEnd = network->cloneshared->vEnd;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetEdgeRange - Get the bounds [start, end) for the local edges

  Not Collective

  Input Parameter:
. dm - the DMNetwork object

  Output Parameters:
+ eStart - The first edge point
- eEnd - One beyond the last edge point

  Level: beginner

.seealso: `DMNetworkGetVertexRange()`
@*/
PetscErrorCode DMNetworkGetEdgeRange(DM dm,PetscInt *eStart,PetscInt *eEnd)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (eStart) *eStart = network->cloneshared->eStart;
  if (eEnd)   *eEnd   = network->cloneshared->eEnd;
  PetscFunctionReturn(0);
}

PetscErrorCode DMNetworkGetIndex(DM dm,PetscInt p,PetscInt *index)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (network->header) {
    *index = network->header[p].index;
  } else {
    PetscInt                 offsetp;
    DMNetworkComponentHeader header;

    PetscCall(PetscSectionGetOffset(network->DataSection,p,&offsetp));
    header = (DMNetworkComponentHeader)(network->componentdataarray+offsetp);
    *index = header->index;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMNetworkGetSubnetID(DM dm,PetscInt p,PetscInt *subnetid)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (network->header) {
    *subnetid= network->header[p].subnetid;
  } else {
    PetscInt                 offsetp;
    DMNetworkComponentHeader header;

    PetscCall(PetscSectionGetOffset(network->DataSection,p,&offsetp));
    header = (DMNetworkComponentHeader)(network->componentdataarray+offsetp);
    *subnetid = header->subnetid;
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalEdgeIndex - Get the global numbering for the edge on the network

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
- p - edge point

  Output Parameters:
. index - the global numbering for the edge

  Level: intermediate

.seealso: `DMNetworkGetGlobalVertexIndex()`
@*/
PetscErrorCode DMNetworkGetGlobalEdgeIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscFunctionBegin;
  PetscCall(DMNetworkGetIndex(dm,p,index));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalVertexIndex - Get the global numbering for the vertex on the network

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
- p  - vertex point

  Output Parameters:
. index - the global numbering for the vertex

  Level: intermediate

.seealso: `DMNetworkGetGlobalEdgeIndex()`, `DMNetworkGetLocalVertexIndex()`
@*/
PetscErrorCode DMNetworkGetGlobalVertexIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscFunctionBegin;
  PetscCall(DMNetworkGetIndex(dm,p,index));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetNumComponents - Get the number of components at a vertex/edge

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - vertex/edge point

  Output Parameters:
. numcomponents - Number of components at the vertex/edge

  Level: beginner

.seealso: `DMNetworkRegisterComponent()`, `DMNetworkAddComponent()`
@*/
PetscErrorCode DMNetworkGetNumComponents(DM dm,PetscInt p,PetscInt *numcomponents)
{
  PetscInt       offset;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offset));
  *numcomponents = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetLocalVecOffset - Get the offset for accessing the variables associated with a component at the given vertex/edge from the local vector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - the edge or vertex point
- compnum - component number; use ALL_COMPONENTS if no specific component is requested

  Output Parameters:
. offset - the local offset

  Notes:
    These offsets can be passed to MatSetValuesLocal() for matrices obtained with DMCreateMatrix().

    For vectors obtained with DMCreateLocalVector() the offsets can be used with VecSetValues().

    For vectors obtained with DMCreateLocalVector() and the array obtained with VecGetArray(vec,&array) you can access or set
    the vector values with array[offset].

    For vectors obtained with DMCreateGlobalVector() the offsets can be used with VecSetValuesLocal().

  Level: intermediate

.seealso: `DMGetLocalVector()`, `DMNetworkGetComponent()`, `DMNetworkGetGlobalVecOffset()`, `DMCreateGlobalVector()`, `VecGetArray()`, `VecSetValuesLocal()`, `MatSetValuesLocal()`
@*/
PetscErrorCode DMNetworkGetLocalVecOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offset)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->plex->localSection,p,&offsetp));
  if (compnum == ALL_COMPONENTS) {
    *offset = offsetp;
    PetscFunctionReturn(0);
  }

  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offsetd));
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetd);
  *offset = offsetp + header->offsetvarrel[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalVecOffset - Get the global offset for accessing the variables associated with a component for the given vertex/edge from the global vector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - the edge or vertex point
- compnum - component number; use ALL_COMPONENTS if no specific component is requested

  Output Parameters:
. offsetg - the global offset

  Notes:
    These offsets can be passed to MatSetValues() for matrices obtained with DMCreateMatrix().

    For vectors obtained with DMCreateGlobalVector() the offsets can be used with VecSetValues().

    For vectors obtained with DMCreateGlobalVector() and the array obtained with VecGetArray(vec,&array) you can access or set
    the vector values with array[offset - rstart] where restart is obtained with VecGetOwnershipRange(v,&rstart,NULL);

  Level: intermediate

.seealso: `DMNetworkGetLocalVecOffset()`, `DMGetGlobalVector()`, `DMNetworkGetComponent()`, `DMCreateGlobalVector()`, `VecGetArray()`, `VecSetValues()`, `MatSetValues()`
@*/
PetscErrorCode DMNetworkGetGlobalVecOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offsetg)
{
  DM_Network               *network = (DM_Network*)dm->data;
  PetscInt                 offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->plex->globalSection,p,&offsetp));
  if (offsetp < 0) offsetp = -(offsetp + 1); /* Convert to actual global offset for ghost vertex */

  if (compnum == ALL_COMPONENTS) {
    *offsetg = offsetp;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offsetd));
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetd);
  *offsetg = offsetp + header->offsetvarrel[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetEdgeOffset - Get the offset for accessing the variables associated with the given edge from the local subvector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the edge point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: `DMNetworkGetLocalVecOffset()`, `DMGetLocalVector()`
@*/
PetscErrorCode DMNetworkGetEdgeOffset(DM dm,PetscInt p,PetscInt *offset)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->edge.DofSection,p,offset));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetVertexOffset - Get the offset for accessing the variables associated with the given vertex from the local subvector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the vertex point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: `DMNetworkGetEdgeOffset()`, `DMGetLocalVector()`
@*/
PetscErrorCode DMNetworkGetVertexOffset(DM dm,PetscInt p,PetscInt *offset)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  p -= network->cloneshared->vStart;
  PetscCall(PetscSectionGetOffset(network->vertex.DofSection,p,offset));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAddComponent - Adds a network component and number of variables at the given point (vertex/edge)

  Collective on dm

  Input Parameters:
+ dm - the DMNetwork
. p - the vertex/edge point. These points are local indices provided by DMNetworkGetSubnetwork()
. componentkey - component key returned while registering the component with DMNetworkRegisterComponent()
. compvalue - pointer to the data structure for the component, or NULL if the component does not require data, this data is not copied so you cannot
              free this space until after DMSetUp() is called.
- nvar - number of variables for the component at the vertex/edge point, zero if the component does not introduce any degrees of freedom at the point

  Notes:
    The owning rank and any other ranks that have this point as a ghost location must call this routine to add a component and number of variables in the same order at the given point.

    DMNetworkLayoutSetUp() must be called before this routine.

  Developer Notes:
     The requirement that all the ranks with access to a vertex (as owner or as ghost) add all the components comes from a limitation of the underlying implementation based on DMPLEX.
  Level: beginner

.seealso: `DMNetworkGetComponent()`, `DMNetworkGetSubnetwork()`, `DMNetworkIsGhostVertex()`, `DMNetworkLayoutSetUp()`
@*/
PetscErrorCode DMNetworkAddComponent(DM dm,PetscInt p,PetscInt componentkey,void* compvalue,PetscInt nvar)
{
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponent       *component = &network->component[componentkey];
  DMNetworkComponentHeader header;
  DMNetworkComponentValue  cvalue;
  PetscInt                 compnum;
  PetscInt                 *compsize,*compkey,*compoffset,*compnvar,*compoffsetvarrel;
  void*                    *compdata;

  PetscFunctionBegin;
  PetscCheck(componentkey >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"componentkey %" PetscInt_FMT " cannot be negative. Input a component key returned while registering the component with DMNetworkRegisterComponent()",componentkey);
  PetscCheck(network->componentsetup == PETSC_FALSE,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The network has already finalized the components. No new components can be added.");
  /* The owning rank and all ghost ranks add nvar */
  PetscCall(PetscSectionAddDof(network->DofSection,p,nvar));

  /* The owning rank and all ghost ranks add a component, including compvalue=NULL */
  header = &network->header[p];
  cvalue = &network->cvalue[p];
  if (header->ndata == header->maxcomps) {
    PetscInt additional_size;

    /* Reached limit so resize header component arrays */
    header->maxcomps += 2;

    /* Allocate arrays for component information and value */
    PetscCall(PetscCalloc5(header->maxcomps,&compsize,header->maxcomps,&compkey,header->maxcomps,&compoffset,header->maxcomps,&compnvar,header->maxcomps,&compoffsetvarrel));
    PetscCall(PetscMalloc1(header->maxcomps,&compdata));

    /* Recalculate header size */
    header->hsize = sizeof(struct _p_DMNetworkComponentHeader) + 5*header->maxcomps*sizeof(PetscInt);

    header->hsize /= sizeof(DMNetworkComponentGenericDataType);

    /* Copy over component info */
    PetscCall(PetscMemcpy(compsize,header->size,header->ndata*sizeof(PetscInt)));
    PetscCall(PetscMemcpy(compkey,header->key,header->ndata*sizeof(PetscInt)));
    PetscCall(PetscMemcpy(compoffset,header->offset,header->ndata*sizeof(PetscInt)));
    PetscCall(PetscMemcpy(compnvar,header->nvar,header->ndata*sizeof(PetscInt)));
    PetscCall(PetscMemcpy(compoffsetvarrel,header->offsetvarrel,header->ndata*sizeof(PetscInt)));

    /* Copy over component data pointers */
    PetscCall(PetscMemcpy(compdata,cvalue->data,header->ndata*sizeof(void*)));

    /* Free old arrays */
    PetscCall(PetscFree5(header->size,header->key,header->offset,header->nvar,header->offsetvarrel));
    PetscCall(PetscFree(cvalue->data));

    /* Update pointers */
    header->size         = compsize;
    header->key          = compkey;
    header->offset       = compoffset;
    header->nvar         = compnvar;
    header->offsetvarrel = compoffsetvarrel;

    cvalue->data = compdata;

    /* Update DataSection Dofs */
    /* The dofs for datasection point p equals sizeof the header (i.e. header->hsize) + sizes of the components added at point p. With the resizing of the header, we need to update the dofs for point p. Hence, we add the extra size added for the header */
    additional_size = (5*(header->maxcomps - header->ndata)*sizeof(PetscInt))/sizeof(DMNetworkComponentGenericDataType);
    PetscCall(PetscSectionAddDof(network->DataSection,p,additional_size));
  }
  header = &network->header[p];
  cvalue = &network->cvalue[p];

  compnum = header->ndata;

  header->size[compnum] = component->size;
  PetscCall(PetscSectionAddDof(network->DataSection,p,component->size));
  header->key[compnum] = componentkey;
  if (compnum != 0) header->offset[compnum] = header->offset[compnum-1] + header->size[compnum-1];
  cvalue->data[compnum] = (void*)compvalue;

  /* variables */
  header->nvar[compnum] += nvar;
  if (compnum != 0) header->offsetvarrel[compnum] = header->offsetvarrel[compnum-1] + header->nvar[compnum-1];

  header->ndata++;
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetComponent - Gets the component key, the component data, and the number of variables at a given network point

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - vertex/edge point
- compnum - component number; use ALL_COMPONENTS if sum up all the components

  Output Parameters:
+ compkey - the key obtained when registering the component (use NULL if not required)
. component - the component data (use NULL if not required)
- nvar  - number of variables (use NULL if not required)

  Level: beginner

.seealso: `DMNetworkAddComponent()`, `DMNetworkGetNumComponents()`
@*/
PetscErrorCode DMNetworkGetComponent(DM dm,PetscInt p,PetscInt compnum,PetscInt *compkey,void **component,PetscInt *nvar)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offset = 0;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  if (compnum == ALL_COMPONENTS) {
    PetscCall(PetscSectionGetDof(network->DofSection,p,nvar));
    PetscFunctionReturn(0);
  }

  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offset));
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  if (compnum >= 0) {
    if (compkey) *compkey = header->key[compnum];
    if (component) {
      offset += header->hsize+header->offset[compnum];
      *component = network->componentdataarray+offset;
    }
  }

  if (nvar) *nvar = header->nvar[compnum];

  PetscFunctionReturn(0);
}

/*
 Sets up the array that holds the data for all components and its associated section.
 It copies the data for all components in a contiguous array called componentdataarray. The component data is stored pointwise with an additional header (metadata) stored for each point. The header has metadata information such as number of components at each point, number of variables for each component, offsets for the components data, etc.
*/
PetscErrorCode DMNetworkComponentSetUp(DM dm)
{
  DM_Network               *network = (DM_Network*)dm->data;
  PetscInt                 arr_size,p,offset,offsetp,ncomp,i,*headerarr;
  DMNetworkComponentHeader header;
  DMNetworkComponentValue  cvalue;
  DMNetworkComponentHeader headerinfo;
  DMNetworkComponentGenericDataType *componentdataarray;

  PetscFunctionBegin;
  PetscCall(PetscSectionSetUp(network->DataSection));
  PetscCall(PetscSectionGetStorageSize(network->DataSection,&arr_size));
  /* arr_size+1 fixes pipeline test of opensolaris-misc for src/dm/tests/ex10.c -- Do not know why */
  PetscCall(PetscCalloc1(arr_size+1,&network->componentdataarray));
  componentdataarray = network->componentdataarray;
  for (p = network->cloneshared->pStart; p < network->cloneshared->pEnd; p++) {
    PetscCall(PetscSectionGetOffset(network->DataSection,p,&offsetp));
    /* Copy header */
    header = &network->header[p];
    headerinfo = (DMNetworkComponentHeader)(componentdataarray+offsetp);
    PetscCall(PetscMemcpy(headerinfo,header,sizeof(struct _p_DMNetworkComponentHeader)));
    headerarr = (PetscInt*)(headerinfo+1);
    PetscCall(PetscMemcpy(headerarr,header->size,header->maxcomps*sizeof(PetscInt)));
    headerinfo->size = headerarr;
    headerarr += header->maxcomps;
    PetscCall(PetscMemcpy(headerarr,header->key,header->maxcomps*sizeof(PetscInt)));
    headerinfo->key = headerarr;
    headerarr += header->maxcomps;
    PetscCall(PetscMemcpy(headerarr,header->offset,header->maxcomps*sizeof(PetscInt)));
    headerinfo->offset = headerarr;
    headerarr += header->maxcomps;
    PetscCall(PetscMemcpy(headerarr,header->nvar,header->maxcomps*sizeof(PetscInt)));
    headerinfo->nvar = headerarr;
    headerarr += header->maxcomps;
    PetscCall(PetscMemcpy(headerarr,header->offsetvarrel,header->maxcomps*sizeof(PetscInt)));
    headerinfo->offsetvarrel = headerarr;

    /* Copy data */
    cvalue = &network->cvalue[p];
    ncomp  = header->ndata;

    for (i = 0; i < ncomp; i++) {
      offset = offsetp + header->hsize + header->offset[i];
      PetscCall(PetscMemcpy(componentdataarray+offset,cvalue->data[i],header->size[i]*sizeof(DMNetworkComponentGenericDataType)));
    }
  }

  for (i=network->cloneshared->pStart; i < network->cloneshared->pEnd; i++) {
    PetscCall(PetscFree5(network->header[i].size,network->header[i].key,network->header[i].offset,network->header[i].nvar,network->header[i].offsetvarrel));
    PetscCall(PetscFree(network->cvalue[i].data));
  }
  PetscCall(PetscFree2(network->header,network->cvalue));
  PetscFunctionReturn(0);
}

/* Sets up the section for dofs. This routine is called during DMSetUp() */
static PetscErrorCode DMNetworkVariablesSetUp(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscSectionSetUp(network->DofSection));
  PetscFunctionReturn(0);
}

/* Get a subsection from a range of points */
static PetscErrorCode DMNetworkGetSubSection_private(PetscSection main,PetscInt pstart,PetscInt pend,PetscSection *subsection)
{
  PetscInt       i, nvar;

  PetscFunctionBegin;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)main), subsection));
  PetscCall(PetscSectionSetChart(*subsection, 0, pend - pstart));
  for (i = pstart; i < pend; i++) {
    PetscCall(PetscSectionGetDof(main,i,&nvar));
    PetscCall(PetscSectionSetDof(*subsection, i - pstart, nvar));
  }

  PetscCall(PetscSectionSetUp(*subsection));
  PetscFunctionReturn(0);
}

/* Create a submap of points with a GlobalToLocal structure */
static PetscErrorCode DMNetworkSetSubMap_private(PetscInt pstart, PetscInt pend, ISLocalToGlobalMapping *map)
{
  PetscInt       i, *subpoints;

  PetscFunctionBegin;
  /* Create index sets to map from "points" to "subpoints" */
  PetscCall(PetscMalloc1(pend - pstart, &subpoints));
  for (i = pstart; i < pend; i++) {
    subpoints[i - pstart] = i;
  }
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,pend-pstart,subpoints,PETSC_COPY_VALUES,map));
  PetscCall(PetscFree(subpoints));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkAssembleGraphStructures - Assembles vertex and edge data structures. Must be called after DMNetworkDistribute

  Collective on dm

  Input Parameters:
. dm - the DMNetworkObject

  Note: the routine will create alternative orderings for the vertices and edges. Assume global network points are:

  points = [0 1 2 3 4 5 6]

  where edges = [0,1,2,3] and vertices = [4,5,6]. The new orderings will be specific to the subset (i.e vertices = [0,1,2] <- [4,5,6]).

  With this new ordering a local PetscSection, global PetscSection and PetscSF will be created specific to the subset.

  Level: intermediate

@*/
PetscErrorCode DMNetworkAssembleGraphStructures(DM dm)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* Create maps for vertices and edges */
  PetscCall(DMNetworkSetSubMap_private(network->cloneshared->vStart,network->cloneshared->vEnd,&network->vertex.mapping));
  PetscCall(DMNetworkSetSubMap_private(network->cloneshared->eStart,network->cloneshared->eEnd,&network->edge.mapping));

  /* Create local sub-sections */
  PetscCall(DMNetworkGetSubSection_private(network->DofSection,network->cloneshared->vStart,network->cloneshared->vEnd,&network->vertex.DofSection));
  PetscCall(DMNetworkGetSubSection_private(network->DofSection,network->cloneshared->eStart,network->cloneshared->eEnd,&network->edge.DofSection));

  if (size > 1) {
    PetscCall(PetscSFGetSubSF(network->plex->sf, network->vertex.mapping, &network->vertex.sf));

    PetscCall(PetscSectionCreateGlobalSection(network->vertex.DofSection, network->vertex.sf, PETSC_FALSE, PETSC_FALSE, &network->vertex.GlobalDofSection));
    PetscCall(PetscSFGetSubSF(network->plex->sf, network->edge.mapping, &network->edge.sf));
    PetscCall(PetscSectionCreateGlobalSection(network->edge.DofSection, network->edge.sf, PETSC_FALSE, PETSC_FALSE, &network->edge.GlobalDofSection));
  } else {
    /* create structures for vertex */
    PetscCall(PetscSectionClone(network->vertex.DofSection,&network->vertex.GlobalDofSection));
    /* create structures for edge */
    PetscCall(PetscSectionClone(network->edge.DofSection,&network->edge.GlobalDofSection));
  }

  /* Add viewers */
  PetscCall(PetscObjectSetName((PetscObject)network->edge.GlobalDofSection,"Global edge dof section"));
  PetscCall(PetscObjectSetName((PetscObject)network->vertex.GlobalDofSection,"Global vertex dof section"));
  PetscCall(PetscSectionViewFromOptions(network->edge.GlobalDofSection, NULL, "-edge_global_section_view"));
  PetscCall(PetscSectionViewFromOptions(network->vertex.GlobalDofSection, NULL, "-vertex_global_section_view"));
  PetscFunctionReturn(0);
}

/*
   Setup a lookup btable for the input v's owning subnetworks
   - add all owing subnetworks that connect to this v to the btable
     vertex_subnetid = supportingedge_subnetid
*/
static inline PetscErrorCode SetSubnetIdLookupBT(DM dm,PetscInt v,PetscInt Nsubnet,PetscBT btable)
{
  PetscInt       e,nedges,offset;
  const PetscInt *edges;
  DM_Network     *newDMnetwork = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscBTMemzero(Nsubnet,btable));
  PetscCall(DMNetworkGetSupportingEdges(dm,v,&nedges,&edges));
  for (e=0; e<nedges; e++) {
    PetscCall(PetscSectionGetOffset(newDMnetwork->DataSection,edges[e],&offset));
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);
    PetscCall(PetscBTSet(btable,header->subnetid));
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkDistribute - Distributes the network and moves associated component data

  Collective

  Input Parameters:
+ DM - the DMNetwork object
- overlap - the overlap of partitions, 0 is the default

  Options Database Key:
+ -dmnetwork_view - Calls DMView() at the conclusion of DMSetUp()
- -dmnetwork_view_distributed - Calls DMView() at the conclusion of DMNetworkDistribute()

  Notes:
  Distributes the network with <overlap>-overlapping partitioning of the edges.

  Level: intermediate

.seealso: `DMNetworkCreate()`
@*/
PetscErrorCode DMNetworkDistribute(DM *dm,PetscInt overlap)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  DM_Network     *oldDMnetwork = (DM_Network*)((*dm)->data);
  DM_Network     *newDMnetwork;
  PetscSF        pointsf=NULL;
  DM             newDM;
  PetscInt       j,e,v,offset,*subnetvtx,*subnetedge,Nsubnet,gidx,svtx_idx,nv;
  PetscInt       net,*sv;
  PetscBT        btable;
  PetscPartitioner         part;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscValidPointer(dm,1);
  PetscValidHeaderSpecific(*dm,DM_CLASSID,1);
  PetscCall(PetscObjectGetComm((PetscObject)*dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) PetscFunctionReturn(0);

  PetscCheck(!overlap,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"overlap %" PetscInt_FMT " != 0 is not supported yet",overlap);

  /* This routine moves the component data to the appropriate processors. It makes use of the DataSection and the componentdataarray to move the component data to appropriate processors and returns a new DataSection and new componentdataarray. */
  PetscCall(PetscLogEventBegin(DMNetwork_Distribute,dm,0,0,0));
  PetscCall(DMNetworkCreate(PetscObjectComm((PetscObject)*dm),&newDM));
  newDMnetwork = (DM_Network*)newDM->data;
  newDMnetwork->max_comps_registered = oldDMnetwork->max_comps_registered;
  PetscCall(PetscMalloc1(newDMnetwork->max_comps_registered,&newDMnetwork->component));

  /* Enable runtime options for petscpartitioner */
  PetscCall(DMPlexGetPartitioner(oldDMnetwork->plex,&part));
  PetscCall(PetscPartitionerSetFromOptions(part));

  /* Distribute plex dm */
  PetscCall(DMPlexDistribute(oldDMnetwork->plex,overlap,&pointsf,&newDMnetwork->plex));

  /* Distribute dof section */
  PetscCall(PetscSectionCreate(comm,&newDMnetwork->DofSection));
  PetscCall(PetscSFDistributeSection(pointsf,oldDMnetwork->DofSection,NULL,newDMnetwork->DofSection));

  /* Distribute data and associated section */
  PetscCall(PetscSectionCreate(comm,&newDMnetwork->DataSection));
  PetscCall(DMPlexDistributeData(newDMnetwork->plex,pointsf,oldDMnetwork->DataSection,MPIU_INT,(void*)oldDMnetwork->componentdataarray,newDMnetwork->DataSection,(void**)&newDMnetwork->componentdataarray));

  PetscCall(PetscSectionGetChart(newDMnetwork->DataSection,&newDMnetwork->cloneshared->pStart,&newDMnetwork->cloneshared->pEnd));
  PetscCall(DMPlexGetHeightStratum(newDMnetwork->plex,0, &newDMnetwork->cloneshared->eStart,&newDMnetwork->cloneshared->eEnd));
  PetscCall(DMPlexGetHeightStratum(newDMnetwork->plex,1,&newDMnetwork->cloneshared->vStart,&newDMnetwork->cloneshared->vEnd));
  newDMnetwork->cloneshared->nEdges    = newDMnetwork->cloneshared->eEnd - newDMnetwork->cloneshared->eStart;
  newDMnetwork->cloneshared->nVertices = newDMnetwork->cloneshared->vEnd - newDMnetwork->cloneshared->vStart;
  newDMnetwork->cloneshared->NVertices = oldDMnetwork->cloneshared->NVertices;
  newDMnetwork->cloneshared->NEdges    = oldDMnetwork->cloneshared->NEdges;
  newDMnetwork->cloneshared->svtable   = oldDMnetwork->cloneshared->svtable; /* global table! */
  oldDMnetwork->cloneshared->svtable   = NULL;

  /* Set Dof section as the section for dm */
  PetscCall(DMSetLocalSection(newDMnetwork->plex,newDMnetwork->DofSection));
  PetscCall(DMGetGlobalSection(newDMnetwork->plex,&newDMnetwork->GlobalDofSection));

  /* Setup subnetwork info in the newDM */
  newDMnetwork->cloneshared->Nsubnet = oldDMnetwork->cloneshared->Nsubnet;
  newDMnetwork->cloneshared->Nsvtx   = oldDMnetwork->cloneshared->Nsvtx;
  oldDMnetwork->cloneshared->Nsvtx   = 0;
  newDMnetwork->cloneshared->svtx    = oldDMnetwork->cloneshared->svtx; /* global vertices! */
  oldDMnetwork->cloneshared->svtx    = NULL;
  PetscCall(PetscCalloc1(newDMnetwork->cloneshared->Nsubnet,&newDMnetwork->cloneshared->subnet));

  /* Copy over the global number of vertices and edges in each subnetwork.
     Note: these are calculated in DMNetworkLayoutSetUp()
  */
  Nsubnet = newDMnetwork->cloneshared->Nsubnet;
  for (j = 0; j < Nsubnet; j++) {
    newDMnetwork->cloneshared->subnet[j].Nvtx  = oldDMnetwork->cloneshared->subnet[j].Nvtx;
    newDMnetwork->cloneshared->subnet[j].Nedge = oldDMnetwork->cloneshared->subnet[j].Nedge;
  }

  /* Count local nedges for subnetworks */
  for (e = newDMnetwork->cloneshared->eStart; e < newDMnetwork->cloneshared->eEnd; e++) {
    PetscCall(PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset));
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);

    /* Update pointers */
    header->size          = (PetscInt*)(header + 1);
    header->key           = header->size   + header->maxcomps;
    header->offset        = header->key    + header->maxcomps;
    header->nvar          = header->offset + header->maxcomps;
    header->offsetvarrel  = header->nvar   + header->maxcomps;

    newDMnetwork->cloneshared->subnet[header->subnetid].nedge++;
  }

  /* Setup a btable to keep track subnetworks owned by this process at a shared vertex */
  if (newDMnetwork->cloneshared->Nsvtx) {
    PetscCall(PetscBTCreate(Nsubnet,&btable));
  }

  /* Count local nvtx for subnetworks */
  for (v = newDMnetwork->cloneshared->vStart; v < newDMnetwork->cloneshared->vEnd; v++) {
    PetscCall(PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset));
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);

    /* Update pointers */
    header->size          = (PetscInt*)(header + 1);
    header->key           = header->size   + header->maxcomps;
    header->offset        = header->key    + header->maxcomps;
    header->nvar          = header->offset + header->maxcomps;
    header->offsetvarrel  = header->nvar   + header->maxcomps;

    /* shared vertices: use gidx=header->index to check if v is a shared vertex */
    gidx = header->index;
    PetscCall(PetscTableFind(newDMnetwork->cloneshared->svtable,gidx+1,&svtx_idx));
    svtx_idx--;

    if (svtx_idx < 0) { /* not a shared vertex */
      newDMnetwork->cloneshared->subnet[header->subnetid].nvtx++;
    } else { /* a shared vertex belongs to more than one subnetworks, it is being counted by multiple subnets */
      /* Setup a lookup btable for this v's owning subnetworks */
      PetscCall(SetSubnetIdLookupBT(newDM,v,Nsubnet,btable));

      for (j=0; j<newDMnetwork->cloneshared->svtx[svtx_idx].n; j++) {
        sv  = newDMnetwork->cloneshared->svtx[svtx_idx].sv + 2*j;
        net = sv[0];
        if (PetscBTLookup(btable,net)) newDMnetwork->cloneshared->subnet[net].nvtx++; /* sv is on net owned by this proces */
      }
    }
  }

  /* Get total local nvtx for subnetworks */
  nv = 0;
  for (j=0; j<Nsubnet; j++) nv += newDMnetwork->cloneshared->subnet[j].nvtx;
  nv += newDMnetwork->cloneshared->Nsvtx;

  /* Now create the vertices and edge arrays for the subnetworks */
  PetscCall(PetscCalloc2(newDMnetwork->cloneshared->nEdges,&subnetedge,nv,&subnetvtx)); /* Maps local vertex to local subnetwork's vertex */
  newDMnetwork->cloneshared->subnetedge = subnetedge;
  newDMnetwork->cloneshared->subnetvtx  = subnetvtx;
  for (j=0; j < newDMnetwork->cloneshared->Nsubnet; j++) {
    newDMnetwork->cloneshared->subnet[j].edges = subnetedge;
    subnetedge                   += newDMnetwork->cloneshared->subnet[j].nedge;

    newDMnetwork->cloneshared->subnet[j].vertices = subnetvtx;
    subnetvtx                       += newDMnetwork->cloneshared->subnet[j].nvtx;

    /* Temporarily setting nvtx and nedge to 0 so we can use them as counters in the below for loop. These get updated when the vertices and edges are added. */
    newDMnetwork->cloneshared->subnet[j].nvtx = newDMnetwork->cloneshared->subnet[j].nedge = 0;
  }
  newDMnetwork->cloneshared->svertices = subnetvtx;

  /* Set the edges and vertices in each subnetwork */
  for (e = newDMnetwork->cloneshared->eStart; e < newDMnetwork->cloneshared->eEnd; e++) {
    PetscCall(PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset));
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);
    newDMnetwork->cloneshared->subnet[header->subnetid].edges[newDMnetwork->cloneshared->subnet[header->subnetid].nedge++] = e;
  }

  nv = 0;
  for (v = newDMnetwork->cloneshared->vStart; v < newDMnetwork->cloneshared->vEnd; v++) {
    PetscCall(PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset));
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);

    /* coupling vertices: use gidx = header->index to check if v is a coupling vertex */
    PetscCall(PetscTableFind(newDMnetwork->cloneshared->svtable,header->index+1,&svtx_idx));
    svtx_idx--;
    if (svtx_idx < 0) {
      newDMnetwork->cloneshared->subnet[header->subnetid].vertices[newDMnetwork->cloneshared->subnet[header->subnetid].nvtx++] = v;
    } else { /* a shared vertex */
      newDMnetwork->cloneshared->svertices[nv++] = v;

      /* Setup a lookup btable for this v's owning subnetworks */
      PetscCall(SetSubnetIdLookupBT(newDM,v,Nsubnet,btable));

      for (j=0; j<newDMnetwork->cloneshared->svtx[svtx_idx].n; j++) {
        sv  = newDMnetwork->cloneshared->svtx[svtx_idx].sv + 2*j;
        net = sv[0];
        if (PetscBTLookup(btable,net))
          newDMnetwork->cloneshared->subnet[net].vertices[newDMnetwork->cloneshared->subnet[net].nvtx++] = v;
      }
    }
  }
  newDMnetwork->cloneshared->nsvtx = nv;   /* num of local shared vertices */

  newDM->setupcalled = (*dm)->setupcalled;
  newDMnetwork->cloneshared->distributecalled = PETSC_TRUE;

  /* Free spaces */
  PetscCall(PetscSFDestroy(&pointsf));
  PetscCall(DMDestroy(dm));
  if (newDMnetwork->cloneshared->Nsvtx) {
    PetscCall(PetscBTDestroy(&btable));
  }

  /* View distributed dmnetwork */
  PetscCall(DMViewFromOptions(newDM,NULL,"-dmnetwork_view_distributed"));

  *dm  = newDM;
  PetscCall(PetscLogEventEnd(DMNetwork_Distribute,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
  PetscSFGetSubSF - Returns an SF for a specific subset of points. Leaves are re-numbered to reflect the new ordering

 Collective

  Input Parameters:
+ mainSF - the original SF structure
- map - a ISLocalToGlobal mapping that contains the subset of points

  Output Parameter:
. subSF - a subset of the mainSF for the desired subset.

  Level: intermediate
@*/
PetscErrorCode PetscSFGetSubSF(PetscSF mainsf,ISLocalToGlobalMapping map,PetscSF *subSF)
{
  PetscInt              nroots, nleaves, *ilocal_sub;
  PetscInt              i, *ilocal_map, nroots_sub, nleaves_sub = 0;
  PetscInt              *local_points, *remote_points;
  PetscSFNode           *iremote_sub;
  const PetscInt        *ilocal;
  const PetscSFNode     *iremote;

  PetscFunctionBegin;
  PetscCall(PetscSFGetGraph(mainsf,&nroots,&nleaves,&ilocal,&iremote));

  /* Look for leaves that pertain to the subset of points. Get the local ordering */
  PetscCall(PetscMalloc1(nleaves,&ilocal_map));
  PetscCall(ISGlobalToLocalMappingApply(map,IS_GTOLM_MASK,nleaves,ilocal,NULL,ilocal_map));
  for (i = 0; i < nleaves; i++) {
    if (ilocal_map[i] != -1) nleaves_sub += 1;
  }
  /* Re-number ilocal with subset numbering. Need information from roots */
  PetscCall(PetscMalloc2(nroots,&local_points,nroots,&remote_points));
  for (i = 0; i < nroots; i++) local_points[i] = i;
  PetscCall(ISGlobalToLocalMappingApply(map,IS_GTOLM_MASK,nroots,local_points,NULL,local_points));
  PetscCall(PetscSFBcastBegin(mainsf, MPIU_INT, local_points, remote_points,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(mainsf, MPIU_INT, local_points, remote_points,MPI_REPLACE));
  /* Fill up graph using local (that is, local to the subset) numbering. */
  PetscCall(PetscMalloc1(nleaves_sub,&ilocal_sub));
  PetscCall(PetscMalloc1(nleaves_sub,&iremote_sub));
  nleaves_sub = 0;
  for (i = 0; i < nleaves; i++) {
    if (ilocal_map[i] != -1) {
      ilocal_sub[nleaves_sub] = ilocal_map[i];
      iremote_sub[nleaves_sub].rank = iremote[i].rank;
      iremote_sub[nleaves_sub].index = remote_points[ilocal[i]];
      nleaves_sub += 1;
    }
  }
  PetscCall(PetscFree2(local_points,remote_points));
  PetscCall(ISLocalToGlobalMappingGetSize(map,&nroots_sub));

  /* Create new subSF */
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,subSF));
  PetscCall(PetscSFSetFromOptions(*subSF));
  PetscCall(PetscSFSetGraph(*subSF,nroots_sub,nleaves_sub,ilocal_sub,PETSC_OWN_POINTER,iremote_sub,PETSC_COPY_VALUES));
  PetscCall(PetscFree(ilocal_map));
  PetscCall(PetscFree(iremote_sub));
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetSupportingEdges - Return the supporting edges for this vertex point

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p  - the vertex point

  Output Parameters:
+ nedges - number of edges connected to this vertex point
- edges  - list of edge points

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: `DMNetworkCreate()`, `DMNetworkGetConnectedVertices()`
@*/
PetscErrorCode DMNetworkGetSupportingEdges(DM dm,PetscInt vertex,PetscInt *nedges,const PetscInt *edges[])
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMPlexGetSupportSize(network->plex,vertex,nedges));
  if (edges) PetscCall(DMPlexGetSupport(network->plex,vertex,edges));
  PetscFunctionReturn(0);
}

/*@C
  DMNetworkGetConnectedVertices - Return the connected vertices for this edge point

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the edge point

  Output Parameters:
. vertices - vertices connected to this edge

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: `DMNetworkCreate()`, `DMNetworkGetSupportingEdges()`
@*/
PetscErrorCode DMNetworkGetConnectedVertices(DM dm,PetscInt edge,const PetscInt *vertices[])
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCone(network->plex,edge,vertices));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkIsSharedVertex - Returns TRUE if the vertex is shared by subnetworks

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the vertex point

  Output Parameter:
. flag - TRUE if the vertex is shared by subnetworks

  Level: beginner

.seealso: `DMNetworkAddSharedVertices()`, `DMNetworkIsGhostVertex()`
@*/
PetscErrorCode DMNetworkIsSharedVertex(DM dm,PetscInt p,PetscBool *flag)
{
  PetscInt       i;

  PetscFunctionBegin;
  *flag = PETSC_FALSE;

  if (dm->setupcalled) { /* DMNetworkGetGlobalVertexIndex() requires DMSetUp() be called */
    DM_Network     *network = (DM_Network*)dm->data;
    PetscInt       gidx;
    PetscCall(DMNetworkGetGlobalVertexIndex(dm,p,&gidx));
    PetscCall(PetscTableFind(network->cloneshared->svtable,gidx+1,&i));
    if (i) *flag = PETSC_TRUE;
  } else { /* would be removed? */
    PetscInt       nv;
    const PetscInt *vtx;
    PetscCall(DMNetworkGetSharedVertices(dm,&nv,&vtx));
    for (i=0; i<nv; i++) {
      if (p == vtx[i]) {
        *flag = PETSC_TRUE;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkIsGhostVertex - Returns TRUE if the vertex is a ghost vertex

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the vertex point

  Output Parameter:
. isghost - TRUE if the vertex is a ghost point

  Level: beginner

.seealso: `DMNetworkGetConnectedVertices()`, `DMNetworkGetVertexRange()`, `DMNetworkIsSharedVertex()`
@*/
PetscErrorCode DMNetworkIsGhostVertex(DM dm,PetscInt p,PetscBool *isghost)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
  *isghost = PETSC_FALSE;
  PetscCall(DMGetGlobalSection(network->plex,&sectiong));
  PetscCall(PetscSectionGetOffset(sectiong,p,&offsetg));
  if (offsetg < 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUp_Network(DM dm)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMNetwork_SetUpNetwork,dm,0,0,0));
  PetscCall(DMNetworkFinalizeComponents(dm));
  /* View dmnetwork */
  PetscCall(DMViewFromOptions(dm,NULL,"-dmnetwork_view"));
  PetscCall(PetscLogEventEnd(DMNetwork_SetUpNetwork,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMNetworkHasJacobian - Sets global flag for using user's sub Jacobian matrices
      -- replaced by DMNetworkSetOption(network,userjacobian,PETSC_TURE)?

  Collective

  Input Parameters:
+ dm - the DMNetwork object
. eflg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE) if user provides Jacobian for edges
- vflg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE) if user provides Jacobian for vertices

 Level: intermediate

@*/
PetscErrorCode DMNetworkHasJacobian(DM dm,PetscBool eflg,PetscBool vflg)
{
  DM_Network     *network=(DM_Network*)dm->data;
  PetscInt       nVertices = network->cloneshared->nVertices;

  PetscFunctionBegin;
  network->userEdgeJacobian   = eflg;
  network->userVertexJacobian = vflg;

  if (eflg && !network->Je) {
    PetscCall(PetscCalloc1(3*network->cloneshared->nEdges,&network->Je));
  }

  if (vflg && !network->Jv && nVertices) {
    PetscInt       i,*vptr,nedges,vStart=network->cloneshared->vStart;
    PetscInt       nedges_total;
    const PetscInt *edges;

    /* count nvertex_total */
    nedges_total = 0;
    PetscCall(PetscMalloc1(nVertices+1,&vptr));

    vptr[0] = 0;
    for (i=0; i<nVertices; i++) {
      PetscCall(DMNetworkGetSupportingEdges(dm,i+vStart,&nedges,&edges));
      nedges_total += nedges;
      vptr[i+1] = vptr[i] + 2*nedges + 1;
    }

    PetscCall(PetscCalloc1(2*nedges_total+nVertices,&network->Jv));
    network->Jvptr = vptr;
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkEdgeSetMatrix - Sets user-provided Jacobian matrices for this edge to the network

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - the edge point
- J - array (size = 3) of Jacobian submatrices for this edge point:
        J[0]: this edge
        J[1] and J[2]: connected vertices, obtained by calling DMNetworkGetConnectedVertices()

  Level: advanced

.seealso: `DMNetworkVertexSetMatrix()`
@*/
PetscErrorCode DMNetworkEdgeSetMatrix(DM dm,PetscInt p,Mat J[])
{
  DM_Network *network=(DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCheck(network->Je,PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"Must call DMNetworkHasJacobian() collectively before calling DMNetworkEdgeSetMatrix");

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
+ dm - The DMNetwork object
. p - the vertex point
- J - array of Jacobian (size = 2*(num of supporting edges) + 1) submatrices for this vertex point:
        J[0]:       this vertex
        J[1+2*i]:   i-th supporting edge
        J[1+2*i+1]: i-th connected vertex

  Level: advanced

.seealso: `DMNetworkEdgeSetMatrix()`
@*/
PetscErrorCode DMNetworkVertexSetMatrix(DM dm,PetscInt p,Mat J[])
{
  DM_Network     *network=(DM_Network*)dm->data;
  PetscInt       i,*vptr,nedges,vStart=network->cloneshared->vStart;
  const PetscInt *edges;

  PetscFunctionBegin;
  PetscCheck(network->Jv,PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"Must call DMNetworkHasJacobian() collectively before calling DMNetworkVertexSetMatrix");

  if (J) {
    vptr = network->Jvptr;
    network->Jv[vptr[p-vStart]] = J[0]; /* Set Jacobian for this vertex */

    /* Set Jacobian for each supporting edge and connected vertex */
    PetscCall(DMNetworkGetSupportingEdges(dm,p,&nedges,&edges));
    for (i=1; i<=2*nedges; i++) network->Jv[vptr[p-vStart]+i] = J[i];
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetPreallocationDenseblock_private(PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscInt       j;
  PetscScalar    val=(PetscScalar)ncols;

  PetscFunctionBegin;
  if (!ghost) {
    for (j=0; j<nrows; j++) {
      PetscCall(VecSetValues(vdnz,1,&rows[j],&val,ADD_VALUES));
    }
  } else {
    for (j=0; j<nrows; j++) {
      PetscCall(VecSetValues(vonz,1,&rows[j],&val,ADD_VALUES));
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetPreallocationUserblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscInt       j,ncols_u;
  PetscScalar    val;

  PetscFunctionBegin;
  if (!ghost) {
    for (j=0; j<nrows; j++) {
      PetscCall(MatGetRow(Ju,j,&ncols_u,NULL,NULL));
      val = (PetscScalar)ncols_u;
      PetscCall(VecSetValues(vdnz,1,&rows[j],&val,ADD_VALUES));
      PetscCall(MatRestoreRow(Ju,j,&ncols_u,NULL,NULL));
    }
  } else {
    for (j=0; j<nrows; j++) {
      PetscCall(MatGetRow(Ju,j,&ncols_u,NULL,NULL));
      val = (PetscScalar)ncols_u;
      PetscCall(VecSetValues(vonz,1,&rows[j],&val,ADD_VALUES));
      PetscCall(MatRestoreRow(Ju,j,&ncols_u,NULL,NULL));
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetPreallocationblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscBool ghost,Vec vdnz,Vec vonz)
{
  PetscFunctionBegin;
  if (Ju) {
    PetscCall(MatSetPreallocationUserblock_private(Ju,nrows,rows,ncols,ghost,vdnz,vonz));
  } else {
    PetscCall(MatSetPreallocationDenseblock_private(nrows,rows,ncols,ghost,vdnz,vonz));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetDenseblock_private(PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscInt       j,*cols;
  PetscScalar    *zeros;

  PetscFunctionBegin;
  PetscCall(PetscCalloc2(ncols,&cols,nrows*ncols,&zeros));
  for (j=0; j<ncols; j++) cols[j] = j+ cstart;
  PetscCall(MatSetValues(*J,nrows,rows,ncols,cols,zeros,INSERT_VALUES));
  PetscCall(PetscFree2(cols,zeros));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetUserblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscInt       j,M,N,row,col,ncols_u;
  const PetscInt *cols;
  PetscScalar    zero=0.0;

  PetscFunctionBegin;
  PetscCall(MatGetSize(Ju,&M,&N));
  PetscCheck(nrows == M && ncols == N,PetscObjectComm((PetscObject)Ju),PETSC_ERR_USER,"%" PetscInt_FMT " by %" PetscInt_FMT " must equal %" PetscInt_FMT " by %" PetscInt_FMT,nrows,ncols,M,N);

  for (row=0; row<nrows; row++) {
    PetscCall(MatGetRow(Ju,row,&ncols_u,&cols,NULL));
    for (j=0; j<ncols_u; j++) {
      col = cols[j] + cstart;
      PetscCall(MatSetValues(*J,1,&rows[row],1,&col,&zero,INSERT_VALUES));
    }
    PetscCall(MatRestoreRow(Ju,row,&ncols_u,&cols,NULL));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetblock_private(Mat Ju,PetscInt nrows,PetscInt *rows,PetscInt ncols,PetscInt cstart,Mat *J)
{
  PetscFunctionBegin;
  if (Ju) {
    PetscCall(MatSetUserblock_private(Ju,nrows,rows,ncols,cstart,J));
  } else {
    PetscCall(MatSetDenseblock_private(nrows,rows,ncols,cstart,J));
  }
  PetscFunctionReturn(0);
}

/* Creates a GlobalToLocal mapping with a Local and Global section. This is akin to the routine DMGetLocalToGlobalMapping but without the need of providing a dm.
*/
PetscErrorCode CreateSubGlobalToLocalMapping_private(PetscSection globalsec, PetscSection localsec, ISLocalToGlobalMapping *ltog)
{
  PetscInt       i,size,dof;
  PetscInt       *glob2loc;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetStorageSize(localsec,&size));
  PetscCall(PetscMalloc1(size,&glob2loc));

  for (i = 0; i < size; i++) {
    PetscCall(PetscSectionGetOffset(globalsec,i,&dof));
    dof = (dof >= 0) ? dof : -(dof + 1);
    glob2loc[i] = dof;
  }

  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD,1,size,glob2loc,PETSC_OWN_POINTER,ltog));
#if 0
  PetscCall(PetscIntView(size,glob2loc,PETSC_VIEWER_STDOUT_WORLD));
#endif
  PetscFunctionReturn(0);
}

#include <petsc/private/matimpl.h>

PetscErrorCode DMCreateMatrix_Network_Nest(DM dm,Mat *J)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       eDof,vDof;
  Mat            j11,j12,j21,j22,bA[2][2];
  MPI_Comm       comm;
  ISLocalToGlobalMapping eISMap,vISMap;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));

  PetscCall(PetscSectionGetConstrainedStorageSize(network->edge.GlobalDofSection,&eDof));
  PetscCall(PetscSectionGetConstrainedStorageSize(network->vertex.GlobalDofSection,&vDof));

  PetscCall(MatCreate(comm, &j11));
  PetscCall(MatSetSizes(j11, eDof, eDof, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(j11, MATMPIAIJ));

  PetscCall(MatCreate(comm, &j12));
  PetscCall(MatSetSizes(j12, eDof, vDof, PETSC_DETERMINE ,PETSC_DETERMINE));
  PetscCall(MatSetType(j12, MATMPIAIJ));

  PetscCall(MatCreate(comm, &j21));
  PetscCall(MatSetSizes(j21, vDof, eDof, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(j21, MATMPIAIJ));

  PetscCall(MatCreate(comm, &j22));
  PetscCall(MatSetSizes(j22, vDof, vDof, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(j22, MATMPIAIJ));

  bA[0][0] = j11;
  bA[0][1] = j12;
  bA[1][0] = j21;
  bA[1][1] = j22;

  PetscCall(CreateSubGlobalToLocalMapping_private(network->edge.GlobalDofSection,network->edge.DofSection,&eISMap));
  PetscCall(CreateSubGlobalToLocalMapping_private(network->vertex.GlobalDofSection,network->vertex.DofSection,&vISMap));

  PetscCall(MatSetLocalToGlobalMapping(j11,eISMap,eISMap));
  PetscCall(MatSetLocalToGlobalMapping(j12,eISMap,vISMap));
  PetscCall(MatSetLocalToGlobalMapping(j21,vISMap,eISMap));
  PetscCall(MatSetLocalToGlobalMapping(j22,vISMap,vISMap));

  PetscCall(MatSetUp(j11));
  PetscCall(MatSetUp(j12));
  PetscCall(MatSetUp(j21));
  PetscCall(MatSetUp(j22));

  PetscCall(MatCreateNest(comm,2,NULL,2,NULL,&bA[0][0],J));
  PetscCall(MatSetUp(*J));
  PetscCall(MatNestSetVecType(*J,VECNEST));
  PetscCall(MatDestroy(&j11));
  PetscCall(MatDestroy(&j12));
  PetscCall(MatDestroy(&j21));
  PetscCall(MatDestroy(&j22));

  PetscCall(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  /* Free structures */
  PetscCall(ISLocalToGlobalMappingDestroy(&eISMap));
  PetscCall(ISLocalToGlobalMappingDestroy(&vISMap));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Network(DM dm,Mat *J)
{
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
  PetscCall(PetscStrcmp(mtype,MATNEST,&isNest));
  if (isNest) {
    PetscCall(DMCreateMatrix_Network_Nest(dm,J));
    PetscCall(MatSetDM(*J,dm));
    PetscFunctionReturn(0);
  }

  if (!network->userEdgeJacobian && !network->userVertexJacobian) {
    /* user does not provide Jacobian blocks */
    PetscCall(DMCreateMatrix_Plex(network->plex,J));
    PetscCall(MatSetDM(*J,dm));
    PetscFunctionReturn(0);
  }

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm),J));
  PetscCall(DMGetGlobalSection(network->plex,&sectionGlobal));
  PetscCall(PetscSectionGetConstrainedStorageSize(sectionGlobal,&localSize));
  PetscCall(MatSetSizes(*J,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE));

  PetscCall(MatSetType(*J,MATAIJ));
  PetscCall(MatSetFromOptions(*J));

  /* (1) Set matrix preallocation */
  /*------------------------------*/
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(VecCreate(comm,&vd_nz));
  PetscCall(VecSetSizes(vd_nz,localSize,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(vd_nz));
  PetscCall(VecSet(vd_nz,0.0));
  PetscCall(VecDuplicate(vd_nz,&vo_nz));

  /* Set preallocation for edges */
  /*-----------------------------*/
  PetscCall(DMNetworkGetEdgeRange(dm,&eStart,&eEnd));

  PetscCall(PetscMalloc1(localSize,&rows));
  for (e=eStart; e<eEnd; e++) {
    /* Get row indices */
    PetscCall(DMNetworkGetGlobalVecOffset(dm,e,ALL_COMPONENTS,&rstart));
    PetscCall(PetscSectionGetDof(network->DofSection,e,&nrows));
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set preallocation for connected vertices */
      PetscCall(DMNetworkGetConnectedVertices(dm,e,&cone));
      for (v=0; v<2; v++) {
        PetscCall(PetscSectionGetDof(network->DofSection,cone[v],&ncols));

        if (network->Je) {
          Juser = network->Je[3*e+1+v]; /* Jacobian(e,v) */
        } else Juser = NULL;
        PetscCall(DMNetworkIsGhostVertex(dm,cone[v],&ghost));
        PetscCall(MatSetPreallocationblock_private(Juser,nrows,rows,ncols,ghost,vd_nz,vo_nz));
      }

      /* Set preallocation for edge self */
      cstart = rstart;
      if (network->Je) {
        Juser = network->Je[3*e]; /* Jacobian(e,e) */
      } else Juser = NULL;
      PetscCall(MatSetPreallocationblock_private(Juser,nrows,rows,nrows,PETSC_FALSE,vd_nz,vo_nz));
    }
  }

  /* Set preallocation for vertices */
  /*--------------------------------*/
  PetscCall(DMNetworkGetVertexRange(dm,&vStart,&vEnd));
  if (vEnd - vStart) vptr = network->Jvptr;

  for (v=vStart; v<vEnd; v++) {
    /* Get row indices */
    PetscCall(DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&rstart));
    PetscCall(PetscSectionGetDof(network->DofSection,v,&nrows));
    if (!nrows) continue;

    PetscCall(DMNetworkIsGhostVertex(dm,v,&ghost));
    if (ghost) {
      PetscCall(PetscMalloc1(nrows,&rows_v));
    } else {
      rows_v = rows;
    }

    for (j=0; j<nrows; j++) rows_v[j] = j + rstart;

    /* Get supporting edges and connected vertices */
    PetscCall(DMNetworkGetSupportingEdges(dm,v,&nedges,&edges));

    for (e=0; e<nedges; e++) {
      /* Supporting edges */
      PetscCall(DMNetworkGetGlobalVecOffset(dm,edges[e],ALL_COMPONENTS,&cstart));
      PetscCall(PetscSectionGetDof(network->DofSection,edges[e],&ncols));

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      PetscCall(MatSetPreallocationblock_private(Juser,nrows,rows_v,ncols,ghost,vd_nz,vo_nz));

      /* Connected vertices */
      PetscCall(DMNetworkGetConnectedVertices(dm,edges[e],&cone));
      vc = (v == cone[0]) ? cone[1]:cone[0];
      PetscCall(DMNetworkIsGhostVertex(dm,vc,&ghost_vc));

      PetscCall(PetscSectionGetDof(network->DofSection,vc,&ncols));

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+2]; /* Jacobian(v,vc) */
      } else Juser = NULL;
      if (ghost_vc||ghost) {
        ghost2 = PETSC_TRUE;
      } else {
        ghost2 = PETSC_FALSE;
      }
      PetscCall(MatSetPreallocationblock_private(Juser,nrows,rows_v,ncols,ghost2,vd_nz,vo_nz));
    }

    /* Set preallocation for vertex self */
    PetscCall(DMNetworkIsGhostVertex(dm,v,&ghost));
    if (!ghost) {
      PetscCall(DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&cstart));
      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]]; /* Jacobian(v,v) */
      } else Juser = NULL;
      PetscCall(MatSetPreallocationblock_private(Juser,nrows,rows_v,nrows,PETSC_FALSE,vd_nz,vo_nz));
    }
    if (ghost) PetscCall(PetscFree(rows_v));
  }

  PetscCall(VecAssemblyBegin(vd_nz));
  PetscCall(VecAssemblyBegin(vo_nz));

  PetscCall(PetscMalloc2(localSize,&dnnz,localSize,&onnz));

  PetscCall(VecAssemblyEnd(vd_nz));
  PetscCall(VecAssemblyEnd(vo_nz));

  PetscCall(VecGetArray(vd_nz,&vdnz));
  PetscCall(VecGetArray(vo_nz,&vonz));
  for (j=0; j<localSize; j++) {
    dnnz[j] = (PetscInt)PetscRealPart(vdnz[j]);
    onnz[j] = (PetscInt)PetscRealPart(vonz[j]);
  }
  PetscCall(VecRestoreArray(vd_nz,&vdnz));
  PetscCall(VecRestoreArray(vo_nz,&vonz));
  PetscCall(VecDestroy(&vd_nz));
  PetscCall(VecDestroy(&vo_nz));

  PetscCall(MatSeqAIJSetPreallocation(*J,0,dnnz));
  PetscCall(MatMPIAIJSetPreallocation(*J,0,dnnz,0,onnz));
  PetscCall(MatSetOption(*J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  PetscCall(PetscFree2(dnnz,onnz));

  /* (2) Set matrix entries for edges */
  /*----------------------------------*/
  for (e=eStart; e<eEnd; e++) {
    /* Get row indices */
    PetscCall(DMNetworkGetGlobalVecOffset(dm,e,ALL_COMPONENTS,&rstart));
    PetscCall(PetscSectionGetDof(network->DofSection,e,&nrows));
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set matrix entries for connected vertices */
      PetscCall(DMNetworkGetConnectedVertices(dm,e,&cone));
      for (v=0; v<2; v++) {
        PetscCall(DMNetworkGetGlobalVecOffset(dm,cone[v],ALL_COMPONENTS,&cstart));
        PetscCall(PetscSectionGetDof(network->DofSection,cone[v],&ncols));

        if (network->Je) {
          Juser = network->Je[3*e+1+v]; /* Jacobian(e,v) */
        } else Juser = NULL;
        PetscCall(MatSetblock_private(Juser,nrows,rows,ncols,cstart,J));
      }

      /* Set matrix entries for edge self */
      cstart = rstart;
      if (network->Je) {
        Juser = network->Je[3*e]; /* Jacobian(e,e) */
      } else Juser = NULL;
      PetscCall(MatSetblock_private(Juser,nrows,rows,nrows,cstart,J));
    }
  }

  /* Set matrix entries for vertices */
  /*---------------------------------*/
  for (v=vStart; v<vEnd; v++) {
    /* Get row indices */
    PetscCall(DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&rstart));
    PetscCall(PetscSectionGetDof(network->DofSection,v,&nrows));
    if (!nrows) continue;

    PetscCall(DMNetworkIsGhostVertex(dm,v,&ghost));
    if (ghost) {
      PetscCall(PetscMalloc1(nrows,&rows_v));
    } else {
      rows_v = rows;
    }
    for (j=0; j<nrows; j++) rows_v[j] = j + rstart;

    /* Get supporting edges and connected vertices */
    PetscCall(DMNetworkGetSupportingEdges(dm,v,&nedges,&edges));

    for (e=0; e<nedges; e++) {
      /* Supporting edges */
      PetscCall(DMNetworkGetGlobalVecOffset(dm,edges[e],ALL_COMPONENTS,&cstart));
      PetscCall(PetscSectionGetDof(network->DofSection,edges[e],&ncols));

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      PetscCall(MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J));

      /* Connected vertices */
      PetscCall(DMNetworkGetConnectedVertices(dm,edges[e],&cone));
      vc = (v == cone[0]) ? cone[1]:cone[0];

      PetscCall(DMNetworkGetGlobalVecOffset(dm,vc,ALL_COMPONENTS,&cstart));
      PetscCall(PetscSectionGetDof(network->DofSection,vc,&ncols));

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+2]; /* Jacobian(v,vc) */
      } else Juser = NULL;
      PetscCall(MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J));
    }

    /* Set matrix entries for vertex self */
    if (!ghost) {
      PetscCall(DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&cstart));
      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]]; /* Jacobian(v,v) */
      } else Juser = NULL;
      PetscCall(MatSetblock_private(Juser,nrows,rows_v,nrows,cstart,J));
    }
    if (ghost) PetscCall(PetscFree(rows_v));
  }
  PetscCall(PetscFree(rows));

  PetscCall(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY));

  PetscCall(MatSetDM(*J,dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMNetworkDestroyComponentData(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       j,np;

  PetscFunctionBegin;
  if (network->header) {
    np = network->cloneshared->pEnd - network->cloneshared->pStart;
    for (j=0; j < np; j++) {
      PetscCall(PetscFree5(network->header[j].size,network->header[j].key,network->header[j].offset,network->header[j].nvar,network->header[j].offsetvarrel));
      PetscCall(PetscFree(network->cvalue[j].data));
    }
    PetscCall(PetscFree2(network->header,network->cvalue));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Network(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       j;

  PetscFunctionBegin;
  /*
    Developers Note: Due to the mixed topological definition of DMNetwork and data defined ON the
    network like DofSection, DataSection, *componentdataarray, and friends, when cloning, we share
    only the true topological data, and make our own data ON the network. Thus refct only refers
    to the number of references to topological data, and data ON the network is always destroyed.
    It is understood this is atypical for a DM, but is very intentional.
  */

  /* Always destroy data ON the network */
  PetscCall(PetscFree(network->Je));
  if (network->Jv) {
    PetscCall(PetscFree(network->Jvptr));
    PetscCall(PetscFree(network->Jv));
  }
  PetscCall(PetscSectionDestroy(&network->DataSection));
  PetscCall(PetscSectionDestroy(&network->DofSection));
  PetscCall(PetscFree(network->component));
  PetscCall(PetscFree(network->componentdataarray));
  PetscCall(DMNetworkDestroyComponentData(dm));

  PetscCall(DMDestroy(&network->plex)); /* this is cloned in DMClone_Network, so safe to destroy */

  /*
  Developers Note: The DMNetworkVertexInfo and DMNetworkEdgeInfo data structures are completely
  destroyed as they are again a mix of topological data:
    ISLocalToGlobalMapping            mapping;
    PetscSF                           sf;
  and data ON the network
    PetscSection                      DofSection;
    PetscSection                      GlobalDofSection;
  And the only way to access them currently is through DMNetworkAssembleGraphStructures which assembles
  everything. So we must destroy everything and require DMNetworkAssembleGraphStructures is called again
  for any clone.
  */
  PetscCall(ISLocalToGlobalMappingDestroy(&network->vertex.mapping));
  PetscCall(PetscSectionDestroy(&network->vertex.DofSection));
  PetscCall(PetscSectionDestroy(&network->vertex.GlobalDofSection));
  PetscCall(PetscSFDestroy(&network->vertex.sf));
  /* edge */
  PetscCall(ISLocalToGlobalMappingDestroy(&network->edge.mapping));
  PetscCall(PetscSectionDestroy(&network->edge.DofSection));
  PetscCall(PetscSectionDestroy(&network->edge.GlobalDofSection));
  PetscCall(PetscSFDestroy(&network->edge.sf));
  /* Destroy the potentially cloneshared data */
  if (--network->cloneshared->refct <= 0) {
  /* Developer Note: I'm not sure if vltog can be reused or not, as I'm not sure what it's purpose is. I
     naively think it can be reused. */
    PetscCall(PetscFree(network->cloneshared->vltog));
    for (j=0; j<network->cloneshared->Nsvtx; j++) PetscCall(PetscFree(network->cloneshared->svtx[j].sv));
    PetscCall(PetscFree(network->cloneshared->svtx));
    PetscCall(PetscFree2(network->cloneshared->subnetedge,network->cloneshared->subnetvtx));
    PetscCall(PetscTableDestroy(&network->cloneshared->svtable));
    PetscCall(PetscFree(network->cloneshared->subnet));
    PetscCall(PetscFree(network->cloneshared));
  }
  PetscCall(PetscFree(network)); /* Always freed as this structure is copied in a clone, not cloneshared */
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Network(DM dm,PetscViewer viewer)
{
  PetscBool      iascii;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    const PetscInt *cone,*vtx,*edges;
    PetscInt       vfrom,vto,i,j,nv,ne,nsv,p,nsubnet;
    DM_Network     *network = (DM_Network*)dm->data;

    nsubnet = network->cloneshared->Nsubnet; /* num of subnetworks */
    if (rank == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"  NSubnets: %" PetscInt_FMT "; NEdges: %" PetscInt_FMT "; NVertices: %" PetscInt_FMT "; NSharedVertices: %" PetscInt_FMT ".\n",nsubnet,network->cloneshared->NEdges,network->cloneshared->NVertices,network->cloneshared->Nsvtx));
    }

    PetscCall(DMNetworkGetSharedVertices(dm,&nsv,NULL));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  [%d] nEdges: %" PetscInt_FMT "; nVertices: %" PetscInt_FMT "; nSharedVertices: %" PetscInt_FMT "\n",rank,network->cloneshared->nEdges,network->cloneshared->nVertices,nsv));

    for (i=0; i<nsubnet; i++) {
      PetscCall(DMNetworkGetSubnetwork(dm,i,&nv,&ne,&vtx,&edges));
      if (ne) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     Subnet %" PetscInt_FMT ": nEdges %" PetscInt_FMT ", nVertices(include shared vertices) %" PetscInt_FMT "\n",i,ne,nv));
        for (j=0; j<ne; j++) {
          p = edges[j];
          PetscCall(DMNetworkGetConnectedVertices(dm,p,&cone));
          PetscCall(DMNetworkGetGlobalVertexIndex(dm,cone[0],&vfrom));
          PetscCall(DMNetworkGetGlobalVertexIndex(dm,cone[1],&vto));
          PetscCall(DMNetworkGetGlobalEdgeIndex(dm,edges[j],&p));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "       edge %" PetscInt_FMT ": %" PetscInt_FMT " ----> %" PetscInt_FMT "\n",p,vfrom,vto));
        }
      }
    }

    /* Shared vertices */
    PetscCall(DMNetworkGetSharedVertices(dm,NULL,&vtx));
    if (nsv) {
      PetscInt       gidx;
      PetscBool      ghost;
      const PetscInt *sv=NULL;

      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     SharedVertices:\n"));
      for (i=0; i<nsv; i++) {
        PetscCall(DMNetworkIsGhostVertex(dm,vtx[i],&ghost));
        if (ghost) continue;

        PetscCall(DMNetworkSharedVertexGetInfo(dm,vtx[i],&gidx,&nv,&sv));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "       svtx %" PetscInt_FMT ": global index %" PetscInt_FMT ", subnet[%" PetscInt_FMT "].%" PetscInt_FMT " ---->\n",i,gidx,sv[0],sv[1]));
        for (j=1; j<nv; j++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "                                           ----> subnet[%" PetscInt_FMT "].%" PetscInt_FMT "\n",sv[2*j],sv[2*j+1]));
        }
      }
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else PetscCheck(iascii,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Viewer type %s not yet supported for DMNetwork writing",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalBegin_Network(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMGlobalToLocalBegin(network->plex,g,mode,l));
  PetscFunctionReturn(0);
}

PetscErrorCode DMGlobalToLocalEnd_Network(DM dm, Vec g, InsertMode mode, Vec l)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMGlobalToLocalEnd(network->plex,g,mode,l));
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalBegin_Network(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMLocalToGlobalBegin(network->plex,l,mode,g));
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalToGlobalEnd_Network(DM dm, Vec l, InsertMode mode, Vec g)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  PetscCall(DMLocalToGlobalEnd(network->plex,l,mode,g));
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

.seealso: `DMNetworkSetVertexLocalToGlobalOrdering()`
@*/
PetscErrorCode DMNetworkGetVertexLocalToGlobalOrdering(DM dm,PetscInt vloc,PetscInt *vg)
{
  DM_Network  *network = (DM_Network*)dm->data;
  PetscInt    *vltog = network->cloneshared->vltog;

  PetscFunctionBegin;
  PetscCheck(vltog,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Must call DMNetworkSetVertexLocalToGlobalOrdering() first");
  *vg = vltog[vloc];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkSetVertexLocalToGlobalOrdering - Create and setup vertex local to global map

  Collective

  Input Parameters:
. dm - the dm object

  Level: advanced

.seealso: `DMNetworkGetGlobalVertexIndex()`
@*/
PetscErrorCode DMNetworkSetVertexLocalToGlobalOrdering(DM dm)
{
  DM_Network        *network=(DM_Network*)dm->data;
  MPI_Comm          comm;
  PetscMPIInt       rank,size,*displs=NULL,*recvcounts=NULL,remoterank;
  PetscBool         ghost;
  PetscInt          *vltog,nroots,nleaves,i,*vrange,k,N,lidx;
  const PetscSFNode *iremote;
  PetscSF           vsf;
  Vec               Vleaves,Vleaves_seq;
  VecScatter        ctx;
  PetscScalar       *varr,val;
  const PetscScalar *varr_read;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  if (size == 1) {
    nroots = network->cloneshared->vEnd - network->cloneshared->vStart;
    PetscCall(PetscMalloc1(nroots, &vltog));
    for (i=0; i<nroots; i++) vltog[i] = i;
    network->cloneshared->vltog = vltog;
    PetscFunctionReturn(0);
  }

  PetscCheck(network->cloneshared->distributecalled,comm, PETSC_ERR_ARG_WRONGSTATE,"Must call DMNetworkDistribute() first");
  if (network->cloneshared->vltog) PetscCall(PetscFree(network->cloneshared->vltog));

  PetscCall(DMNetworkSetSubMap_private(network->cloneshared->vStart,network->cloneshared->vEnd,&network->vertex.mapping));
  PetscCall(PetscSFGetSubSF(network->plex->sf, network->vertex.mapping, &network->vertex.sf));
  vsf = network->vertex.sf;

  PetscCall(PetscMalloc3(size+1,&vrange,size,&displs,size,&recvcounts));
  PetscCall(PetscSFGetGraph(vsf,&nroots,&nleaves,NULL,&iremote));

  for (i=0; i<size; i++) { displs[i] = i; recvcounts[i] = 1;}

  i         = nroots - nleaves; /* local number of vertices, excluding ghosts */
  vrange[0] = 0;
  PetscCallMPI(MPI_Allgatherv(&i,1,MPIU_INT,vrange+1,recvcounts,displs,MPIU_INT,comm));
  for (i=2; i<size+1; i++) {vrange[i] += vrange[i-1];}

  PetscCall(PetscMalloc1(nroots, &vltog));
  network->cloneshared->vltog = vltog;

  /* Set vltog for non-ghost vertices */
  k = 0;
  for (i=0; i<nroots; i++) {
    PetscCall(DMNetworkIsGhostVertex(dm,i+network->cloneshared->vStart,&ghost));
    if (ghost) continue;
    vltog[i] = vrange[rank] + k++;
  }
  PetscCall(PetscFree3(vrange,displs,recvcounts));

  /* Set vltog for ghost vertices */
  /* (a) create parallel Vleaves and sequential Vleaves_seq to convert local iremote[*].index to global index */
  PetscCall(VecCreate(comm,&Vleaves));
  PetscCall(VecSetSizes(Vleaves,2*nleaves,PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(Vleaves));
  PetscCall(VecGetArray(Vleaves,&varr));
  for (i=0; i<nleaves; i++) {
    varr[2*i]   = (PetscScalar)(iremote[i].rank);  /* rank of remote process */
    varr[2*i+1] = (PetscScalar)(iremote[i].index); /* local index in remote process */
  }
  PetscCall(VecRestoreArray(Vleaves,&varr));

  /* (b) scatter local info to remote processes via VecScatter() */
  PetscCall(VecScatterCreateToAll(Vleaves,&ctx,&Vleaves_seq));
  PetscCall(VecScatterBegin(ctx,Vleaves,Vleaves_seq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,Vleaves,Vleaves_seq,INSERT_VALUES,SCATTER_FORWARD));

  /* (c) convert local indices to global indices in parallel vector Vleaves */
  PetscCall(VecGetSize(Vleaves_seq,&N));
  PetscCall(VecGetArrayRead(Vleaves_seq,&varr_read));
  for (i=0; i<N; i+=2) {
    remoterank = (PetscMPIInt)PetscRealPart(varr_read[i]);
    if (remoterank == rank) {
      k = i+1; /* row number */
      lidx = (PetscInt)PetscRealPart(varr_read[i+1]);
      val  = (PetscScalar)vltog[lidx]; /* global index for non-ghost vertex computed above */
      PetscCall(VecSetValues(Vleaves,1,&k,&val,INSERT_VALUES));
    }
  }
  PetscCall(VecRestoreArrayRead(Vleaves_seq,&varr_read));
  PetscCall(VecAssemblyBegin(Vleaves));
  PetscCall(VecAssemblyEnd(Vleaves));

  /* (d) Set vltog for ghost vertices by copying local values of Vleaves */
  PetscCall(VecGetArrayRead(Vleaves,&varr_read));
  k = 0;
  for (i=0; i<nroots; i++) {
    PetscCall(DMNetworkIsGhostVertex(dm,i+network->cloneshared->vStart,&ghost));
    if (!ghost) continue;
    vltog[i] = (PetscInt)PetscRealPart(varr_read[2*k+1]); k++;
  }
  PetscCall(VecRestoreArrayRead(Vleaves,&varr_read));

  PetscCall(VecDestroy(&Vleaves));
  PetscCall(VecDestroy(&Vleaves_seq));
  PetscCall(VecScatterDestroy(&ctx));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMISAddSize_private(DM_Network *network,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *nidx)
{
  PetscInt                 i,j,ncomps,nvar,key,offset=0;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offset));
  ncomps = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  for (i=0; i<ncomps; i++) {
    key  = header->key[i];
    nvar = header->nvar[i];
    for (j=0; j<numkeys; j++) {
      if (key == keys[j]) {
        if (!blocksize || blocksize[j] == -1) {
          *nidx += nvar;
        } else {
          *nidx += nselectedvar[j]*nvar/blocksize[j];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMISComputeIdx_private(DM dm,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],PetscInt *ii,PetscInt *idx)
{
  PetscInt                 i,j,ncomps,nvar,key,offsetg,k,k1,offset=0;
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offset));
  ncomps = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  for (i=0; i<ncomps; i++) {
    key  = header->key[i];
    nvar = header->nvar[i];
    for (j=0; j<numkeys; j++) {
      if (key != keys[j]) continue;

      PetscCall(DMNetworkGetGlobalVecOffset(dm,p,i,&offsetg));
      if (!blocksize || blocksize[j] == -1) {
        for (k=0; k<nvar; k++) idx[(*ii)++] = offsetg + k;
      } else {
        for (k=0; k<nvar; k+=blocksize[j]) {
          for (k1=0; k1<nselectedvar[j]; k1++) idx[(*ii)++] = offsetg + k + selectedvar[j][k1];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkCreateIS - Create an index set object from the global vector of the network

  Collective

  Input Parameters:
+ dm - DMNetwork object
. numkeys - number of keys
. keys - array of keys that define the components of the variables you wish to extract
. blocksize - block size of the variables associated to the component
. nselectedvar - number of variables in each block to select
- selectedvar - the offset into the block of each variable in each block to select

  Output Parameters:
. is - the index set

  Level: Advanced

  Notes:
    Use blocksize[i] of -1 to indicate select all the variables of the i-th component, for which nselectvar[i] and selectedvar[i] are ignored. Use NULL, NULL, NULL to indicate for all selected components one wishes to obtain all the values of that component. For example, DMNetworkCreateIS(dm,1,&key,NULL,NULL,NULL,&is) will return an is that extracts all the variables for the 0-th component.

.seealso: `DMNetworkCreate()`, `ISCreateGeneral()`, `DMNetworkCreateLocalIS()`
@*/
PetscErrorCode DMNetworkCreateIS(DM dm,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],IS *is)
{
  MPI_Comm       comm;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,p,estart,eend,vstart,vend,nidx,*idx;
  PetscBool      ghost;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));

  /* Check input parameters */
  for (i=0; i<numkeys; i++) {
    if (!blocksize || blocksize[i] == -1) continue;
    PetscCheck(nselectedvar[i] <= blocksize[i],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"number of selectedvariables %" PetscInt_FMT " cannot be larger than blocksize %" PetscInt_FMT,nselectedvar[i],blocksize[i]);
  }

  PetscCall(DMNetworkGetEdgeRange(dm,&estart,&eend));
  PetscCall(DMNetworkGetVertexRange(dm,&vstart,&vend));

  /* Get local number of idx */
  nidx = 0;
  for (p=estart; p<eend; p++) {
    PetscCall(DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx));
  }
  for (p=vstart; p<vend; p++) {
    PetscCall(DMNetworkIsGhostVertex(dm,p,&ghost));
    if (ghost) continue;
    PetscCall(DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx));
  }

  /* Compute idx */
  PetscCall(PetscMalloc1(nidx,&idx));
  i = 0;
  for (p=estart; p<eend; p++) {
    PetscCall(DMISComputeIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx));
  }
  for (p=vstart; p<vend; p++) {
    PetscCall(DMNetworkIsGhostVertex(dm,p,&ghost));
    if (ghost) continue;
    PetscCall(DMISComputeIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx));
  }

  /* Create is */
  PetscCall(ISCreateGeneral(comm,nidx,idx,PETSC_COPY_VALUES,is));
  PetscCall(PetscFree(idx));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMISComputeLocalIdx_private(DM dm,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],PetscInt *ii,PetscInt *idx)
{
  PetscInt                 i,j,ncomps,nvar,key,offsetl,k,k1,offset=0;
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  PetscCall(PetscSectionGetOffset(network->DataSection,p,&offset));
  ncomps = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  for (i=0; i<ncomps; i++) {
    key  = header->key[i];
    nvar = header->nvar[i];
    for (j=0; j<numkeys; j++) {
      if (key != keys[j]) continue;

      PetscCall(DMNetworkGetLocalVecOffset(dm,p,i,&offsetl));
      if (!blocksize || blocksize[j] == -1) {
        for (k=0; k<nvar; k++) idx[(*ii)++] = offsetl + k;
      } else {
        for (k=0; k<nvar; k+=blocksize[j]) {
          for (k1=0; k1<nselectedvar[j]; k1++) idx[(*ii)++] = offsetl + k + selectedvar[j][k1];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMNetworkCreateLocalIS - Create an index set object from the local vector of the network

  Not collective

  Input Parameters:
+ dm - DMNetwork object
. numkeys - number of keys
. keys - array of keys that define the components of the variables you wish to extract
. blocksize - block size of the variables associated to the component
. nselectedvar - number of variables in each block to select
- selectedvar - the offset into the block of each variable in each block to select

  Output Parameters:
. is - the index set

  Level: Advanced

  Notes:
    Use blocksize[i] of -1 to indicate select all the variables of the i-th component, for which nselectvar[i] and selectedvar[i] are ignored. Use NULL, NULL, NULL to indicate for all selected components one wishes to obtain all the values of that component. For example, DMNetworkCreateLocalIS(dm,1,&key,NULL,NULL,NULL,&is) will return an is that extracts all the variables for the 0-th component.

.seealso: `DMNetworkCreate()`, `DMNetworkCreateIS`, `ISCreateGeneral()`
@*/
PetscErrorCode DMNetworkCreateLocalIS(DM dm,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],IS *is)
{
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,p,pstart,pend,nidx,*idx;

  PetscFunctionBegin;
  /* Check input parameters */
  for (i=0; i<numkeys; i++) {
    if (!blocksize || blocksize[i] == -1) continue;
    PetscCheck(nselectedvar[i] <= blocksize[i],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"number of selectedvariables %" PetscInt_FMT " cannot be larger than blocksize %" PetscInt_FMT,nselectedvar[i],blocksize[i]);
  }

  pstart = network->cloneshared->pStart;
  pend   = network->cloneshared->pEnd;

  /* Get local number of idx */
  nidx = 0;
  for (p=pstart; p<pend; p++) {
    PetscCall(DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx));
  }

  /* Compute local idx */
  PetscCall(PetscMalloc1(nidx,&idx));
  i = 0;
  for (p=pstart; p<pend; p++) {
    PetscCall(DMISComputeLocalIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx));
  }

  /* Create is */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nidx,idx,PETSC_COPY_VALUES,is));
  PetscCall(PetscFree(idx));
  PetscFunctionReturn(0);
}
/*@
  DMNetworkFinalizeComponents - Sets up interal data structures for the sections and components. It is called after registering new components and adding all components
  to the cloned network. After calling this subroutine, no new components can be added to the network.

  Collective

  Input Parameters:
. dm - the dmnetwork object

  Level: beginner

.seealso: `DMNetworkAddComponent()`, `DMNetworkRegisterComponent()`, `DMSetUp()`
@*/
PetscErrorCode DMNetworkFinalizeComponents(DM dm)
{
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (network->componentsetup) PetscFunctionReturn(0);
  PetscCall(DMNetworkComponentSetUp(dm));
  PetscCall(DMNetworkVariablesSetUp(dm));
  PetscCall(DMSetLocalSection(network->plex,network->DofSection));
  PetscCall(DMGetGlobalSection(network->plex,&network->GlobalDofSection));
  network->componentsetup = PETSC_TRUE;
  PetscFunctionReturn(0);
}
