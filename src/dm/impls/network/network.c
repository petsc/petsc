#include <petsc/private/dmnetworkimpl.h>  /*I  "petscdmnetwork.h"  I*/

/*
  Creates the component header and value objects for a network point
*/
static PetscErrorCode SetUpNetworkHeaderComponentValue(DM dm,DMNetworkComponentHeader header,DMNetworkComponentValue cvalue)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Allocate arrays for component information */
  ierr = PetscCalloc5(header->maxcomps,&header->size,header->maxcomps,&header->key,header->maxcomps,&header->offset,header->maxcomps,&header->nvar,header->maxcomps,&header->offsetvarrel);CHKERRQ(ierr);
  ierr = PetscCalloc1(header->maxcomps,&cvalue->data);CHKERRQ(ierr);

  /* The size of the header is the size of struct _p_DMNetworkComponentHeader. Since the struct contains PetscInt pointers we cannot use sizeof(struct). So, we need to explicitly calculate the size.
   If the data header struct changes then this header size calculation needs to be updated. */
  header->hsize = sizeof(struct _p_DMNetworkComponentHeader) + 5*header->maxcomps*sizeof(PetscInt);
  header->hsize /= sizeof(DMNetworkComponentGenericDataType);
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

.seealso: DMNetworkCreate()
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

.seealso: DMNetworkCreate(), DMNetworkSetNumSubNetworks()
@*/
PetscErrorCode DMNetworkGetNumSubNetworks(DM dm,PetscInt *nsubnet,PetscInt *Nsubnet)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (nsubnet) *nsubnet = network->nsubnet;
  if (Nsubnet) *Nsubnet = network->Nsubnet;
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

.seealso: DMNetworkCreate(), DMNetworkGetNumSubNetworks()
@*/
PetscErrorCode DMNetworkSetNumSubNetworks(DM dm,PetscInt nsubnet,PetscInt Nsubnet)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (network->Nsubnet != 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Network sizes alread set, cannot resize the network");

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(dm,nsubnet,2);
  PetscValidLogicalCollectiveInt(dm,Nsubnet,3);

  if (Nsubnet == PETSC_DECIDE) {
    if (nsubnet < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of local subnetworks %D cannot be less than 0",nsubnet);
    ierr = MPIU_Allreduce(&nsubnet,&Nsubnet,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
  }
  if (Nsubnet < 1) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Number of global subnetworks %D cannot be less than 1",Nsubnet);

  network->Nsubnet  = Nsubnet;
  network->nsubnet  = 0;       /* initia value; will be determind by DMNetworkAddSubnetwork() */
  ierr = PetscCalloc1(Nsubnet,&network->subnet);CHKERRQ(ierr);

  /* num of shared vertices */
  network->nsvtx = 0;
  network->Nsvtx = 0;
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

.seealso: DMNetworkCreate(), DMNetworkSetNumSubnetworks()
@*/
PetscErrorCode DMNetworkAddSubnetwork(DM dm,const char* name,PetscInt ne,PetscInt edgelist[],PetscInt *netnum)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,Nedge,j,Nvtx,nvtx;
  PetscBT        table;

  PetscFunctionBegin;
  for (i=0; i<ne; i++) {
    if (edgelist[2*i] == edgelist[2*i+1]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Edge %D has the same vertex %D at each endpoint",i,edgelist[2*i]);
  }
  /* Get global total Nvtx = max(edgelist[])+1 for this subnet */
  nvtx = -1; i = 0;
  for (j=0; j<ne; j++) {
    nvtx = PetscMax(nvtx, edgelist[i]); i++;
    nvtx = PetscMax(nvtx, edgelist[i]); i++;
  }
  nvtx++;
  ierr = MPIU_Allreduce(&nvtx,&Nvtx,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);

  /* Get local nvtx for this subnet */
  ierr = PetscBTCreate(Nvtx,&table);CHKERRQ(ierr);
  ierr = PetscBTMemzero(Nvtx,table);CHKERRQ(ierr);
  i = 0;
  for (j=0; j<ne; j++) {
    ierr = PetscBTSet(table,edgelist[i]);CHKERRQ(ierr);
    i++;
    ierr = PetscBTSet(table,edgelist[i]);CHKERRQ(ierr);
    i++;
  }
  nvtx = 0;
  for (j=0; j<Nvtx; j++) {
    if (PetscBTLookup(table,j)) nvtx++;
  }
  ierr = PetscBTDestroy(&table);CHKERRQ(ierr);

  /* Get global total Nedge for this subnet */
  ierr = MPIU_Allreduce(&ne,&Nedge,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);

  i = network->nsubnet;
  if (name) {
    ierr = PetscStrcpy(network->subnet[i].name,name);CHKERRQ(ierr);
  }
  network->subnet[i].nvtx     = nvtx; /* include ghost vertices */
  network->subnet[i].nedge    = ne;
  network->subnet[i].edgelist = edgelist;
  network->subnet[i].Nvtx     = Nvtx;
  network->subnet[i].Nedge    = Nedge;

  /* ----------------------------------------------------------
   p=v or e;
   subnet[0].pStart   = 0
   subnet[i+1].pStart = subnet[i].pEnd = subnet[i].pStart + (nE[i] or NV[i])
   ----------------------------------------------------------------------- */
  /* GLOBAL subnet[].vStart and vEnd, used by DMNetworkLayoutSetUp() */
  network->subnet[i].vStart = network->NVertices;
  network->subnet[i].vEnd   = network->subnet[i].vStart + network->subnet[i].Nvtx; /* global vEnd of subnet[i] */

  network->nVertices += nvtx; /* include ghost vertices */
  network->NVertices += network->subnet[i].Nvtx;

  /* LOCAL subnet[].eStart and eEnd, used by DMNetworkLayoutSetUp() */
  network->subnet[i].eStart = network->nEdges;
  network->subnet[i].eEnd   = network->subnet[i].eStart + ne;
  network->nEdges += ne;
  network->NEdges += network->subnet[i].Nedge;

  ierr = PetscStrcpy(network->subnet[i].name,name);CHKERRQ(ierr);
  if (netnum) *netnum = network->nsubnet;
  network->nsubnet++;
  PetscFunctionReturn(0);
}

/*
  SetUp a single svtx struct. See SVtx defined in dmnetworkimpl.h
  Set gidx and type if input v=(net,idx) is a from_vertex;
  Get gid, type and index in the svtx array if input v=(net,idx) is a to_vertex.

  Input:  Nsvtx, svtx, net, idx, gidx
  Output: gidx, svtype, svtx_idx
 */
static PetscErrorCode SVtxSetUp(PetscInt Nsvtx,SVtx *svtx,PetscInt net,PetscInt idx,PetscInt *gidx,SVtxType *svtype,PetscInt *svtx_idx)
{
  PetscInt i,j,*svto;
  SVtxType vtype;

  PetscFunctionBegin;
  if (!Nsvtx) PetscFunctionReturn(0);

  vtype = SVNONE;
  for (i=0; i<Nsvtx; i++) {
    if (net == svtx[i].sv[0] && idx == svtx[i].sv[1]) {
      /* (1) input vertex net.idx is a shared from_vertex, set its global index and output its svtype */
      svtx[i].gidx = *gidx; /* set gidx */
      vtype        = SVFROM;
    } else { /* loop over svtx[i].n */
      for (j=1; j<svtx[i].n; j++) {
        svto = svtx[i].sv + 2*j;
        if (net == svto[0] && idx == svto[1]) {
          /* input vertex net.idx is a shared to_vertex, output its global index and its svtype */
          *gidx = svtx[i].gidx; /* output gidx for to_vertex */
          vtype = SVTO;
        }
      }
    }
    if (vtype != SVNONE) break;
  }
  if (svtype)   *svtype   = vtype;
  if (svtx_idx) *svtx_idx = i;
  PetscFunctionReturn(0);
}

/*
 Add a new shared vertex sv=(net,idx) to table svtas[ita]
*/
static PetscErrorCode TableAddSVtx(PetscTable *svtas,PetscInt ita,PetscInt* tdata,PetscInt *sv_wk,PetscInt *ii,PetscInt *sedgelist,PetscInt k,DM_Network *network,PetscInt **ta2sv)
{
  PetscInt       net,idx,gidx,i=*ii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  net = sv_wk[2*i]   = sedgelist[k];
  idx = sv_wk[2*i+1] = sedgelist[k+1];
  gidx = network->subnet[net].vStart + idx;
  ierr = PetscTableAdd(svtas[ita],gidx+1,tdata[ita]+1,INSERT_VALUES);CHKERRQ(ierr);
  *(ta2sv[ita] + tdata[ita]) = i; /* maps tdata to index of sv_wk; sv_wk keeps (net,idx) info */
  tdata[ita]++; (*ii)++;
  PetscFunctionReturn(0);
}

/*
  Create an array of shared vertices. See SVtx and SVtxType in dmnetworkimpl.h

  Input:  dm, Nsedgelist, sedgelist
  Output: Nsvtx,svtx

  Note: Output svtx is organized as
        sv(net[0],idx[0]) --> sv(net[1],idx[1])
                          --> sv(net[1],idx[1])
                          ...
                          --> sv(net[n-1],idx[n-1])
        and net[0] < net[1] < ... < net[n-1]
        where sv[0] has SVFROM type, sv[i], i>0, has SVTO type.
 */
static PetscErrorCode SVtxCreate(DM dm,PetscInt Nsedgelist,PetscInt *sedgelist,PetscInt *Nsvtx,SVtx **svtx)
{
  PetscErrorCode ierr;
  SVtx           *sedges = NULL;
  PetscInt       *sv,k,j,nsv,*tdata,**ta2sv;
  PetscTable     *svtas;
  PetscInt       gidx,net,idx,i,nta,ita,idx_from,idx_to,n,*sv_wk;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscTablePosition ppos;

  PetscFunctionBegin;
  /* (1) Crete ctables svtas */
  ierr = PetscCalloc4(Nsedgelist,&svtas,Nsedgelist,&tdata,4*Nsedgelist,&sv_wk,2*Nsedgelist,&ta2sv);CHKERRQ(ierr);

  j   = 0;   /* sedgelist counter */
  k   = 0;   /* sedgelist vertex counter j = 4*k */
  i   = 0;   /* sv_wk (vertices added to the ctables) counter */
  nta = 0;   /* num of sv tables created */

  /* for j=0 */
  ierr = PetscTableCreate(2*Nsedgelist,network->NVertices+1,&svtas[nta]);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*Nsedgelist,&ta2sv[nta]);CHKERRQ(ierr);

  ierr = TableAddSVtx(svtas,nta,tdata,sv_wk,&i,sedgelist,k,network,ta2sv);CHKERRQ(ierr);
  ierr = TableAddSVtx(svtas,nta,tdata,sv_wk,&i,sedgelist,k+2,network,ta2sv);CHKERRQ(ierr);
  nta++; k += 4;

  for (j = 1; j < Nsedgelist; j++) {
    for (ita = 0; ita < nta; ita++) {
      /* vfrom */
      net = sedgelist[k]; idx = sedgelist[k+1];
      gidx = network->subnet[net].vStart + idx; /* global index of the vertex net.idx before merging shared vertices */
      ierr = PetscTableFind(svtas[ita],gidx+1,&idx_from);CHKERRQ(ierr);

      /* vto */
      net = sedgelist[k+2]; idx = sedgelist[k+3];
      gidx = network->subnet[net].vStart + idx;
      ierr = PetscTableFind(svtas[ita],gidx+1,&idx_to);CHKERRQ(ierr);

      if (idx_from || idx_to) { /* vfrom or vto is on table svtas[ita] */
        idx_from--; idx_to--;
        if (idx_from < 0) { /* vto is on svtas[ita] */
          ierr = TableAddSVtx(svtas,ita,tdata,sv_wk,&i,sedgelist,k,network,ta2sv);CHKERRQ(ierr);
          break;
        } else if (idx_to < 0) {
          ierr = TableAddSVtx(svtas,ita,tdata,sv_wk,&i,sedgelist,k+2,network,ta2sv);CHKERRQ(ierr);
          break;
        }
      }
    }

    if (ita == nta) {
      ierr = PetscTableCreate(2*Nsedgelist,network->NVertices+1,&svtas[nta]);CHKERRQ(ierr);
      ierr = PetscMalloc1(2*Nsedgelist, &ta2sv[nta]);CHKERRQ(ierr);

      ierr = TableAddSVtx(svtas,nta,tdata,sv_wk,&i,sedgelist,k,network,ta2sv);CHKERRQ(ierr);
      ierr = TableAddSVtx(svtas,nta,tdata,sv_wk,&i,sedgelist,k+2,network,ta2sv);CHKERRQ(ierr);
      nta++;
    }
    k += 4;
  }

  /* (2) Construct sedges from ctable
     sedges: edges connect vertex sv[0]=(net[0],idx[0]) to vertices sv[k], k=1,...,n-1;
     net[k], k=0, ...,n-1, are in ascending order */
  ierr = PetscMalloc1(nta,&sedges);CHKERRQ(ierr);
  for (nsv = 0; nsv < nta; nsv++) {
    /* for a single svtx, put shared vertices in ascending order of gidx */
    ierr = PetscTableGetCount(svtas[nsv],&n);CHKERRQ(ierr);
    ierr = PetscCalloc1(2*n,&sv);CHKERRQ(ierr);
    sedges[nsv].sv   = sv;
    sedges[nsv].n    = n;
    sedges[nsv].gidx = -1; /* initialization */

    ierr = PetscTableGetHeadPosition(svtas[nsv],&ppos);CHKERRQ(ierr);
    for (k=0; k<n; k++) { /* gidx is sorted in ascending order */
      ierr = PetscTableGetNext(svtas[nsv],&ppos,&gidx,&i);CHKERRQ(ierr);
      gidx--; i--;

      j = ta2sv[nsv][i]; /* maps i to index of sv_wk */
      sv[2*k]   = sv_wk[2*j];
      sv[2*k+1] = sv_wk[2*j + 1];
    }
  }

  for (j=0; j<nta; j++) {
    ierr = PetscTableDestroy(&svtas[j]);CHKERRQ(ierr);
    ierr = PetscFree(ta2sv[j]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(svtas,tdata,sv_wk,ta2sv);CHKERRQ(ierr);

  *Nsvtx = nta;
  *svtx  = sedges;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMNetworkLayoutSetUp_Coupling(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,j,ctr,np,*edges,*subnetvtx,*subnetedge,vStart;
  PetscInt       k,*vidxlTog,Nsv=0,Nsubnet=network->Nsubnet;
  PetscInt       *sedgelist=network->sedgelist;
  const PetscInt *cone;
  MPI_Comm       comm;
  PetscMPIInt    size,rank,*recvcounts=NULL,*displs=NULL;
  PetscInt       net,idx,gidx,nmerged,e,v,vfrom,vto,*vrange,*eowners,gidx_from,net_from,sv_idx;
  SVtxType       svtype = SVNONE;
  SVtx           *svtx=NULL;
  PetscSection   sectiong;

  PetscFunctionBegin;
  /* This implementation requires user input each subnet by a single processor, thus subnet[net].nvtx=subnet[net].Nvtx */
  for (net=0; net<Nsubnet; net++) {
    if (network->subnet[net].nvtx && network->subnet[net].nvtx != network->subnet[net].Nvtx) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"subnetwork %D local num of vertices %D != %D global num",net,network->subnet[net].nvtx,network->subnet[net].Nvtx);
  }

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* (1) Create svtx[] from sedgelist */
  /* -------------------------------- */
  /* Nsv: global number of SVtx; svtx: shared vertices, see SVtx in dmnetworkimpl.h */
  ierr = SVtxCreate(dm,network->Nsvtx,sedgelist,&Nsv,&svtx);CHKERRQ(ierr);

  /* (2) Setup svtx; Shared vto vertices are merged to their vfrom vertex with same global vetex index (gidx) */
  /* -------------------------------------------------------------------------------------------------------- */
  /* (2.1) compute vrage[rank]: global index of 1st local vertex in proc[rank] */
  ierr = PetscMalloc3(size+1,&vrange,size,&displs,size,&recvcounts);CHKERRQ(ierr);
  for (i=0; i<size; i++) {displs[i] = i; recvcounts[i] = 1;}

  vrange[0] = 0;
  ierr = MPI_Allgatherv(&network->nVertices,1,MPIU_INT,vrange+1,recvcounts,displs,MPIU_INT,comm);CHKERRMPI(ierr);
  for (i=2; i<size+1; i++) {
    vrange[i] += vrange[i-1];
  }

  /* (2.2) Create vidxlTog: maps UN-MERGED local vertex index i to global index gidx (plex, excluding ghost vertices) */
  ierr = PetscMalloc1(network->nVertices,&vidxlTog);CHKERRQ(ierr);
  i = 0; gidx = 0;
  nmerged = 0; /* local num of merged vertices */
  network->nsvtx = 0;
  for (net=0; net<Nsubnet; net++) {
    for (idx=0; idx<network->subnet[net].Nvtx; idx++) {
      gidx_from = gidx;
      sv_idx    = -1;

      ierr = SVtxSetUp(Nsv,svtx,net,idx,&gidx_from,&svtype,&sv_idx);CHKERRQ(ierr);
      if (svtype == SVTO) {
        if (network->subnet[net].nvtx) {/* this proc owns sv_to */
          net_from = svtx[sv_idx].sv[0]; /* subnet num of its shared vertex */
          if (network->subnet[net_from].nvtx == 0) {
            /* this proc does not own v_from, thus a new local coupling vertex */
            network->nsvtx++;
          }
          vidxlTog[i++] = gidx_from;
          nmerged++; /* a coupling vertex -- merged */
        }
      } else {
        if (svtype == SVFROM) {
          if (network->subnet[net].nvtx) {
            /* this proc owns this v_from, a new local coupling vertex */
            network->nsvtx++;
          }
        }
        if (network->subnet[net].nvtx) vidxlTog[i++] = gidx;
        gidx++;
      }
    }
  }
#if defined(PETSC_USE_DEBUG)
  if (i != network->nVertices) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"%D != %D nVertices",i,network->nVertices);
#endif

  /* (2.3) Setup svtable for querry shared vertices */
  for (v=0; v<Nsv; v++) {
    gidx = svtx[v].gidx;
    ierr = PetscTableAdd(network->svtable,gidx+1,v+1,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* (2.4) Shared vertices in the subnetworks are merged, update global NVertices: np = sum(local nmerged) */
  ierr = MPI_Allreduce(&nmerged,&np,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
  network->NVertices -= np;

  ierr = PetscCalloc1(2*network->nEdges,&edges);CHKERRQ(ierr);

  ctr = 0;
  for (net=0; net<Nsubnet; net++) {
    for (j = 0; j < network->subnet[net].nedge; j++) {
      /* vfrom: */
      i = network->subnet[net].edgelist[2*j] + (network->subnet[net].vStart - vrange[rank]);
      edges[2*ctr] = vidxlTog[i];

      /* vto */
      i = network->subnet[net].edgelist[2*j+1] + (network->subnet[net].vStart - vrange[rank]);
      edges[2*ctr+1] = vidxlTog[i];
      ctr++;
    }
  }
  ierr = PetscFree3(vrange,displs,recvcounts);CHKERRQ(ierr);
  ierr = PetscFree(vidxlTog);CHKERRQ(ierr);

  /* (3) Create network->plex */
  ierr = DMCreate(comm,&network->plex);CHKERRQ(ierr);
  ierr = DMSetType(network->plex,DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(network->plex,1);CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMPlexBuildFromCellList(network->plex,network->nEdges,network->nVertices-nmerged,2,edges);CHKERRQ(ierr);
  } else {
    ierr = DMPlexBuildFromCellListParallel(network->plex,network->nEdges,network->nVertices-nmerged,PETSC_DECIDE,2,edges,NULL);CHKERRQ(ierr);
  }

  ierr = DMPlexGetChart(network->plex,&network->pStart,&network->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,0,&network->eStart,&network->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,1,&network->vStart,&network->vEnd);CHKERRQ(ierr);
  vStart = network->vStart;

  ierr = PetscSectionCreate(comm,&network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm,&network->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DataSection,network->pStart,network->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DofSection,network->pStart,network->pEnd);CHKERRQ(ierr);

  np = network->pEnd - network->pStart;
  ierr = PetscCalloc2(np,&network->header,np,&network->cvalue);CHKERRQ(ierr);
  for (i=0; i<np; i++) {
    network->header[i].maxcomps = 1;
    ierr = SetUpNetworkHeaderComponentValue(dm,&network->header[i],&network->cvalue[i]);CHKERRQ(ierr);
  }

  /* (4) Create edge and vertex arrays for the subnetworks */
  ierr = PetscCalloc2(network->nEdges,&subnetedge,network->nVertices+network->nsvtx,&subnetvtx);CHKERRQ(ierr); /* Maps local edge/vertex to local subnetwork's edge/vertex */
  network->subnetedge = subnetedge;
  network->subnetvtx  = subnetvtx;
  for (j=0; j < Nsubnet; j++) {
    network->subnet[j].edges = subnetedge;
    subnetedge              += network->subnet[j].nedge;

    network->subnet[j].vertices = subnetvtx;
    subnetvtx                  += network->subnet[j].nvtx;
  }
  network->svertices = subnetvtx;

  /* Get edge ownership */
  ierr = PetscMalloc1(size+1,&eowners);CHKERRQ(ierr);
  np = network->eEnd - network->eStart; /* num of local edges */
  ierr = MPI_Allgather(&np,1,MPIU_INT,eowners+1,1,MPIU_INT,comm);CHKERRMPI(ierr);
  eowners[0] = 0;
  for (i=2; i<=size; i++) eowners[i] += eowners[i-1];

  /* Setup edge and vertex arrays for subnetworks */
  e = 0;
  for (i=0; i < Nsubnet; i++) {
    ctr = 0;
    for (j = 0; j < network->subnet[i].nedge; j++) {
      /* edge e */
      network->header[e].index    = e + eowners[rank]; /* Global edge index */
      network->header[e].subnetid = i;                 /* Subnetwork id */
      network->subnet[i].edges[j] = e;
      network->header[e].ndata           = 0;
      network->header[e].offset[0]       = 0;
      network->header[e].offsetvarrel[0] = 0;
      ierr = PetscSectionAddDof(network->DataSection,e,network->header[e].hsize);CHKERRQ(ierr);

      /* connected vertices */
      ierr = DMPlexGetCone(network->plex,e,&cone);CHKERRQ(ierr);

      /* vertex cone[0] */
      v = cone[0];
      network->header[v].index    = edges[2*e];   /* Global vertex index */
      network->header[v].subnetid = i;            /* Subnetwork id */
      vfrom = network->subnet[i].edgelist[2*ctr];
      network->subnet[i].vertices[vfrom] = v;     /* user's subnet[].dix = petsc's v */

      /* vertex cone[1] */
      v = cone[1];
      network->header[v].index    = edges[2*e+1]; /* Global vertex index */
      network->header[v].subnetid = i;
      vto = network->subnet[i].edgelist[2*ctr+1];
      network->subnet[i].vertices[vto]= v;

      e++; ctr++;
    }
  }
  ierr = PetscFree(eowners);CHKERRQ(ierr);
  ierr = PetscFree(edges);CHKERRQ(ierr);

  /* Set vertex array for the subnetworks */
  k = 0;
  for (v=vStart; v<network->vEnd; v++) { /* local vertices, including ghosts */
    network->header[v].ndata           = 0;
    network->header[v].offset[0]       = 0;
    network->header[v].offsetvarrel[0] = 0;
    ierr = PetscSectionAddDof(network->DataSection,v,network->header[v].hsize);CHKERRQ(ierr);

    /* shared vertex */
    ierr = PetscTableFind(network->svtable,network->header[v].index+1,&i);CHKERRQ(ierr);
    if (i) network->svertices[k++] = v;
  }
#if defined(PETSC_USE_DEBUG)
  if (k != network->nsvtx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"k %D != %D nsvtx",k,network->nsvtx);
#endif

  network->svtx  = svtx;
  network->Nsvtx = Nsv;
  ierr = PetscFree(sedgelist);CHKERRQ(ierr);

  /* Create a global section to be used by DMNetworkIsGhostVertex() which is a non-collective routine */
  /* see snes_tutorials_network-ex1_4 */
  ierr = DMGetGlobalSection(network->plex,&sectiong);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMNetworkLayoutSetUp - Sets up the bare layout (graph) for the network

  Collective on dm

  Input Parameters:
. dm - the dmnetwork object

  Notes:
  This routine should be called after the network sizes and edgelists have been provided. It creates
  the bare layout of the network and sets up the network to begin insertion of components.

  All the components should be registered before calling this routine.

  Level: beginner

.seealso: DMNetworkSetNumSubNetworks(), DMNetworkAddSubnetwork()
@*/
PetscErrorCode DMNetworkLayoutSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,j,ctr,Nsubnet=network->Nsubnet,*eowners,np,*edges,*subnetvtx,*subnetedge,e,v,vfrom,vto;
  const PetscInt *cone;
  MPI_Comm       comm;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  if (network->nsubnet != network->Nsubnet) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Must call DMNetworkAddSubnetwork() %D times",network->Nsubnet);

  /* Create svtable for querry shared vertices */
  ierr = PetscTableCreate(network->Nsvtx,network->NVertices+1,&network->svtable);CHKERRQ(ierr);

  if (network->Nsvtx) {
    ierr = DMNetworkLayoutSetUp_Coupling(dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* Create LOCAL edgelist in global vertex ordering for the network by concatenating local input edgelists of the subnetworks */
  ierr = PetscCalloc1(2*network->nEdges,&edges);CHKERRQ(ierr);
  ctr = 0;
  for (i=0; i < Nsubnet; i++) {
    for (j = 0; j < network->subnet[i].nedge; j++) {
      edges[2*ctr]   = network->subnet[i].vStart + network->subnet[i].edgelist[2*j];
      edges[2*ctr+1] = network->subnet[i].vStart + network->subnet[i].edgelist[2*j+1];
      ctr++;
    }
  }

  /* Create network->plex; One dimensional network, numCorners=2 */
  ierr = DMCreate(comm,&network->plex);CHKERRQ(ierr);
  ierr = DMSetType(network->plex,DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(network->plex,1);CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMPlexBuildFromCellList(network->plex,network->nEdges,network->nVertices,2,edges);CHKERRQ(ierr);
  } else {
    ierr = DMPlexBuildFromCellListParallel(network->plex,network->nEdges,network->nVertices,PETSC_DECIDE,2,edges,NULL);CHKERRQ(ierr);
  }

  ierr = DMPlexGetChart(network->plex,&network->pStart,&network->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,0,&network->eStart,&network->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(network->plex,1,&network->vStart,&network->vEnd);CHKERRQ(ierr);

  ierr = PetscSectionCreate(comm,&network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm,&network->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DataSection,network->pStart,network->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(network->DofSection,network->pStart,network->pEnd);CHKERRQ(ierr);

  np = network->pEnd - network->pStart;
  ierr = PetscCalloc2(np,&network->header,np,&network->cvalue);CHKERRQ(ierr);
  for (i=0; i < np; i++) {
    network->header[i].maxcomps = 1;
    ierr = SetUpNetworkHeaderComponentValue(dm,&network->header[i],&network->cvalue[i]);CHKERRQ(ierr);
  }

  /* Create edge and vertex arrays for the subnetworks
     This implementation assumes that DMNetwork reads
     (1) a single subnetwork in parallel; or
     (2) n subnetworks using n processors, one subnetwork/processor.
   */
  ierr = PetscCalloc2(network->nEdges,&subnetedge,network->nVertices,&subnetvtx);CHKERRQ(ierr); /* Maps local edge/vertex to local subnetwork's edge/vertex */

  network->subnetedge = subnetedge;
  network->subnetvtx  = subnetvtx;
  for (j=0; j < network->Nsubnet; j++) {
    network->subnet[j].edges = subnetedge;
    subnetedge              += network->subnet[j].nedge;

    network->subnet[j].vertices = subnetvtx;
    subnetvtx                  += network->subnet[j].nvtx;
  }

  /* Get edge ownership */
  ierr = PetscMalloc1(size+1,&eowners);CHKERRQ(ierr);
  np = network->eEnd - network->eStart;
  ierr = MPI_Allgather(&np,1,MPIU_INT,eowners+1,1,MPIU_INT,comm);CHKERRMPI(ierr);
  eowners[0] = 0;
  for (i=2; i<=size; i++) eowners[i] += eowners[i-1];

  /* Setup edge and vertex arrays for subnetworks */
  e = 0;
  for (i=0; i < Nsubnet; i++) {
    ctr = 0;
    for (j = 0; j < network->subnet[i].nedge; j++) {
      /* edge e */
      network->header[e].index    = e + eowners[rank];   /* Global edge index */
      network->header[e].subnetid = i;
      network->subnet[i].edges[j] = e;

      network->header[e].ndata           = 0;
      network->header[e].offset[0]       = 0;
      network->header[e].offsetvarrel[0] = 0;
      ierr = PetscSectionAddDof(network->DataSection,e,network->header[e].hsize);CHKERRQ(ierr);

      /* connected vertices */
      ierr = DMPlexGetCone(network->plex,e,&cone);CHKERRQ(ierr);

      /* vertex cone[0] */
      v = cone[0];
      network->header[v].index     = edges[2*e];  /* Global vertex index */
      network->header[v].subnetid  = i;           /* Subnetwork id */
      if (Nsubnet == 1) {
        network->subnet[i].vertices[v - network->vStart] = v; /* user's subnet[].idx = petsc's v */
      } else {
        vfrom = network->subnet[i].edgelist[2*ctr];     /* =subnet[i].idx, Global index! */
        network->subnet[i].vertices[vfrom] = v; /* user's subnet[].dix = petsc's v */
      }

      /* vertex cone[1] */
      v = cone[1];
      network->header[v].index    = edges[2*e+1];   /* Global vertex index */
      network->header[v].subnetid = i;              /* Subnetwork id */
      if (Nsubnet == 1) {
        network->subnet[i].vertices[v - network->vStart] = v; /* user's subnet[].idx = petsc's v */
      } else {
        vto = network->subnet[i].edgelist[2*ctr+1];     /* =subnet[i].idx, Global index! */
        network->subnet[i].vertices[vto] = v; /* user's subnet[].dix = petsc's v */
      }

      e++; ctr++;
    }
  }
  ierr = PetscFree(eowners);CHKERRQ(ierr);
  ierr = PetscFree(edges);CHKERRQ(ierr); /* local edge list with global idx used by DMPlexBuildFromCellList() */

  for (v = network->vStart; v < network->vEnd; v++) {
    network->header[v].ndata           = 0;
    network->header[v].offset[0]       = 0;
    network->header[v].offsetvarrel[0] = 0;
    ierr = PetscSectionAddDof(network->DataSection,v,network->header[e].hsize);CHKERRQ(ierr);
  }
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

  Level: intermediate

.seealso: DMNetworkCreate(), DMNetworkAddSubnetwork(), DMNetworkLayoutSetUp()
@*/
PetscErrorCode DMNetworkGetSubnetwork(DM dm,PetscInt netnum,PetscInt *nv,PetscInt *ne,const PetscInt **vtx,const PetscInt **edge)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (netnum >= network->Nsubnet) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Subnet index %D exceeds the num of subnets %D",netnum,network->Nsubnet);
  if (nv) *nv     = network->subnet[netnum].nvtx;
  if (ne) *ne     = network->subnet[netnum].nedge;
  if (vtx) *vtx   = network->subnet[netnum].vertices;
  if (edge) *edge = network->subnet[netnum].edges;
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

.seealso: DMNetworkCreate(), DMNetworkAddSubnetwork(), DMNetworkGetSharedVertices()
@*/
PetscErrorCode DMNetworkAddSharedVertices(DM dm,PetscInt anetnum,PetscInt bnetnum,PetscInt nsvtx,PetscInt asvtx[],PetscInt bsvtx[])
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,nsubnet = network->Nsubnet,*sedgelist,Nsvtx=network->Nsvtx;

  PetscFunctionBegin;
  if (anetnum == bnetnum) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Subnetworks must have different netnum");
  if (anetnum < 0 || bnetnum < 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"netnum cannot be negative");
  if (!Nsvtx) {
    /* allocate network->sedgelist to hold at most 2*nsubnet pairs of shared vertices */
    ierr = PetscMalloc1(2*4*nsubnet,&network->sedgelist);CHKERRQ(ierr);
  }

  sedgelist = network->sedgelist;
  for (i=0; i<nsvtx; i++) {
    sedgelist[4*Nsvtx]   = anetnum; sedgelist[4*Nsvtx+1] = asvtx[i];
    sedgelist[4*Nsvtx+2] = bnetnum; sedgelist[4*Nsvtx+3] = bsvtx[i];
    Nsvtx++;
  }
  if (Nsvtx > 2*nsubnet) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"allocate more space for coupling edgelist");
  network->Nsvtx = Nsvtx;
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

.seealso: DMNetworkGetSubnetwork(), DMNetworkLayoutSetUp(), DMNetworkAddSharedVertices()
@*/
PetscErrorCode DMNetworkGetSharedVertices(DM dm,PetscInt *nsv,const PetscInt **svtx)
{
  DM_Network *net = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (net->Nsvtx) {
    *nsv  = net->nsvtx;
    *svtx = net->svertices;
  } else {
    *nsv  = 0;
    *svtx = NULL;
  }
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

.seealso: DMNetworkCreate(), DMNetworkLayoutSetUp()
@*/
PetscErrorCode DMNetworkRegisterComponent(DM dm,const char *name,size_t size,PetscInt *key)
{
  PetscErrorCode        ierr;
  DM_Network            *network = (DM_Network*) dm->data;
  DMNetworkComponent    *component=NULL,*newcomponent=NULL;
  PetscBool             flg=PETSC_FALSE;
  PetscInt              i;

  PetscFunctionBegin;
  if (!network->component) {
    ierr = PetscCalloc1(network->max_comps_registered,&network->component);CHKERRQ(ierr);
  }

  for (i=0; i < network->ncomponent; i++) {
    ierr = PetscStrcmp(network->component[i].name,name,&flg);CHKERRQ(ierr);
    if (flg) {
      *key = i;
      PetscFunctionReturn(0);
    }
  }

  if (network->ncomponent == network->max_comps_registered) {
    /* Reached max allowed so resize component */
    network->max_comps_registered += 2;
    ierr = PetscCalloc1(network->max_comps_registered,&newcomponent);CHKERRQ(ierr);
    /* Copy over the previous component info */
    for (i=0; i < network->ncomponent; i++) {
      ierr = PetscStrcpy(newcomponent[i].name,network->component[i].name);CHKERRQ(ierr);
      newcomponent[i].size = network->component[i].size;
    }
    /* Free old one */
    ierr = PetscFree(network->component);CHKERRQ(ierr);
    /* Update pointer */
    network->component = newcomponent;
  }

  component = &network->component[network->ncomponent];

  ierr = PetscStrcpy(component->name,name);CHKERRQ(ierr);
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

.seealso: DMNetworkGetEdgeRange()
@*/
PetscErrorCode DMNetworkGetVertexRange(DM dm,PetscInt *vStart,PetscInt *vEnd)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (vStart) *vStart = network->vStart;
  if (vEnd) *vEnd = network->vEnd;
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

.seealso: DMNetworkGetVertexRange()
@*/
PetscErrorCode DMNetworkGetEdgeRange(DM dm,PetscInt *eStart,PetscInt *eEnd)
{
  DM_Network *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  if (eStart) *eStart = network->eStart;
  if (eEnd) *eEnd = network->eEnd;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMNetworkGetIndex(DM dm,PetscInt p,PetscInt *index)
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
  DMNetworkGetGlobalEdgeIndex - Get the global numbering for the edge on the network

  Not Collective

  Input Parameters:
+ dm - DMNetwork object
- p - edge point

  Output Parameters:
. index - the global numbering for the edge

  Level: intermediate

.seealso: DMNetworkGetGlobalVertexIndex()
@*/
PetscErrorCode DMNetworkGetGlobalEdgeIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMNetworkGetIndex(dm,p,index);CHKERRQ(ierr);
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

.seealso: DMNetworkGetGlobalEdgeIndex()
@*/
PetscErrorCode DMNetworkGetGlobalVertexIndex(DM dm,PetscInt p,PetscInt *index)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMNetworkGetIndex(dm,p,index);CHKERRQ(ierr);
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

.seealso: DMNetworkRegisterComponent(), DMNetworkAddComponent()
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
  DMNetworkGetLocalVecOffset - Get the offset for accessing the variables associated with a component at the given vertex/edge from the local vector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - the edge/vertex point
- compnum - component number; use ALL_COMPONENTS if no specific component is requested

  Output Parameters:
. offset - the local offset

  Level: intermediate

.seealso: DMGetLocalVector(), DMNetworkGetComponent(), DMNetworkGetGlobalVecOffset()
@*/
PetscErrorCode DMNetworkGetLocalVecOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->plex->localSection,p,&offsetp);CHKERRQ(ierr);
  if (compnum == ALL_COMPONENTS) {
    *offset = offsetp;
    PetscFunctionReturn(0);
  }

  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetd);CHKERRQ(ierr);
  header = (DMNetworkComponentHeader)(network->componentdataarray+offsetd);
  *offset = offsetp + header->offsetvarrel[compnum];
  PetscFunctionReturn(0);
}

/*@
  DMNetworkGetGlobalVecOffset - Get the global offset for accessing the variables associated with a component for the given vertex/edge from the global vector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
. p - the edge/vertex point
- compnum - component number; use ALL_COMPONENTS if no specific component is requested

  Output Parameters:
. offsetg - the global offset

  Level: intermediate

.seealso: DMNetworkGetLocalVecOffset(), DMGetGlobalVector(), DMNetworkGetComponent()
@*/
PetscErrorCode DMNetworkGetGlobalVecOffset(DM dm,PetscInt p,PetscInt compnum,PetscInt *offsetg)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetp,offsetd;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->plex->globalSection,p,&offsetp);CHKERRQ(ierr);
  if (offsetp < 0) offsetp = -(offsetp + 1); /* Convert to actual global offset for ghost vertex */

  if (compnum == ALL_COMPONENTS) {
    *offsetg = offsetp;
    PetscFunctionReturn(0);
  }
  ierr = PetscSectionGetOffset(network->DataSection,p,&offsetd);CHKERRQ(ierr);
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

.seealso: DMNetworkGetLocalVecOffset(), DMGetLocalVector()
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
  DMNetworkGetVertexOffset - Get the offset for accessing the variables associated with the given vertex from the local subvector

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the vertex point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: DMNetworkGetEdgeOffset(), DMGetLocalVector()
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
  DMNetworkAddComponent - Adds a network component and number of variables at the given point (vertex/edge)

  Not Collective

  Input Parameters:
+ dm - the DMNetwork
. p - the vertex/edge point
. componentkey - component key returned while registering the component; ignored if compvalue=NULL
. compvalue - pointer to the data structure for the component, or NULL if not required.
- nvar - number of variables for the component at the vertex/edge point

  Level: beginner

.seealso: DMNetworkGetComponent()
@*/
PetscErrorCode DMNetworkAddComponent(DM dm,PetscInt p,PetscInt componentkey,void* compvalue,PetscInt nvar)
{
  PetscErrorCode           ierr;
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponent       *component = &network->component[componentkey];
  DMNetworkComponentHeader header;
  DMNetworkComponentValue  cvalue;
  PetscBool                sharedv=PETSC_FALSE;
  PetscInt                 compnum;
  PetscInt                 *compsize,*compkey,*compoffset,*compnvar,*compoffsetvarrel;
  void*                    *compdata;

  PetscFunctionBegin;
  ierr = PetscSectionAddDof(network->DofSection,p,nvar);CHKERRQ(ierr);
  if (!compvalue) PetscFunctionReturn(0);

  ierr = DMNetworkIsSharedVertex(dm,p,&sharedv);CHKERRQ(ierr);
  if (sharedv) {
    PetscBool ghost;
    ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
    if (ghost) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Adding a component at a leaf(ghost) shared vertex is not supported");
  }

  header = &network->header[p];
  cvalue = &network->cvalue[p];
  if (header->ndata == header->maxcomps) {
    PetscInt additional_size;

    /* Reached limit so resize header component arrays */
    header->maxcomps += 2;

    /* Allocate arrays for component information and value */
    ierr = PetscCalloc5(header->maxcomps,&compsize,header->maxcomps,&compkey,header->maxcomps,&compoffset,header->maxcomps,&compnvar,header->maxcomps,&compoffsetvarrel);CHKERRQ(ierr);
    ierr = PetscMalloc1(header->maxcomps,&compdata);CHKERRQ(ierr);

    /* Recalculate header size */
    header->hsize = sizeof(struct _p_DMNetworkComponentHeader) + 5*header->maxcomps*sizeof(PetscInt);

    header->hsize /= sizeof(DMNetworkComponentGenericDataType);

    /* Copy over component info */
    ierr = PetscMemcpy(compsize,header->size,header->ndata*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(compkey,header->key,header->ndata*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(compoffset,header->offset,header->ndata*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(compnvar,header->nvar,header->ndata*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(compoffsetvarrel,header->offsetvarrel,header->ndata*sizeof(PetscInt));CHKERRQ(ierr);

    /* Copy over component data pointers */
    ierr = PetscMemcpy(compdata,cvalue->data,header->ndata*sizeof(void*));CHKERRQ(ierr);

    /* Free old arrays */
    ierr = PetscFree5(header->size,header->key,header->offset,header->nvar,header->offsetvarrel);CHKERRQ(ierr);
    ierr = PetscFree(cvalue->data);CHKERRQ(ierr);

    /* Update pointers */
    header->size = compsize;
    header->key  = compkey;
    header->offset = compoffset;
    header->nvar = compnvar;
    header->offsetvarrel = compoffsetvarrel;

    cvalue->data = compdata;

    /* Update DataSection Dofs */
    /* The dofs for datasection point p equals sizeof the header (i.e. header->hsize) + sizes of the components added at point p. With the resizing of the header, we need to update the dofs for point p. Hence, we add the extra size added for the header */
    additional_size = (5*(header->maxcomps - header->ndata)*sizeof(PetscInt))/sizeof(DMNetworkComponentGenericDataType);
    ierr = PetscSectionAddDof(network->DataSection,p,additional_size);CHKERRQ(ierr);
  }
  header = &network->header[p];
  cvalue = &network->cvalue[p];

  compnum = header->ndata;

  header->size[compnum] = component->size;
  ierr = PetscSectionAddDof(network->DataSection,p,component->size);CHKERRQ(ierr);
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

.seealso: DMNetworkAddComponent(), DMNetworkGetNumComponents()
@*/
PetscErrorCode DMNetworkGetComponent(DM dm,PetscInt p,PetscInt compnum,PetscInt *compkey,void **component,PetscInt *nvar)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offset = 0;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  if (compnum == ALL_COMPONENTS) {
    ierr = PetscSectionGetDof(network->DofSection,p,nvar);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscSectionGetOffset(network->DataSection,p,&offset);CHKERRQ(ierr);
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
  PetscErrorCode           ierr;
  DM_Network               *network = (DM_Network*)dm->data;
  PetscInt                 arr_size,p,offset,offsetp,ncomp,i,*headerarr;
  DMNetworkComponentHeader header;
  DMNetworkComponentValue  cvalue;
  DMNetworkComponentHeader headerinfo;
  DMNetworkComponentGenericDataType *componentdataarray;

  PetscFunctionBegin;
  ierr = PetscSectionSetUp(network->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(network->DataSection,&arr_size);CHKERRQ(ierr);
  /* arr_size+1 fixes pipeline test of opensolaris-misc for src/dm/tests/ex10.c -- Do not know why */
  ierr = PetscCalloc1(arr_size+1,&network->componentdataarray);CHKERRQ(ierr);
  componentdataarray = network->componentdataarray;
  for (p = network->pStart; p < network->pEnd; p++) {
    ierr = PetscSectionGetOffset(network->DataSection,p,&offsetp);CHKERRQ(ierr);
    /* Copy header */
    header = &network->header[p];
    headerinfo = (DMNetworkComponentHeader)(componentdataarray+offsetp);
    ierr = PetscMemcpy(headerinfo,header,sizeof(struct _p_DMNetworkComponentHeader));CHKERRQ(ierr);
    headerarr = (PetscInt*)(headerinfo+1);
    ierr = PetscMemcpy(headerarr,header->size,header->maxcomps*sizeof(PetscInt));CHKERRQ(ierr);
    headerarr += header->maxcomps;
    ierr = PetscMemcpy(headerarr,header->key,header->maxcomps*sizeof(PetscInt));CHKERRQ(ierr);
    headerarr += header->maxcomps;
    ierr = PetscMemcpy(headerarr,header->offset,header->maxcomps*sizeof(PetscInt));CHKERRQ(ierr);
    headerarr += header->maxcomps;
    ierr = PetscMemcpy(headerarr,header->nvar,header->maxcomps*sizeof(PetscInt));CHKERRQ(ierr);
    headerarr += header->maxcomps;
    ierr = PetscMemcpy(headerarr,header->offsetvarrel,header->maxcomps*sizeof(PetscInt));CHKERRQ(ierr);

    /* Copy data */
    cvalue = &network->cvalue[p];
    ncomp  = header->ndata;

    for (i = 0; i < ncomp; i++) {
      offset = offsetp + header->hsize + header->offset[i];
      ierr = PetscMemcpy(componentdataarray+offset,cvalue->data[i],header->size[i]*sizeof(DMNetworkComponentGenericDataType));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Sets up the section for dofs. This routine is called during DMSetUp() */
static PetscErrorCode DMNetworkVariablesSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  MPI_Comm       comm;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  if (size > 1) { /* Sync nvar at shared vertices for all processes */
    PetscSF           sf = network->plex->sf;
    PetscInt          *local_nvar, *remote_nvar,nroots,nleaves,p=-1,i,nsv;
    const PetscInt    *svtx;
    PetscBool         ghost;
    /*
     PetscMPIInt       rank;
     const PetscInt    *ilocal;
     const PetscSFNode *iremote;
     ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
     ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    */
    ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscCalloc2(nroots,&local_nvar,nroots,&remote_nvar);CHKERRQ(ierr);

    /* Leaves copy user's nvar to local_nvar */
    ierr = DMNetworkGetSharedVertices(dm,&nsv,&svtx);CHKERRQ(ierr);
    for (i=0; i<nsv; i++) {
      p = svtx[i];
      ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
      if (!ghost) continue;
      ierr = PetscSectionGetDof(network->DofSection,p,&local_nvar[p]);CHKERRQ(ierr);
      /* printf("[%d] Before SFReduce: leaf local_nvar[%d] = %d\n",rank,p,local_nvar[p]); */
    }

    /* Leaves add local_nvar to root remote_nvar */
    ierr = PetscSFReduceBegin(sf, MPIU_INT, local_nvar, remote_nvar, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf, MPIU_INT, local_nvar, remote_nvar, MPI_SUM);CHKERRQ(ierr);
    /* printf("[%d] remote_nvar[%d] = %d\n",rank,p,remote_nvar[p]); */

    /* Update roots' local_nvar */
    for (i=0; i<nsv; i++) {
      p = svtx[i];
      ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
      if (ghost) continue;

      ierr = PetscSectionAddDof(network->DofSection,p,remote_nvar[p]);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(network->DofSection,p,&local_nvar[p]);CHKERRQ(ierr);
      /* printf("[%d]  After SFReduce: root local_nvar[%d] = %d\n",rank,p,local_nvar[p]); */
    }

    /* Roots Bcast nvar to leaves */
    ierr = PetscSFBcastBegin(sf, MPIU_INT, local_nvar, remote_nvar,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, local_nvar, remote_nvar,MPI_REPLACE);CHKERRQ(ierr);

    /* Leaves reset receved/remote nvar to dm */
    for (i=0; i<nsv; i++) {
      ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
      if (!ghost) continue;
      p = svtx[i];
      /* printf("[%d] leaf reset nvar %d at p= %d \n",rank,remote_nvar[p],p); */
      /* DMNetworkSetNumVariables(dm,p,remote_nvar[p]) */
      ierr = PetscSectionSetDof(network->DofSection,p,remote_nvar[p]);CHKERRQ(ierr);
    }

    ierr = PetscFree2(local_nvar,remote_nvar);CHKERRQ(ierr);
  }

  ierr = PetscSectionSetUp(network->DofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Get a subsection from a range of points */
static PetscErrorCode DMNetworkGetSubSection_private(PetscSection main,PetscInt pstart,PetscInt pend,PetscSection *subsection)
{
  PetscErrorCode ierr;
  PetscInt       i, nvar;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)main), subsection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*subsection, 0, pend - pstart);CHKERRQ(ierr);
  for (i = pstart; i < pend; i++) {
    ierr = PetscSectionGetDof(main,i,&nvar);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(*subsection, i - pstart, nvar);CHKERRQ(ierr);
  }

  ierr = PetscSectionSetUp(*subsection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create a submap of points with a GlobalToLocal structure */
static PetscErrorCode DMNetworkSetSubMap_private(PetscInt pstart, PetscInt pend, ISLocalToGlobalMapping *map)
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
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    size;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);

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

/*
   Add all subnetid for the input vertex v in this process to the btable
   vertex_subnetid = supportingedge_subnetid
*/
PETSC_STATIC_INLINE PetscErrorCode SetSubnetIdLookupBT(DM dm,PetscInt v,PetscInt Nsubnet,PetscBT btable)
{
  PetscErrorCode ierr;
  PetscInt       e,nedges,offset;
  const PetscInt *edges;
  DM_Network     *newDMnetwork = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = DMNetworkGetSupportingEdges(dm,v,&nedges,&edges);CHKERRQ(ierr);
  ierr = PetscBTMemzero(Nsubnet,btable);CHKERRQ(ierr);
  for (e=0; e<nedges; e++) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,edges[e],&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);
    ierr = PetscBTSet(btable,header->subnetid);CHKERRQ(ierr);
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

.seealso: DMNetworkCreate()
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
  PetscInt       j,e,v,offset,*subnetvtx,*subnetedge,Nsubnet,gidx,svtx_idx,nv;
  PetscInt       to_net,from_net,*svto;
  PetscBT        btable;
  PetscPartitioner         part;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)*dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  if (size == 1) PetscFunctionReturn(0);

  if (overlap) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"overlap %D != 0 is not supported yet",overlap);

  /* This routine moves the component data to the appropriate processors. It makes use of the DataSection and the componentdataarray to move the component data to appropriate processors and returns a new DataSection and new componentdataarray. */
  ierr = DMNetworkCreate(PetscObjectComm((PetscObject)*dm),&newDM);CHKERRQ(ierr);
  newDMnetwork = (DM_Network*)newDM->data;
  newDMnetwork->max_comps_registered = oldDMnetwork->max_comps_registered;
  ierr = PetscMalloc1(newDMnetwork->max_comps_registered,&newDMnetwork->component);CHKERRQ(ierr);

  /* Enable runtime options for petscpartitioner */
  ierr = DMPlexGetPartitioner(oldDMnetwork->plex,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

  /* Distribute plex dm */
  ierr = DMPlexDistribute(oldDMnetwork->plex,overlap,&pointsf,&newDMnetwork->plex);CHKERRQ(ierr);

  /* Distribute dof section */
  ierr = PetscSectionCreate(comm,&newDMnetwork->DofSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointsf,oldDMnetwork->DofSection,NULL,newDMnetwork->DofSection);CHKERRQ(ierr);

  /* Distribute data and associated section */
  ierr = PetscSectionCreate(comm,&newDMnetwork->DataSection);CHKERRQ(ierr);
  ierr = DMPlexDistributeData(newDMnetwork->plex,pointsf,oldDMnetwork->DataSection,MPIU_INT,(void*)oldDMnetwork->componentdataarray,newDMnetwork->DataSection,(void**)&newDMnetwork->componentdataarray);CHKERRQ(ierr);

  ierr = PetscSectionGetChart(newDMnetwork->DataSection,&newDMnetwork->pStart,&newDMnetwork->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMnetwork->plex,0, &newDMnetwork->eStart,&newDMnetwork->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMnetwork->plex,1,&newDMnetwork->vStart,&newDMnetwork->vEnd);CHKERRQ(ierr);
  newDMnetwork->nEdges    = newDMnetwork->eEnd - newDMnetwork->eStart;
  newDMnetwork->nVertices = newDMnetwork->vEnd - newDMnetwork->vStart;
  newDMnetwork->NVertices = oldDMnetwork->NVertices;
  newDMnetwork->NEdges    = oldDMnetwork->NEdges;
  newDMnetwork->svtable   = oldDMnetwork->svtable; /* global table! */
  oldDMnetwork->svtable   = NULL;

  /* Set Dof section as the section for dm */
  ierr = DMSetLocalSection(newDMnetwork->plex,newDMnetwork->DofSection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(newDMnetwork->plex,&newDMnetwork->GlobalDofSection);CHKERRQ(ierr);

  /* Setup subnetwork info in the newDM */
  newDMnetwork->Nsubnet = oldDMnetwork->Nsubnet;
  newDMnetwork->Nsvtx   = oldDMnetwork->Nsvtx;
  oldDMnetwork->Nsvtx   = 0;
  newDMnetwork->svtx    = oldDMnetwork->svtx; /* global vertices! */
  oldDMnetwork->svtx    = NULL;
  ierr = PetscCalloc1(newDMnetwork->Nsubnet,&newDMnetwork->subnet);CHKERRQ(ierr);

  /* Copy over the global number of vertices and edges in each subnetwork.
     Note: these are calculated in DMNetworkLayoutSetUp()
  */
  Nsubnet = newDMnetwork->Nsubnet;
  for (j = 0; j < Nsubnet; j++) {
    newDMnetwork->subnet[j].Nvtx  = oldDMnetwork->subnet[j].Nvtx;
    newDMnetwork->subnet[j].Nedge = oldDMnetwork->subnet[j].Nedge;
  }

  /* Count local nedges for subnetworks */
  for (e = newDMnetwork->eStart; e < newDMnetwork->eEnd; e++) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);
    /* Update pointers */
    header->size          = (PetscInt*)(header + 1);
    header->key           = header->size   + header->maxcomps;
    header->offset        = header->key    + header->maxcomps;
    header->nvar          = header->offset + header->maxcomps;
    header->offsetvarrel  = header->nvar   + header->maxcomps;

    newDMnetwork->subnet[header->subnetid].nedge++;
  }

  /* Count local nvtx for subnetworks */
  ierr = PetscBTCreate(Nsubnet,&btable);CHKERRQ(ierr);
  for (v = newDMnetwork->vStart; v < newDMnetwork->vEnd; v++) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    /* Update pointers */
    header->size          = (PetscInt*)(header + 1);
    header->key           = header->size   + header->maxcomps;
    header->offset        = header->key    + header->maxcomps;
    header->nvar          = header->offset + header->maxcomps;
    header->offsetvarrel  = header->nvar   + header->maxcomps;

    /* shared vertices: use gidx=header->index to check if v is a shared vertex */
    gidx = header->index;
    ierr = PetscTableFind(newDMnetwork->svtable,gidx+1,&svtx_idx);CHKERRQ(ierr);
    svtx_idx--;

    if (svtx_idx < 0) { /* not a shared vertex */
      newDMnetwork->subnet[header->subnetid].nvtx++;
    } else { /* a shared vertex belongs to more than one subnetworks, it is being counted by multiple subnets */
      ierr = SetSubnetIdLookupBT(newDM,v,Nsubnet,btable);CHKERRQ(ierr);

      from_net = newDMnetwork->svtx[svtx_idx].sv[0];
      if (PetscBTLookup(btable,from_net)) newDMnetwork->subnet[from_net].nvtx++; /* sv is on from_net */

      for (j=1; j<newDMnetwork->svtx[svtx_idx].n; j++) {
        svto   = newDMnetwork->svtx[svtx_idx].sv + 2*j;
        to_net = svto[0];
        if (PetscBTLookup(btable,to_net)) newDMnetwork->subnet[to_net].nvtx++; /* sv is on to_net */
      }
    }
  }

  /* Get total local nvtx for subnetworks */
  nv = 0;
  for (j=0; j<Nsubnet; j++) nv += newDMnetwork->subnet[j].nvtx;
  nv += newDMnetwork->Nsvtx;

  /* Now create the vertices and edge arrays for the subnetworks */
  ierr = PetscCalloc2(newDMnetwork->nEdges,&subnetedge,nv,&subnetvtx);CHKERRQ(ierr); /* Maps local vertex to local subnetwork's vertex */
  newDMnetwork->subnetedge = subnetedge;
  newDMnetwork->subnetvtx  = subnetvtx;
  for (j=0; j < newDMnetwork->Nsubnet; j++) {
    newDMnetwork->subnet[j].edges = subnetedge;
    subnetedge                   += newDMnetwork->subnet[j].nedge;

    newDMnetwork->subnet[j].vertices = subnetvtx;
    subnetvtx                       += newDMnetwork->subnet[j].nvtx;

    /* Temporarily setting nvtx and nedge to 0 so we can use them as counters in the below for loop. These get updated when the vertices and edges are added. */
    newDMnetwork->subnet[j].nvtx = newDMnetwork->subnet[j].nedge = 0;
  }
  newDMnetwork->svertices = subnetvtx;

  /* Set the edges and vertices in each subnetwork */
  for (e = newDMnetwork->eStart; e < newDMnetwork->eEnd; e++) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,e,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);
    newDMnetwork->subnet[header->subnetid].edges[newDMnetwork->subnet[header->subnetid].nedge++] = e;
  }

  nv = 0;
  for (v = newDMnetwork->vStart; v < newDMnetwork->vEnd; v++) {
    ierr = PetscSectionGetOffset(newDMnetwork->DataSection,v,&offset);CHKERRQ(ierr);
    header = (DMNetworkComponentHeader)(newDMnetwork->componentdataarray+offset);CHKERRQ(ierr);

    /* coupling vertices: use gidx = header->index to check if v is a coupling vertex */
    ierr = PetscTableFind(newDMnetwork->svtable,header->index+1,&svtx_idx);CHKERRQ(ierr);
    svtx_idx--;
    if (svtx_idx < 0) {
      newDMnetwork->subnet[header->subnetid].vertices[newDMnetwork->subnet[header->subnetid].nvtx++] = v;
    } else { /* a shared vertex */
      newDMnetwork->svertices[nv++] = v;

      /* add all subnetid for this shared vertex in this process to btable */
      ierr = SetSubnetIdLookupBT(newDM,v,Nsubnet,btable);CHKERRQ(ierr);

      from_net = newDMnetwork->svtx[svtx_idx].sv[0];
      if (PetscBTLookup(btable,from_net))
        newDMnetwork->subnet[from_net].vertices[newDMnetwork->subnet[from_net].nvtx++] = v;

      for (j=1; j<newDMnetwork->svtx[svtx_idx].n; j++) {
        svto   = newDMnetwork->svtx[svtx_idx].sv + 2*j;
        to_net = svto[0];
        if (PetscBTLookup(btable,to_net))
          newDMnetwork->subnet[to_net].vertices[newDMnetwork->subnet[to_net].nvtx++] = v;
      }
    }
  }
  newDMnetwork->nsvtx = nv;   /* num of local shared vertices */

  newDM->setupcalled = (*dm)->setupcalled;
  newDMnetwork->distributecalled = PETSC_TRUE;

  /* Free spaces */
  ierr = PetscSFDestroy(&pointsf);CHKERRQ(ierr);
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&btable);CHKERRQ(ierr);

  /* View distributed dmnetwork */
  ierr = DMViewFromOptions(newDM,NULL,"-dmnetwork_view_distributed");CHKERRQ(ierr);

  *dm  = newDM;
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
  PetscErrorCode        ierr;
  PetscInt              nroots, nleaves, *ilocal_sub;
  PetscInt              i, *ilocal_map, nroots_sub, nleaves_sub = 0;
  PetscInt              *local_points, *remote_points;
  PetscSFNode           *iremote_sub;
  const PetscInt        *ilocal;
  const PetscSFNode     *iremote;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(mainsf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);

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
  ierr = PetscSFBcastBegin(mainsf, MPIU_INT, local_points, remote_points,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(mainsf, MPIU_INT, local_points, remote_points,MPI_REPLACE);CHKERRQ(ierr);
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
+ dm - the DMNetwork object
- p  - the vertex point

  Output Parameters:
+ nedges - number of edges connected to this vertex point
- edges  - list of edge points

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMNetworkCreate(), DMNetworkGetConnectedVertices()
@*/
PetscErrorCode DMNetworkGetSupportingEdges(DM dm,PetscInt vertex,PetscInt *nedges,const PetscInt *edges[])
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;

  PetscFunctionBegin;
  ierr = DMPlexGetSupportSize(network->plex,vertex,nedges);CHKERRQ(ierr);
  if (edges) {ierr = DMPlexGetSupport(network->plex,vertex,edges);CHKERRQ(ierr);}
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

.seealso: DMNetworkCreate(), DMNetworkGetSupportingEdges()
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
  DMNetworkIsSharedVertex - Returns TRUE if the vertex is shared by subnetworks

  Not Collective

  Input Parameters:
+ dm - the DMNetwork object
- p - the vertex point

  Output Parameter:
. flag - TRUE if the vertex is shared by subnetworks

  Level: beginner

.seealso: DMNetworkAddSharedVertices(), DMNetworkIsGhostVertex()
@*/
PetscErrorCode DMNetworkIsSharedVertex(DM dm,PetscInt p,PetscBool *flag)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  *flag = PETSC_FALSE;

  if (dm->setupcalled) { /* DMNetworkGetGlobalVertexIndex() requires DMSetUp() be called */
    DM_Network     *network = (DM_Network*)dm->data;
    PetscInt       gidx;
    ierr = DMNetworkGetGlobalVertexIndex(dm,p,&gidx);CHKERRQ(ierr);
    ierr = PetscTableFind(network->svtable,gidx+1,&i);CHKERRQ(ierr);
    if (i) *flag = PETSC_TRUE;
  } else { /* would be removed? */
    PetscInt       nv;
    const PetscInt *vtx;
    ierr = DMNetworkGetSharedVertices(dm,&nv,&vtx);CHKERRQ(ierr);
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

.seealso: DMNetworkGetConnectedVertices(), DMNetworkGetVertexRange(), DMNetworkIsSharedVertex()
@*/
PetscErrorCode DMNetworkIsGhostVertex(DM dm,PetscInt p,PetscBool *isghost)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
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

  /* View dmnetwork */
  ierr = DMViewFromOptions(dm,NULL,"-dmnetwork_view");CHKERRQ(ierr);
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
+ dm - the DMNetwork object
. p - the edge point
- J - array (size = 3) of Jacobian submatrices for this edge point:
        J[0]: this edge
        J[1] and J[2]: connected vertices, obtained by calling DMNetworkGetConnectedVertices()

  Level: advanced

.seealso: DMNetworkVertexSetMatrix()
@*/
PetscErrorCode DMNetworkEdgeSetMatrix(DM dm,PetscInt p,Mat J[])
{
  DM_Network *network=(DM_Network*)dm->data;

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
+ dm - The DMNetwork object
. p - the vertex point
- J - array of Jacobian (size = 2*(num of supporting edges) + 1) submatrices for this vertex point:
        J[0]:       this vertex
        J[1+2*i]:   i-th supporting edge
        J[1+2*i+1]: i-th connected vertex

  Level: advanced

.seealso: DMNetworkEdgeSetMatrix()
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
  PetscInt       eDof,vDof;
  Mat            j11,j12,j21,j22,bA[2][2];
  MPI_Comm       comm;
  ISLocalToGlobalMapping eISMap,vISMap;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);

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
    ierr = DMNetworkGetGlobalVecOffset(dm,e,ALL_COMPONENTS,&rstart);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(network->DofSection,e,&nrows);CHKERRQ(ierr);
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set preallocation for connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,e,&cone);CHKERRQ(ierr);
      for (v=0; v<2; v++) {
        ierr = PetscSectionGetDof(network->DofSection,cone[v],&ncols);CHKERRQ(ierr);

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
    ierr = DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&rstart);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(network->DofSection,v,&nrows);CHKERRQ(ierr);
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
      ierr = DMNetworkGetGlobalVecOffset(dm,edges[e],ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(network->DofSection,edges[e],&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      ierr = MatSetPreallocationblock_private(Juser,nrows,rows_v,ncols,ghost,vd_nz,vo_nz);CHKERRQ(ierr);

      /* Connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,edges[e],&cone);CHKERRQ(ierr);
      vc = (v == cone[0]) ? cone[1]:cone[0];
      ierr = DMNetworkIsGhostVertex(dm,vc,&ghost_vc);CHKERRQ(ierr);

      ierr = PetscSectionGetDof(network->DofSection,vc,&ncols);CHKERRQ(ierr);

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
      ierr = DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
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
    ierr = DMNetworkGetGlobalVecOffset(dm,e,ALL_COMPONENTS,&rstart);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(network->DofSection,e,&nrows);CHKERRQ(ierr);
    if (nrows) {
      for (j=0; j<nrows; j++) rows[j] = j + rstart;

      /* Set matrix entries for connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,e,&cone);CHKERRQ(ierr);
      for (v=0; v<2; v++) {
        ierr = DMNetworkGetGlobalVecOffset(dm,cone[v],ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(network->DofSection,cone[v],&ncols);CHKERRQ(ierr);

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
    ierr = DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&rstart);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(network->DofSection,v,&nrows);CHKERRQ(ierr);
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
      ierr = DMNetworkGetGlobalVecOffset(dm,edges[e],ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(network->DofSection,edges[e],&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+1]; /* Jacobian(v,e) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J);CHKERRQ(ierr);

      /* Connected vertices */
      ierr = DMNetworkGetConnectedVertices(dm,edges[e],&cone);CHKERRQ(ierr);
      vc = (v == cone[0]) ? cone[1]:cone[0];

      ierr = DMNetworkGetGlobalVecOffset(dm,vc,ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(network->DofSection,vc,&ncols);CHKERRQ(ierr);

      if (network->Jv) {
        Juser = network->Jv[vptr[v-vStart]+2*e+2]; /* Jacobian(v,vc) */
      } else Juser = NULL;
      ierr = MatSetblock_private(Juser,nrows,rows_v,ncols,cstart,J);CHKERRQ(ierr);
    }

    /* Set matrix entries for vertex self */
    if (!ghost) {
      ierr = DMNetworkGetGlobalVecOffset(dm,v,ALL_COMPONENTS,&cstart);CHKERRQ(ierr);
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
  PetscInt       j,np;

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

  for (j=0; j<network->Nsvtx; j++) {
    ierr = PetscFree(network->svtx[j].sv);CHKERRQ(ierr);
  }
  if (network->svtx) {ierr = PetscFree(network->svtx);CHKERRQ(ierr);}
  ierr = PetscFree2(network->subnetedge,network->subnetvtx);CHKERRQ(ierr);

  ierr = PetscTableDestroy(&network->svtable);CHKERRQ(ierr);
  ierr = PetscFree(network->subnet);CHKERRQ(ierr);
  ierr = PetscFree(network->component);CHKERRQ(ierr);
  ierr = PetscFree(network->componentdataarray);CHKERRQ(ierr);

  if (network->header) {
    np = network->pEnd - network->pStart;
    for (j=0; j < np; j++) {
      ierr = PetscFree5(network->header[j].size,network->header[j].key,network->header[j].offset,network->header[j].nvar,network->header[j].offsetvarrel);CHKERRQ(ierr);
      ierr = PetscFree(network->cvalue[j].data);CHKERRQ(ierr);
    }
    ierr = PetscFree2(network->header,network->cvalue);CHKERRQ(ierr);
  }
  ierr = PetscFree(network);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Network(DM dm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() first");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  PetscValidHeaderSpecific(dm,DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    const PetscInt *cone,*vtx,*edges;
    PetscInt       vfrom,vto,i,j,nv,ne,ncv,p,nsubnet;
    DM_Network     *network = (DM_Network*)dm->data;

    nsubnet = network->Nsubnet; /* num of subnetworks */
    if (rank == 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  NSubnets: %D; NEdges: %D; NVertices: %D; NSharedVertices: %D.\n",nsubnet,network->NEdges,network->NVertices,network->Nsvtx);CHKERRQ(ierr);
    }

    ierr = DMNetworkGetSharedVertices(dm,&ncv,&vtx);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "  [%d] nEdges: %D; nVertices: %D; nSharedVertices: %D\n",rank,network->nEdges,network->nVertices,ncv);CHKERRQ(ierr);

    for (i=0; i<nsubnet; i++) {
      ierr = DMNetworkGetSubnetwork(dm,i,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
      if (ne) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "     Subnet %D: nEdges %D, nVertices(include shared vertices) %D\n",i,ne,nv);CHKERRQ(ierr);
        for (j=0; j<ne; j++) {
          p = edges[j];
          ierr = DMNetworkGetConnectedVertices(dm,p,&cone);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[0],&vfrom);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(dm,cone[1],&vto);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalEdgeIndex(dm,edges[j],&p);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "       edge %D: %D ----> %D\n",p,vfrom,vto);CHKERRQ(ierr);
        }
      }
    }

    /* Shared vertices */
    ierr = DMNetworkGetSharedVertices(dm,&ncv,&vtx);CHKERRQ(ierr);
    if (ncv) {
      SVtx       *svtx = network->svtx;
      PetscInt    gidx,svtx_idx,nvto,vfrom_net,vfrom_idx,*svto;
      PetscBool   ghost;
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "     SharedVertices:\n");CHKERRQ(ierr);
      for (i=0; i<ncv; i++) {
        ierr = DMNetworkIsGhostVertex(dm,vtx[i],&ghost);CHKERRQ(ierr);
        if (ghost) continue;

        ierr = DMNetworkGetGlobalVertexIndex(dm,vtx[i],&gidx);CHKERRQ(ierr);
        ierr = PetscTableFind(network->svtable,gidx+1,&svtx_idx);CHKERRQ(ierr);
        svtx_idx--;
        nvto = svtx[svtx_idx].n;

        vfrom_net = svtx[svtx_idx].sv[0];
        vfrom_idx = svtx[svtx_idx].sv[1];
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "       svtx %D: global index %D, subnet[%D].%D ---->\n",i,gidx,vfrom_net,vfrom_idx);CHKERRQ(ierr);
        for (j=1; j<nvto; j++) {
          svto = svtx[svtx_idx].sv + 2*j;
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "                                           ----> subnet[%D].%D\n",svto[0],svto[1]);CHKERRQ(ierr);
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
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

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

  ierr = PetscMalloc3(size+1,&vrange,size,&displs,size,&recvcounts);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(vsf,&nroots,&nleaves,NULL,&iremote);CHKERRQ(ierr);

  for (i=0; i<size; i++) { displs[i] = i; recvcounts[i] = 1;}

  i         = nroots - nleaves; /* local number of vertices, excluding ghosts */
  vrange[0] = 0;
  ierr = MPI_Allgatherv(&i,1,MPIU_INT,vrange+1,recvcounts,displs,MPIU_INT,comm);CHKERRMPI(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode DMISAddSize_private(DM_Network *network,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *nidx)
{
  PetscErrorCode           ierr;
  PetscInt                 i,j,ncomps,nvar,key,offset=0;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->DataSection,p,&offset);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode DMISComputeIdx_private(DM dm,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],PetscInt *ii,PetscInt *idx)
{
  PetscErrorCode           ierr;
  PetscInt                 i,j,ncomps,nvar,key,offsetg,k,k1,offset=0;
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->DataSection,p,&offset);CHKERRQ(ierr);
  ncomps = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  for (i=0; i<ncomps; i++) {
    key  = header->key[i];
    nvar = header->nvar[i];
    for (j=0; j<numkeys; j++) {
      if (key != keys[j]) continue;

      ierr = DMNetworkGetGlobalVecOffset(dm,p,i,&offsetg);CHKERRQ(ierr);
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

.seealso: DMNetworkCreate(), ISCreateGeneral(), DMNetworkCreateLocalIS()
@*/
PetscErrorCode DMNetworkCreateIS(DM dm,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],IS *is)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,p,estart,eend,vstart,vend,nidx,*idx;
  PetscBool      ghost;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);

  /* Check input parameters */
  for (i=0; i<numkeys; i++) {
    if (!blocksize || blocksize[i] == -1) continue;
    if (nselectedvar[i] > blocksize[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"number of selectedvariables %D cannot be larger than blocksize %D",nselectedvar[i],blocksize[i]);
  }

  ierr = DMNetworkGetEdgeRange(dm,&estart,&eend);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(dm,&vstart,&vend);CHKERRQ(ierr);

  /* Get local number of idx */
  nidx = 0;
  for (p=estart; p<eend; p++) {
    ierr = DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx);CHKERRQ(ierr);
  }
  for (p=vstart; p<vend; p++) {
    ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
    if (ghost) continue;
    ierr = DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx);CHKERRQ(ierr);
  }

  /* Compute idx */
  ierr = PetscMalloc1(nidx,&idx);CHKERRQ(ierr);
  i = 0;
  for (p=estart; p<eend; p++) {
    ierr = DMISComputeIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx);CHKERRQ(ierr);
  }
  for (p=vstart; p<vend; p++) {
    ierr = DMNetworkIsGhostVertex(dm,p,&ghost);CHKERRQ(ierr);
    if (ghost) continue;
    ierr = DMISComputeIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx);CHKERRQ(ierr);
  }

  /* Create is */
  ierr = ISCreateGeneral(comm,nidx,idx,PETSC_COPY_VALUES,is);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMISComputeLocalIdx_private(DM dm,PetscInt p,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],PetscInt *ii,PetscInt *idx)
{
  PetscErrorCode           ierr;
  PetscInt                 i,j,ncomps,nvar,key,offsetl,k,k1,offset=0;
  DM_Network               *network = (DM_Network*)dm->data;
  DMNetworkComponentHeader header;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(network->DataSection,p,&offset);CHKERRQ(ierr);
  ncomps = ((DMNetworkComponentHeader)(network->componentdataarray+offset))->ndata;
  header = (DMNetworkComponentHeader)(network->componentdataarray+offset);

  for (i=0; i<ncomps; i++) {
    key  = header->key[i];
    nvar = header->nvar[i];
    for (j=0; j<numkeys; j++) {
      if (key != keys[j]) continue;

      ierr = DMNetworkGetLocalVecOffset(dm,p,i,&offsetl);CHKERRQ(ierr);
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

.seealso: DMNetworkCreate(), DMNetworkCreateIS, ISCreateGeneral()
@*/
PetscErrorCode DMNetworkCreateLocalIS(DM dm,PetscInt numkeys,PetscInt keys[],PetscInt blocksize[],PetscInt nselectedvar[],PetscInt *selectedvar[],IS *is)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*)dm->data;
  PetscInt       i,p,pstart,pend,nidx,*idx;

  PetscFunctionBegin;
  /* Check input parameters */
  for (i=0; i<numkeys; i++) {
    if (!blocksize || blocksize[i] == -1) continue;
    if (nselectedvar[i] > blocksize[i]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"number of selectedvariables %D cannot be larger than blocksize %D",nselectedvar[i],blocksize[i]);
  }

  pstart = network->pStart;
  pend   = network->pEnd;

  /* Get local number of idx */
  nidx = 0;
  for (p=pstart; p<pend; p++) {
    ierr = DMISAddSize_private(network,p,numkeys,keys,blocksize,nselectedvar,&nidx);CHKERRQ(ierr);
  }

  /* Compute local idx */
  ierr = PetscMalloc1(nidx,&idx);CHKERRQ(ierr);
  i = 0;
  for (p=pstart; p<pend; p++) {
    ierr = DMISComputeLocalIdx_private(dm,p,numkeys,keys,blocksize,nselectedvar,selectedvar,&i,idx);CHKERRQ(ierr);
  }

  /* Create is */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx,PETSC_COPY_VALUES,is);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
