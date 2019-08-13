#if !defined(__SFBASIC_H)
#define __SFBASIC_H

#include <../src/vec/is/sf/impls/basic/sfpack.h>

typedef enum {PETSCSF_LEAF2ROOT_REDUCE=0, PETSCSF_ROOT2LEAF_BCAST=1} PetscSFDirection;

typedef struct _n_PetscSFPack_Basic *PetscSFPack_Basic;


/* Why do we want to double MPI requests?
   Note each PetscSFPack link supports either leaf2root or root2leaf communication, but not simultaneously both.
   We use persistent MPI requests in SFBasic. By doubling the requests, the communications in both direction can
   shared rootbuf and leafbuf. SFNeighbor etc do not need this since MPI does not support persistent requests for
   collectives yet. But once MPI adds this feature, SFNeighbor etc can also benefit from this design.
 */
#define SPPACKBASICHEADER \
  SFPACKHEADER;                                                                                                                    \
  PetscMPIInt   half;           /* Number of MPI_Requests used for either leaf2root or root2leaf communication */                  \
  MPI_Request   *requests       /* [2*half] requests arranged in this order: leaf2root root/leaf reqs, root2leaf root/leaf reqs */

struct _n_PetscSFPack_Basic {
  SPPACKBASICHEADER;
  PetscBool     initialized[2]; /* Is the communcation pattern in each direction initialized? [0] for leaf2root, [1] for root2leaf */
};

#define SFBASICHEADER \
  PetscMPIInt      niranks;         /* Number of incoming ranks (ranks accessing my roots) */                                      \
  PetscMPIInt      ndiranks;        /* Number of incoming ranks (ranks accessing my roots) in distinguished set */                 \
  PetscMPIInt      *iranks;         /* Array of ranks that reference my roots */                                                   \
  PetscInt         itotal;          /* Total number of graph edges referencing my roots */                                         \
  PetscInt         *ioffset;        /* Array of length niranks+1 holding offset in irootloc[] for each rank */                     \
  PetscInt         *irootloc;       /* Incoming roots referenced by ranks starting at ioffset[rank] */                             \
  PetscSFPackOpt   rootpackopt;     /* Optimization plans to (un)pack roots based on patterns in irootloc[]. NULL for no plans */  \
  PetscSFPackOpt   selfrootpackopt; /* Optimization plans to (un)pack roots connected to local leaves */                           \
  PetscSFPack      avail;           /* One or more entries per MPI Datatype, lazily constructed */                                 \
  PetscSFPack      inuse            /* Buffers being used for transactions that have not yet completed */

typedef struct {
  SFBASICHEADER;
} PetscSF_Basic;

PETSC_STATIC_INLINE PetscErrorCode PetscSFGetRootInfo_Basic(PetscSF sf,PetscInt *nrootranks,PetscInt *ndrootranks,const PetscMPIInt **rootranks,const PetscInt **rootoffset,const PetscInt **rootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (nrootranks)  *nrootranks  = bas->niranks;
  if (ndrootranks) *ndrootranks = bas->ndiranks;
  if (rootranks)   *rootranks   = bas->iranks;
  if (rootoffset)  *rootoffset  = bas->ioffset;
  if (rootloc)     *rootloc     = bas->irootloc;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFGetLeafInfo_Basic(PetscSF sf,PetscInt *nleafranks,PetscInt *ndleafranks,const PetscMPIInt **leafranks,const PetscInt **leafoffset,const PetscInt **leafloc,const PetscInt **leafrremote)
{
  PetscFunctionBegin;
  if (nleafranks)  *nleafranks  = sf->nranks;
  if (ndleafranks) *ndleafranks = sf->ndranks;
  if (leafranks)   *leafranks   = sf->ranks;
  if (leafoffset)  *leafoffset  = sf->roffset;
  if (leafloc)     *leafloc     = sf->rmine;
  if (leafrremote) *leafrremote = sf->rremote;
  PetscFunctionReturn(0);
}

/* Get root locations either on Host (CPU) or Device (GPU) */
PETSC_STATIC_INLINE PetscErrorCode PetscSFGetRootIndicesAtPlace_Basic(PetscSF sf,PetscBool isdevice, const PetscInt **rootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscFunctionBegin;
  if (rootloc)     *rootloc     = bas->irootloc;
  PetscFunctionReturn(0);
}

/* Get leaf locations either on Host (CPU) or Device (GPU) */
PETSC_STATIC_INLINE PetscErrorCode PetscSFGetLeafIndicesAtPlace_Basic(PetscSF sf,PetscBool isdevice, const PetscInt **leafloc)
{
  PetscFunctionBegin;
  if (leafloc)     *leafloc     = sf->rmine;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt       count;  /* Number of entries to pack, unpack etc. */
  PetscInt       offset; /* Offset of the first entry */
  PetscSFPackOpt opt;    /* Pack optimizations */
  char           *buf;   /* The contiguous buffer where we pack to or unpack from */
} PackInfo;

/* Utility routine to pack selected entries of rootdata into root buffer */
PETSC_STATIC_INLINE PetscErrorCode PetscSFPackRootData(PetscSF sf,PetscSFPack link,PetscInt nrootranks,PetscInt ndrootranks,const PetscInt *rootoffset,const PetscInt *rootloc,const void *rootdata)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i;
  PackInfo       pinfo[2] = {{rootoffset[ndrootranks], 0, bas->selfrootpackopt, link->selfbuf}, {rootoffset[nrootranks]-rootoffset[ndrootranks], rootoffset[ndrootranks], bas->rootpackopt, link->rootbuf}};

  PetscFunctionBegin;
  /* Only do packing when count != 0 so that we can avoid invoking CUDA kernels on GPU. */
  for (i=0; i<2; i++) {if (pinfo[i].count) {ierr = (*link->Pack)(pinfo[i].count,rootloc+pinfo[i].offset,link->bs,pinfo[i].opt,rootdata,pinfo[i].buf);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}

/* Utility routine to pack selected entries of leafdata into leaf buffer */
PETSC_STATIC_INLINE PetscErrorCode PetscSFPackLeafData(PetscSF sf,PetscSFPack link,PetscInt nleafranks,PetscInt ndleafranks,const PetscInt *leafoffset,const PetscInt *leafloc,const void *leafdata)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PackInfo       pinfo[2] = {{leafoffset[ndleafranks], 0, sf->selfleafpackopt, link->selfbuf}, {leafoffset[nleafranks]-leafoffset[ndleafranks], leafoffset[ndleafranks], sf->leafpackopt, link->leafbuf}};

  PetscFunctionBegin;
  for (i=0; i<2; i++) {if (pinfo[i].count) {ierr = (*link->Pack)(pinfo[i].count,leafloc+pinfo[i].offset,link->bs,pinfo[i].opt,leafdata,pinfo[i].buf);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}

/* Utility routine to unpack data from root buffer and Op it into selected entries of rootdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFUnpackAndOpRootData(PetscSF sf,PetscSFPack link,PetscInt nrootranks,PetscInt ndrootranks,const PetscInt *rootoffset,const PetscInt *rootloc,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode (*UnpackAndOp)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);
  PackInfo       pinfo[2] = {{rootoffset[ndrootranks], 0, bas->selfrootpackopt, link->selfbuf}, {rootoffset[nrootranks]-rootoffset[ndrootranks], rootoffset[ndrootranks], bas->rootpackopt, link->rootbuf}};

  PetscFunctionBegin;
  ierr = PetscSFPackGetUnpackAndOp(sf,(PetscSFPack)link,op,&UnpackAndOp);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    if (UnpackAndOp && pinfo[i].count) {ierr = (*UnpackAndOp)(pinfo[i].count,rootloc+pinfo[i].offset,link->bs,pinfo[i].opt,rootdata,pinfo[i].buf);CHKERRQ(ierr);}
    else {for (j=0; j<pinfo[i].count; j++) {ierr = MPI_Reduce_local(pinfo[i].buf+j*link->unitbytes,(char *)rootdata+(rootloc[pinfo[i].offset+j])*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);}}
  }
  PetscFunctionReturn(0);
}

/* Utility routine to unpack data from leaf buffer and Op it into selected entries of leafdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFUnpackAndOpLeafData(PetscSF sf,PetscSFPack link,PetscInt nleafranks,PetscInt ndleafranks,const PetscInt *leafoffset,const PetscInt *leafloc, void *leafdata,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscErrorCode (*UnpackAndOp)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);
  PackInfo       pinfo[2] = {{leafoffset[ndleafranks], 0, sf->selfleafpackopt, link->selfbuf}, {leafoffset[nleafranks]-leafoffset[ndleafranks], leafoffset[ndleafranks], sf->leafpackopt, link->leafbuf}};

  PetscFunctionBegin;
  ierr = PetscSFPackGetUnpackAndOp(sf,(PetscSFPack)link,op,&UnpackAndOp);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    if (UnpackAndOp && pinfo[i].count) {ierr = (*UnpackAndOp)(pinfo[i].count,leafloc+pinfo[i].offset,link->bs,pinfo[i].opt,leafdata,pinfo[i].buf);CHKERRQ(ierr);}
    else {for (j=0; j<pinfo[i].count; j++) {ierr = MPI_Reduce_local(pinfo[i].buf+j*link->unitbytes,(char *)leafdata+(leafloc[pinfo[i].offset+j])*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);}}
  }
  PetscFunctionReturn(0);
}

/* Utility routine to fetch and Op selected entries of rootdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFFetchAndOpRootData(PetscSF sf,PetscSFPack link,PetscInt nrootranks,PetscInt ndrootranks,const PetscInt *rootoffset,const PetscInt *rootloc,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode (*FetchAndOp)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);
  PackInfo       pinfo[2] = {{rootoffset[ndrootranks], 0, bas->selfrootpackopt, link->selfbuf}, {rootoffset[nrootranks]-rootoffset[ndrootranks], rootoffset[ndrootranks], bas->rootpackopt, link->rootbuf}};

  PetscFunctionBegin;
  ierr = PetscSFPackGetFetchAndOp(sf,(PetscSFPack)link,op,&FetchAndOp);CHKERRQ(ierr);
  for (i=0; i<2; i++) {if (pinfo[i].count) {ierr = (*FetchAndOp)(pinfo[i].count,rootloc+pinfo[i].offset,link->bs,pinfo[i].opt,rootdata,pinfo[i].buf);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscSFPackSetupOptimization_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackSetupOptimization(sf->ndranks,               sf->roffset,               sf->rmine,    &sf->selfleafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackSetupOptimization(sf->nranks-sf->ndranks,    sf->roffset+sf->ndranks,   sf->rmine,    &sf->leafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackSetupOptimization(bas->ndiranks,             bas->ioffset,              bas->irootloc,&bas->selfrootpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackSetupOptimization(bas->niranks-bas->ndiranks,bas->ioffset+bas->ndiranks,bas->irootloc,&bas->rootpackopt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFPackDestroyOptimization_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackDestoryOptimization(&sf->leafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackDestoryOptimization(&sf->selfleafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackDestoryOptimization(&bas->rootpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackDestoryOptimization(&bas->selfrootpackopt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFPackWaitall_Basic(PetscSFPack_Basic link,PetscSFDirection direction)
{
  PetscErrorCode ierr;
  MPI_Request    *requests = (direction == PETSCSF_LEAF2ROOT_REDUCE) ? link->requests : link->requests  + link->half;

  PetscFunctionBegin;
  ierr = MPI_Waitall(link->half,requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFSetUp_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFView_Basic(PetscSF,PetscViewer);
PETSC_INTERN PetscErrorCode PetscSFReset_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFDestroy_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFBcastAndOpEnd_Basic(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Basic(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedSF_Basic(PetscSF,PetscInt,const PetscInt*,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedLeafSF_Basic(PetscSF,PetscInt,const PetscInt*,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Basic(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFPackGet_Basic_Common(PetscSF,MPI_Datatype,const void*,const void*,PetscInt,PetscSFPack_Basic*);
#endif
