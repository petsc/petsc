#if !defined(__SFBASIC_H)
#define __SFBASIC_H

#include <../src/vec/is/sf/impls/basic/sfpack.h>

typedef enum {PETSCSF_LEAF2ROOT_REDUCE=0, PETSCSF_ROOT2LEAF_BCAST=1} PetscSFDirection;

typedef struct _n_PetscSFPack_Basic *PetscSFPack_Basic;

#define SPPACKBASICHEADER \
  SFPACKHEADER;                                                                                                                    \
  char          **root;         /* Packed root data, indexed by leaf rank */                                                       \
  char          **leaf;         /* Packed leaf data, indexed by root rank */                                                       \
  PetscMPIInt   half;           /* Number of MPI_Requests used for either leaf2root or root2leaf communication */                  \
  MPI_Request   *requests       /* [2*half] requests arranged in this order: leaf2root root/leaf reqs, root2leaf root/leaf reqs */

struct _n_PetscSFPack_Basic {
  SPPACKBASICHEADER;
  PetscBool     initialized[2]; /* Is the communcation pattern in each direction initialized? [0] for leaf2root, [1] for root2leaf */
};

#define SFBASICHEADER \
  PetscMPIInt      niranks;     /* Number of incoming ranks (ranks accessing my roots) */                                      \
  PetscMPIInt      ndiranks;    /* Number of incoming ranks (ranks accessing my roots) in distinguished set */                 \
  PetscMPIInt      *iranks;     /* Array of ranks that reference my roots */                                                   \
  PetscInt         itotal;      /* Total number of graph edges referencing my roots */                                         \
  PetscInt         *ioffset;    /* Array of length niranks+1 holding offset in irootloc[] for each rank */                     \
  PetscInt         *irootloc;   /* Incoming roots referenced by ranks starting at ioffset[rank] */                             \
  PetscSFPackOpt   rootpackopt; /* Optimization plans to (un)pack roots based on patterns in irootloc[]. NULL for no plans */  \
  PetscSFPack      avail;       /* One or more entries per MPI Datatype, lazily constructed */                                 \
  PetscSFPack      inuse        /* Buffers being used for transactions that have not yet completed */

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
#endif
