#if !defined(_packimpl_h)
#define _packimpl_h

#include <petscdmcomposite.h>    /*I "petscdmcomposite.h" I*/
#include <petsc-private/dmimpl.h>      /*I      "petscdm.h"     I*/

/*
   rstart is where an array/subvector starts in the global parallel vector, so arrays
   rstarts are meaningless (and set to the previous one) except on the processor where the array lives
*/

struct DMCompositeLink {
  struct DMCompositeLink *next;
  PetscInt               n;             /* number of owned */
  PetscInt               rstart;        /* rstart is relative to this process */
  PetscInt               grstart;       /* grstart is relative to all processes */
  PetscInt               nlocal;

  /* only used for DMCOMPOSITE_DM */
  PetscInt               *grstarts;     /* global row for first unknown of this DM on each process */
  DM                     dm;
};

typedef struct {
  PetscInt               n,N,rstart;           /* rstart is relative to all processors, n unknowns owned by this process, N is total unknowns */
  PetscInt               nghost;               /* number of all local entries (includes DMDA ghost points) */
  PetscInt               nDM,nmine;            /* how many DM's and seperate redundant arrays used to build DM(nmine is ones on this process) */
  PetscBool              setup;                /* after this is set, cannot add new links to the DM*/
  struct DMCompositeLink *next;

  PetscErrorCode (*FormCoupleLocations)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt);
} DM_Composite;

extern PetscErrorCode DMCreateMatrix_Composite(DM,const MatType,Mat*);

#endif
