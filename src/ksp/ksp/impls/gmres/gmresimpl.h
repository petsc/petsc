/*
   Private data structure used by the GMRES method. This data structure
  must be identical to the beginning of the KSP_FGMRES data structure
  so if you CHANGE anything here you must also change it there.
*/
#if !defined(__GMRES)
#define __GMRES

#include <petsc-private/kspimpl.h>        /*I "petscksp.h" I*/

#define KSPGMRESHEADER                                                  \
  /* Hessenberg matrix and orthogonalization information. */            \
  PetscScalar *hh_origin;   /* holds hessenburg matrix that has been multiplied by plane rotations (upper tri) */ \
  PetscScalar *hes_origin;  /* holds the original (unmodified) hessenberg matrix which may be used to estimate the Singular Values of the matrix */ \
  PetscScalar *cc_origin;   /* holds cosines for rotation matrices */   \
  PetscScalar *ss_origin;   /* holds sines for rotation matrices */     \
  PetscScalar *rs_origin;   /* holds the right-hand-side of the Hessenberg system */ \
                                                                        \
  PetscScalar *orthogwork; /* holds dot products computed in orthogonalization */ \
                                                                        \
  /* Work space for computing eigenvalues/singular values */            \
  PetscReal   *Dsvd;                                                    \
  PetscScalar *Rsvd;                                                    \
                                                                        \
                                                                        \
  PetscReal haptol;                                       /* tolerance for happy ending */ \
  PetscInt  max_k;                                        /* number of vectors in Krylov space, restart size */ \
  PetscInt  nextra_vecs;                                  /* number of extra vecs needed, e.g. for a pipeline */ \
                                                                        \
  PetscErrorCode            (*orthog)(KSP,PetscInt);                    \
  KSPGMRESCGSRefinementType cgstype;                                    \
                                                                        \
  Vec         *vecs;                                     /* the work vectors */ \
  PetscInt    q_preallocate; /* 0=don't preallocate space for work vectors */ \
  PetscInt    delta_allocate; /* number of vectors to preallocaate in each block if not preallocated */ \
  PetscInt    vv_allocated;   /* number of allocated gmres direction vectors */ \
  PetscInt    vecs_allocated;                           /*   total number of vecs available */ \
  /* Since we may call the user "obtain_work_vectors" several times, we have to keep track of the pointers that it has returned */ \
  Vec         **user_work;                                              \
  PetscInt    *mwork_alloc;    /* Number of work vectors allocated as part of  a work-vector chunck */ \
  PetscInt    nwork_alloc;     /* Number of work vector chunks allocated */ \
                                                                        \
  /* Information for building solution */                               \
  PetscInt    it;              /* Current iteration: inside restart */  \
  PetscScalar *nrs;            /* temp that holds the coefficients of the Krylov vectors that form the minimum residual solution */ \
  Vec         sol_temp;        /* used to hold temporary solution */

typedef struct {
  KSPGMRESHEADER
} KSP_GMRES;

extern PetscErrorCode KSPView_GMRES(KSP,PetscViewer);
extern PetscErrorCode KSPSetUp_GMRES(KSP);
extern PetscErrorCode KSPSetFromOptions_GMRES(KSP);
extern PetscErrorCode KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
extern PetscErrorCode KSPComputeEigenvalues_GMRES(KSP,PetscInt,PetscReal *,PetscReal *,PetscInt *);
extern PetscErrorCode KSPReset_GMRES(KSP);
extern PetscErrorCode KSPDestroy_GMRES(KSP);
extern PetscErrorCode KSPGMRESGetNewVectors(KSP,PetscInt);

typedef PetscErrorCode (*FCN)(KSP,PetscInt); /* force argument to next function to not be extern C*/

PETSC_EXTERN_C PetscErrorCode KSPGMRESSetHapTol_GMRES(KSP,PetscReal);
PETSC_EXTERN_C PetscErrorCode KSPGMRESSetPreAllocateVectors_GMRES(KSP);
PETSC_EXTERN_C PetscErrorCode KSPGMRESSetRestart_GMRES(KSP,PetscInt);
PETSC_EXTERN_C PetscErrorCode KSPGMRESGetRestart_GMRES(KSP,PetscInt*);
PETSC_EXTERN_C PetscErrorCode KSPGMRESSetOrthogonalization_GMRES(KSP,FCN);
PETSC_EXTERN_C PetscErrorCode KSPGMRESGetOrthogonalization_GMRES(KSP,FCN*);
PETSC_EXTERN_C PetscErrorCode KSPGMRESSetCGSRefinementType_GMRES(KSP,KSPGMRESCGSRefinementType);
PETSC_EXTERN_C PetscErrorCode KSPGMRESGetCGSRefinementType_GMRES(KSP,KSPGMRESCGSRefinementType*);

/* These macros are guarded because they are redefined by derived implementations */
#if !defined(KSPGMRES_NO_MACROS)
#define HH(a,b)  (gmres->hh_origin + (b)*(gmres->max_k+2)+(a))
#define HES(a,b) (gmres->hes_origin + (b)*(gmres->max_k+1)+(a))
#define CC(a)    (gmres->cc_origin + (a))
#define SS(a)    (gmres->ss_origin + (a))
#define GRS(a)   (gmres->rs_origin + (a))

/* vector names */
#define VEC_OFFSET     2
#define VEC_TEMP       gmres->vecs[0]
#define VEC_TEMP_MATOP gmres->vecs[1]
#define VEC_VV(i)      gmres->vecs[VEC_OFFSET+i]
#endif

#endif
