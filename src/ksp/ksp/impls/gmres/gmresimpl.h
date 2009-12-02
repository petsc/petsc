/*
   Private data structure used by the GMRES method. This data structure
  must be identical to the beginning of the KSP_FGMRES data structure
  so if you CHANGE anything here you must also change it there.
*/
#if !defined(__GMRES)
#define __GMRES

#include "private/kspimpl.h"        /*I "petscksp.h" I*/

typedef struct {
  /* Hessenberg matrix and orthogonalization information.  Hes holds
       the original (unmodified) hessenberg matrix which may be used
       to estimate the Singular Values of the matrix */
  PetscScalar *hh_origin,*hes_origin,*cc_origin,*ss_origin,*rs_origin;

  PetscScalar *orthogwork; /* holds dot products computed in orthogonalization */

  /* Work space for computing eigenvalues/singular values */
  PetscReal   *Dsvd;
  PetscScalar *Rsvd;
      
  /* parameters */
  PetscReal haptol;
  PetscInt  max_k;

  PetscErrorCode (*orthog)(KSP,PetscInt); /* Functions to use (special to gmres) */
  KSPGMRESCGSRefinementType cgstype;
    
  Vec   *vecs;  /* holds the work vectors */
  /* vv_allocated is the number of allocated gmres direction vectors */
  PetscInt    q_preallocate,delta_allocate;
  PetscInt    vv_allocated;
  /* vecs_allocated is the total number of vecs available (used to 
       simplify the dynamic allocation of vectors */
  PetscInt    vecs_allocated;
  /* Since we may call the user "obtain_work_vectors" several times, 
       we have to keep track of the pointers that it has returned 
      (so that we may free the storage) */
  Vec         **user_work;
  PetscInt    *mwork_alloc;    /* Number of work vectors allocated as part of
                               a work-vector chunck */
  PetscInt    nwork_alloc;     /* Number of work vectors allocated */

  /* In order to allow the solution to be constructed during the solution
     process, we need some additional information: */

  PetscInt    it;              /* Current iteration: inside restart */
  PetscScalar *nrs;            /* temp that holds the coefficients of the 
                               Krylov vectors that form the minimum residual
                               solution */
  Vec         sol_temp;        /* used to hold temporary solution */
} KSP_GMRES;

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
