/* $Id: gmresp.h,v 1.18 2001/08/07 03:03:51 balay Exp $ */
/*
   Private data structure used by the GMRES method.
*/
#if !defined(__GMRES)
#define __GMRES

#include "src/ksp/ksp/kspimpl.h"        /*I "petscksp.h" I*/

typedef struct {
  /* Hessenberg matrix and orthogonalization information.  Hes holds
       the original (unmodified) hessenberg matrix which may be used
       to estimate the Singular Values of the matrix */
  PetscScalar *hh_origin,*hes_origin,*cc_origin,*ss_origin,*rs_origin;
  /* Work space for computing eigenvalues/singular values */
  PetscReal   *Dsvd;
  PetscScalar *Rsvd;
      
  /* parameters */
  PetscReal haptol;
  int       max_k;

  int                       (*orthog)(KSP,int); /* Functions to use (special to gmres) */
  KSPGMRESCGSRefinementType cgstype;
    
  Vec   *vecs;  /* holds the work vectors */
  /* vv_allocated is the number of allocated gmres direction vectors */
  int    q_preallocate,delta_allocate;
  int    vv_allocated;
  /* vecs_allocated is the total number of vecs available (used to 
       simplify the dynamic allocation of vectors */
  int    vecs_allocated;
  /* Since we may call the user "obtain_work_vectors" several times, 
       we have to keep track of the pointers that it has returned 
      (so that we may free the storage) */
  Vec    **user_work;
  int    *mwork_alloc;    /* Number of work vectors allocated as part of
                               a work-vector chunck */
  int    nwork_alloc;     /* Number of work vectors allocated */

  /* In order to allow the solution to be constructed during the solution
     process, we need some additional information: */

  int    it;              /* Current iteration: inside restart */
  PetscScalar *nrs;            /* temp that holds the coefficients of the 
                               Krylov vectors that form the minimum residual
                               solution */
  Vec    sol_temp;        /* used to hold temporary solution */
} KSP_GMRES;

#define HH(a,b)  (gmres->hh_origin + (b)*(gmres->max_k+2)+(a))
#define HES(a,b) (gmres->hes_origin + (b)*(gmres->max_k+1)+(a))
#define CC(a)    (gmres->cc_origin + (a))
#define SS(a)    (gmres->ss_origin + (a))
#define GRS(a)    (gmres->rs_origin + (a))

/* vector names */
#define VEC_OFFSET     2
#define VEC_SOLN       ksp->vec_sol
#define VEC_RHS        ksp->vec_rhs
#define VEC_TEMP       gmres->vecs[0]
#define VEC_TEMP_MATOP gmres->vecs[1]
#define VEC_VV(i)      gmres->vecs[VEC_OFFSET+i]

#endif
