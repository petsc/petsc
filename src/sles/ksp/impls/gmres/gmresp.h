/* $Id: gmresp.h,v 1.3 1995/10/30 03:50:33 bsmith Exp bsmith $ */
/*
   Private data structure used by the GMRES method.
*/
#if !defined(__GMRES)
#define __GMRES

#include "petsc.h"
#include "kspimpl.h"        /*I "ksp.h" I*/

typedef struct {
    /* Hessenberg matrix and orthogonalization information.  Hes holds
       the original (unmodified) hessenberg matrix which may be used
       to estimate the eigenvalues of the matrix */
    Scalar *hh_origin, *hes_origin, *cc_origin, *ss_origin, *rs_origin;

    /* parameters */
    double haptol, epsabs;        
    int    max_k;

    int   (*orthog)(KSP,int); /* Functions to use (special to gmres) */
    
    Vec   *vecs;  /* holds the work vectors */
    /* vv_allocated is the number of allocated gmres direction vectors */
    int    q_preallocate, delta_allocate;
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
    int    it;              /* Current iteration */
    Scalar *nrs;            /* temp that holds the coefficients of the 
                               Krylov vectors that form the minimum residual
                               solution */
    Vec    sol_temp;       /* used to hold temporary solution */
    } KSP_GMRES;

#define HH(a,b)  (gmresP->hh_origin + (b)*(gmresP->max_k+2)+(a))
#define HES(a,b) (gmresP->hes_origin + (b)*(gmresP->max_k+1)+(a))
#define CC(a)    (gmresP->cc_origin + (a))
#define SS(a)    (gmresP->ss_origin + (a))
#define RS(a)    (gmresP->rs_origin + (a))

/* vector names */
#define VEC_OFFSET     3
#define VEC_SOLN       itP->vec_sol
#define VEC_RHS        itP->vec_rhs
#define VEC_TEMP       gmresP->vecs[0]
#define VEC_TEMP_MATOP gmresP->vecs[1]
#define VEC_BINVF      gmresP->vecs[2]
#define VEC_VV(i)      gmresP->vecs[VEC_OFFSET+i]

#endif
