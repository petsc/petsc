/*$Id: fgmresp.h,v 1.7 2001/08/07 03:03:55 balay Exp $*/
/*
   Private data structure used by the FGMRES method.
*/

#if !defined(__FGMRES)
#define __FGMRES

#include "src/sles/ksp/kspimpl.h"

  typedef struct {
    /* Hessenberg matrix and orthogonalization information. */ 
    PetscScalar *hh_origin;   /* holds hessenburg matrix that has been
                            multiplied by plane rotations (upper tri) */
    PetscScalar *hes_origin;  /* holds the original (unmodified) hessenberg matrix 
                            which may be used to estimate the Singular Values
                            of the matrix */
    PetscScalar *cc_origin;   /* holds cosines for rotation matrices */
    PetscScalar *ss_origin;   /* holds sines for rotation matrices */
    PetscScalar *rs_origin;   /* holds the right-hand-side of the Hessenberg system */

    /* Work space for computing eigenvalues/singular values */
    PetscReal *Dsvd;
    PetscScalar *Rsvd;
      
    /* parameters */
    PetscReal haptol;            /* tolerance used for the "HAPPY BREAK DOWN"  */
    int    max_k;             /* maximum number of Krylov directions to find 
                                 before restarting */

    int   (*orthog)(KSP,int); /* orthogonalization function to use */
    
    Vec   *vecs;              /* holds the work vectors */
   
    int    q_preallocate;     /* 0 = don't pre-allocate space for work vectors */
    int    delta_allocate;    /* the number of vectors to allocate in each block 
                                 if not pre-allocated */
    int    vv_allocated;      /* vv_allocated is the number of allocated fgmres 
                                 direction vectors */
    
    int    vecs_allocated;    /* vecs_allocated is the total number of vecs 
                                 available - used to simplify the dynamic
                                 allocation of vectors */
   
    Vec    **user_work;       /* Since we may call the user "obtain_work_vectors" 
                                 several times, we have to keep track of the pointers
                                 that it has returned (so that we may free the 
                                 storage) */

    int    *mwork_alloc;      /* Number of work vectors allocated as part of
                                 a work-vector chunck */
    int    nwork_alloc;       /* Number of work-vector chunks allocated */


    /* In order to allow the solution to be constructed during the solution
       process, we need some additional information: */

    int    it;              /* Current iteration */
    PetscScalar *nrs;            /* temp that holds the coefficients of the 
                               Krylov vectors that form the minimum residual
                               solution */
    Vec    sol_temp;        /* used to hold temporary solution */


    /* new storage for fgmres */
    Vec  *prevecs;            /* holds the preconditioned basis vectors for fgmres.  
                                We will allocate these at the same time as vecs 
                                above (and in the same "chunks". */
    Vec  **prevecs_user_work; /* same purpose as user_work above, but this one is
                                for our preconditioned vectors */

    /* we need a function for interacting with the pcfamily */
   
    int    (*modifypc)(KSP,int,int,PetscReal,void*);  /* function to modify the preconditioner*/
    int    (*modifydestroy)(void*);
    void   *modifyctx;
} KSP_FGMRES;


#define HH(a,b)  (fgmres->hh_origin + (b)*(fgmres->max_k+2)+(a)) 
                 /* HH will be size (max_k+2)*(max_k+1)  -  think of HH as 
                    being stored columnwise for access purposes. */
#define HES(a,b) (fgmres->hes_origin + (b)*(fgmres->max_k+1)+(a)) 
                  /* HES will be size (max_k + 1) * (max_k + 1) - 
                     again, think of HES as being stored columnwise */
#define CC(a)    (fgmres->cc_origin + (a)) /* CC will be length (max_k+1) - cosines */
#define SS(a)    (fgmres->ss_origin + (a)) /* SS will be length (max_k+1) - sines */
#define RS(a)    (fgmres->rs_origin + (a)) /* RS will be length (max_k+2) - rt side */

/* vector names */
#define VEC_OFFSET     2
#define VEC_SOLN       ksp->vec_sol                  /* solution */ 
#define VEC_RHS        ksp->vec_rhs                  /* right-hand side */
#define VEC_TEMP       fgmres->vecs[0]               /* work space */  
#define VEC_TEMP_MATOP fgmres->vecs[1]               /* work space */
#define VEC_VV(i)      fgmres->vecs[VEC_OFFSET+i]    /* use to access
                                                        othog basis vectors */
#define PREVEC(i)      fgmres->prevecs[i]            /* use to access 
                                                        preconditioned basis */ 

#endif



