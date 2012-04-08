#if !defined(__FGMRES)
#define __FGMRES

#include <petsc-private/kspimpl.h>
#define KSPGMRES_NO_MACROS
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

typedef struct {
  KSPGMRESHEADER

    /* new storage for fgmres */
    Vec         *prevecs;        /* holds the preconditioned basis vectors for fgmres.  
                                    We will allocate these at the same time as vecs 
                                    above (and in the same "chunks". */
    Vec          **prevecs_user_work; /* same purpose as user_work above, but this one is
                                    for our preconditioned vectors */

    /* we need a function for interacting with the pcfamily */
   
    PetscErrorCode (*modifypc)(KSP,PetscInt,PetscInt,PetscReal,void*);  /* function to modify the preconditioner*/
    PetscErrorCode (*modifydestroy)(void*);
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
#define VEC_TEMP       fgmres->vecs[0]               /* work space */  
#define VEC_TEMP_MATOP fgmres->vecs[1]               /* work space */
#define VEC_VV(i)      fgmres->vecs[VEC_OFFSET+i]    /* use to access
                                                        othog basis vectors */
#define PREVEC(i)      fgmres->prevecs[i]            /* use to access 
                                                        preconditioned basis */ 

#endif



