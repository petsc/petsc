#ifndef __TAOLINESEARCH_IMPL_H
#define __TAOLINESEARCH_IMPL_H
#include "petscvec.h"
#include "taosolver.h"
#include "taolinesearch.h"

    
typedef struct _TaoLineSearchOps *TaoLineSearchOps;
struct _TaoLineSearchOps {
    PetscErrorCode (*computeobjective)(TaoLineSearch, Vec, PetscReal*, void*);
    PetscErrorCode (*computegradient)(TaoLineSearch, Vec, Vec, void*);
    PetscErrorCode (*computeobjectiveandgradient)(TaoLineSearch, Vec, PetscReal *, Vec, void*);
    PetscErrorCode (*setup)(TaoLineSearch);
    PetscErrorCode (*apply)(TaoLineSearch,Vec,PetscReal*,Vec,Vec);
    PetscErrorCode (*view)(TaoLineSearch,PetscViewer);
    PetscErrorCode (*setfromoptions)(TaoLineSearch);
    PetscErrorCode (*destroy)(TaoLineSearch);
};

struct _p_TaoLineSearch {
    PETSCHEADER(struct _TaoLineSearchOps);
    void *userctx_func;
    void *userctx_grad;
    void *userctx_funcgrad;
    
    PetscTruth setupcalled;
    void *data;


    Vec start_x;
    PetscReal f_fullstep;
    PetscReal new_f;
    Vec new_x;
    Vec new_g;
    PetscReal step;

    PetscInt maxfev;
    PetscInt nfev;
    TaoLineSearchTerminationReason reason;

    double rtol;	 /* relative tol for acceptable step (rtol>0) */
    double ftol;	 /* tol for sufficient decr. condition (ftol>0) */
    double gtol;	 /* tol for curvature condition (gtol>0)*/
    double stepmin;	 /* lower bound for step */
    double stepmax;	 /* upper bound for step */

    TaoSolver taosolver;
    
};

extern PetscLogEvent TaoLineSearch_ApplyEvent, TaoLineSearch_EvalEvent;
#endif
