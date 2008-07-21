#ifndef __TAOSOLVER_IMPL_H
#define __TAOSOLVER_IMPL_H

#include "taosolver.h"
#include "taolinesearch.h"

typedef struct _TaoSolverOps *TaoSolverOps;

struct _TaoSolverOps {
    PetscErrorCode (*computeobjective)(TaoSolver, Vec, PetscScalar*, void*);
    PetscErrorCode (*computeobjectiveandgradient)(TaoSolver, Vec, PetscScalar*, Vec, void*);
    PetscErrorCode (*computegradient)(TaoSolver, Vec, Vec, void*);
    PetscErrorCode (*converged)(TaoSolver,void*);
    PetscErrorCode (*convergeddestroy)(void*);
    PetscErrorCode (*setup)(TaoSolver);
    PetscErrorCode (*solve)(TaoSolver);
    PetscErrorCode (*view)(TaoSolver, PetscViewer);
    PetscErrorCode (*setfromoptions)(TaoSolver);
    PetscErrorCode (*destroy)(TaoSolver);
};

#define MAXTAOMONITORS 10

struct _p_TaoSolver {
    PETSCHEADER(struct _TaoSolverOps);
    void *user_obj;
    void *user_grad;

    PetscErrorCode (*monitor[MAXTAOMONITORS])(TaoSolver,void*);
    PetscErrorCode (*monitordestroy[MAXTAOMONITORS])(void*);
    void *monitorcontext[MAXTAOMONITORS];
    PetscInt numbermonitors;
    void *cnvP; 
    TaoSolverConvergedReason reason;

    PetscTruth setupcalled;
    void *data;

    Vec solution;
    PetscInt  max_its;
    PetscInt  max_funcs;
    PetscInt  nfuncs;
    PetscInt  ngrads;
    PetscInt  nfuncgrads;
    PetscInt  nhess;
    PetscInt  iter;
    
    TaoLineSearch *ls;

    PetscScalar residual;
    PetscScalar rtol;

    PetscTruth printreason;

    PetscReal *conv_hist; /* Number of iteration histories to keep */
    PetscInt *conv_hist_feval; /* Number of func evals at each iteration */
    PetscInt *conv_hist_geval; /* Number of grad evals at each iteration */
    PetscInt *conv_hist_heval; /* Number of hess evals at each iteration */
    PetscInt conv_hist_len;
    PetscInt conv_hist_max;
    PetscTruth conv_hist_reset;

    
};

extern PetscLogEvent TaoSolver_Solve, TaoSolver_ObjectiveEval, TaoSolver_GradientEval, TaoSolver_HessianEval, TaoSolver_JacobianEval;
    


#endif
