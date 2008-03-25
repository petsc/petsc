#ifndef __TAOSOLVER_IMPL_H
#define __TAOSOLVER_IMPL_H

#include "taosolver.h"
#include "taolinesearch.h"

typedef struct _TaoSolverOps *TaoSolverOps;

struct _TaoSolverOps {
    PetscErrorCode (*computeobjective)(TaoSolver, Vec, PetscScalar*, void*);
    PetscErrorCode (*computegradient)(TaoSolver, Vec, Vec, void*);
    PetscErrorCode (*setup)(TaoSolver);
    PetscErrorCode (*solve)(TaoSolver);
    PetscErrorCode (*view)(TaoSolver, PetscViewer);
    PetscErrorCode (*setfromoptions)(TaoSolver);
    PetscErrorCode (*destroy)(TaoSolver);
};

struct _p_TaoSolver {
    PETSCHEADER(struct _TaoSolverOps);
    void *user_func;
    void *user_grad;


    PetscTruth setupcalled;
    void *data;

    Vec solution;
    PetscInt  max_its;
    PetscInt  max_funcs;
    PetscInt  nfuncs;
    PetscInt  ngrads;
    PetscInt  iter;
    
    TaoLineSearch *ls;

    PetscScalar residual;
    PetscScalar rtol;
};

extern PetscEvent TaoSolver_Solve, TaoSolver_FunctionEval, TaoSolver_GradientEval;
    


#endif
