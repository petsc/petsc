#ifndef __TAOSOLVER_IMPL_H
#define __TAOSOLVER_IMPL_H

#include "taosolver.h"
#include "taolinesearch.h"
#include "petscksp.h"

typedef struct _TaoSolverOps *TaoSolverOps;

struct _TaoSolverOps {
    PetscErrorCode (*computeobjective)(TaoSolver, Vec, PetscScalar*, void*);
    PetscErrorCode (*computeobjectiveandgradient)(TaoSolver, Vec, PetscScalar*, Vec, void*);
    PetscErrorCode (*computegradient)(TaoSolver, Vec, Vec, void*);
    PetscErrorCode (*convergencetest)(TaoSolver,void*);
    PetscErrorCode (*convergencedestroy)(void*);

    PetscErrorCode (*setup)(TaoSolver);
    PetscErrorCode (*solve)(TaoSolver);
    PetscErrorCode (*view)(TaoSolver, PetscViewer);
    PetscErrorCode (*setfromoptions)(TaoSolver);
    PetscErrorCode (*destroy)(TaoSolver);
};

#define MAXTAOMONITORS 10

struct _p_TaoSolver {
    PETSCHEADER(struct _TaoSolverOps);
    void *user_objP;
    void *user_objgradP;
    void *user_gradP;
    void *user_hessP;
    void *user_jacP;

    PetscErrorCode (*monitor[MAXTAOMONITORS])(TaoSolver,void*);
    PetscErrorCode (*monitordestroy[MAXTAOMONITORS])(void*);
    void *monitorcontext[MAXTAOMONITORS];
    PetscInt numbermonitors;
    void *cnvP; 
    TaoSolverConvergedReason reason;

    PetscTruth setupcalled;
    void *data;

    Vec solution;
    Vec gradient;
    Vec stepdirection;
    PetscScalar step;
    PetscScalar residual;
    PetscScalar gnorm0;
    PetscScalar cnorm;
    PetscScalar cnorm0;
    PetscScalar fc;
    

    PetscInt  max_its;
    PetscInt  max_funcs;
    PetscInt  max_constraints;
    PetscInt  nfuncs;
    PetscInt  ngrads;
    PetscInt  nfuncgrads;
    PetscInt  nhess;
    PetscInt  niter;
    PetscInt  nconstraints;
    PetscInt  njac;

    
    TaoLineSearch linesearch;
    PetscTruth lsflag; /* goes up when line search fails */
    KSP *ksp;

    PetscScalar fatol;
    PetscScalar frtol;
    PetscScalar gatol;
    PetscScalar grtol;
    PetscScalar gttol;
    PetscScalar catol;
    PetscScalar crtol;
    PetscScalar xtol;
    PetscScalar trtol;
    PetscScalar fmin;

    PetscTruth printreason;
    PetscTruth viewtao;
    PetscTruth viewgradient;
    PetscTruth viewconstraint;
    PetscTruth viewhessian;
    PetscTruth viewjacobian;

    PetscInt conv_hist_max;/* Number of iteration histories to keep */
    PetscScalar *conv_hist; 
    PetscInt *conv_hist_feval; /* Number of func evals at each iteration */
    PetscInt *conv_hist_fgeval; /* Number of func/grad evals at each iteration */
    PetscInt *conv_hist_geval; /* Number of grad evals at each iteration */
    PetscInt *conv_hist_heval; /* Number of hess evals at each iteration */
    PetscInt conv_hist_len;
    PetscTruth conv_hist_reset;

    
};

extern PetscLogEvent TaoSolver_Solve, TaoSolver_ObjectiveEval, TaoSolver_ObjGradientEval, TaoSolver_GradientEval, TaoSolver_HessianEval, TaoSolver_JacobianEval;
    


#endif
