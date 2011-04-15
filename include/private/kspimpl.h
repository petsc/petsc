
#ifndef _KSPIMPL_H
#define _KSPIMPL_H

#include <petscksp.h>

typedef struct _KSPOps *KSPOps;

struct _KSPOps {
  PetscErrorCode (*buildsolution)(KSP,Vec,Vec*);       /* Returns a pointer to the solution, or
                                                calculates the solution in a 
				                user-provided area. */
  PetscErrorCode (*buildresidual)(KSP,Vec,Vec,Vec*);   /* Returns a pointer to the residual, or
				                calculates the residual in a 
				                user-provided area.  */
  PetscErrorCode (*solve)(KSP);                        /* actual solver */
  PetscErrorCode (*setup)(KSP);
  PetscErrorCode (*setfromoptions)(KSP);
  PetscErrorCode (*publishoptions)(KSP);
  PetscErrorCode (*computeextremesingularvalues)(KSP,PetscReal*,PetscReal*);
  PetscErrorCode (*computeeigenvalues)(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
  PetscErrorCode (*destroy)(KSP);
  PetscErrorCode (*view)(KSP,PetscViewer);
  PetscErrorCode (*reset)(KSP);
};

typedef struct {PetscInt model,curl,maxl;Mat mat; KSP ksp;}* KSPGuessFischer;

/*
     Maximum number of monitors you can run with a single KSP
*/
#define MAXKSPMONITORS 5 
typedef enum {KSP_SETUP_NEW, KSP_SETUP_NEWMATRIX, KSP_SETUP_NEWRHS} KSPSetUpStage;

/*
   Defines the KSP data structure.
*/
struct _p_KSP {
  PETSCHEADER(struct _KSPOps);
  DM              dm;
  PetscBool       dmActive;
  /*------------------------- User parameters--------------------------*/
  PetscInt        max_it;                     /* maximum number of iterations */
  KSPFischerGuess guess;
  PetscBool       guess_zero,                  /* flag for whether initial guess is 0 */
                  calc_sings,                  /* calculate extreme Singular Values */
                  guess_knoll;                /* use initial guess of PCApply(ksp->B,b */
  PCSide          pc_side;                  /* flag for left, right, or symmetric 
                                      preconditioning */
  PetscReal       rtol,                     /* relative tolerance */
                  abstol,                     /* absolute tolerance */
                  ttol,                     /* (not set by user)  */
                  divtol;                   /* divergence tolerance */
  PetscReal       rnorm0;                   /* initial residual norm (used for divergence testing) */
  PetscReal       rnorm;                    /* current residual norm */
  KSPConvergedReason reason;     
  PetscBool          printreason;     /* prints converged reason after solve */
  PetscBool          errorifnotconverged;    /* create an error if the KSPSolve() does not converge */

  Vec vec_sol,vec_rhs;            /* pointer to where user has stashed 
                                      the solution and rhs, these are 
                                      never touched by the code, only 
                                      passed back to the user */ 
  PetscReal     *res_hist;            /* If !0 stores residual at iterations*/
  PetscReal     *res_hist_alloc;      /* If !0 means user did not provide buffer, needs deallocation */
  PetscInt      res_hist_len;         /* current size of residual history array */
  PetscInt      res_hist_max;         /* actual amount of data in residual_history */
  PetscBool     res_hist_reset;       /* reset history to size zero for each new solve */

  PetscInt      chknorm;             /* only compute/check norm if iterations is great than this */
  PetscBool     lagnorm;             /* Lag the residual norm calculation so that it is computed as part of the 
                                        MPI_Allreduce() for computing the inner products for the next iteration. */ 
  /* --------User (or default) routines (most return -1 on error) --------*/
  PetscErrorCode (*monitor[MAXKSPMONITORS])(KSP,PetscInt,PetscReal,void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXKSPMONITORS])(void*);         /* */
  void *monitorcontext[MAXKSPMONITORS];                  /* residual calculation, allows user */
  PetscInt  numbermonitors;                                   /* to, for instance, print residual norm, etc. */

  PetscErrorCode (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  void       *cnvP; 

  PC         pc;

  void       *data;                      /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  PetscInt       nwork;
  Vec            *work;

  KSPSetUpStage  setupstage;

  PetscInt       its;       /* number of iterations so far computed */

  PetscBool      transpose_solve;    /* solve transpose system instead */

  KSPNormType    normtype;          /* type of norm used for convergence tests */

  /*   Allow diagonally scaling the matrix before computing the preconditioner or using 
       the Krylov method. Note this is NOT just Jacobi preconditioning */

  PetscBool    dscale;       /* diagonal scale system; used with KSPSetDiagonalScale() */
  PetscBool    dscalefix;    /* unscale system after solve */
  PetscBool    dscalefix2;   /* system has been unscaled */
  Vec          diagonal;     /* 1/sqrt(diag of matrix) */
  Vec          truediagonal;

  MatNullSpace nullsp;      /* Null space of the operator, removed from Krylov space */
};

typedef struct {
  PetscBool  initialrtol;    /* default relative residual decrease is computing from initial residual, not rhs */
  PetscBool  mininitialrtol; /* default relative residual decrease is computing from min of initial residual and rhs */
  Vec        work;
} KSPDefaultConvergedCtx;

#define KSPLogResidualHistory(ksp,norm) \
    {if (ksp->res_hist && ksp->res_hist_max > ksp->res_hist_len) \
     ksp->res_hist[ksp->res_hist_len++] = norm;}

#define KSPMonitor(ksp,it,rnorm) \
        { PetscErrorCode _ierr; PetscInt _i,_im = ksp->numbermonitors; \
          for (_i=0; _i<_im; _i++) {\
            _ierr = (*ksp->monitor[_i])(ksp,it,rnorm,ksp->monitorcontext[_i]);CHKERRQ(_ierr); \
	  } \
	}

extern PetscErrorCode KSPDefaultDestroy(KSP*);
extern PetscErrorCode KSPDefaultReset(KSP);
extern PetscErrorCode KSPGetVecs(KSP,PetscInt,Vec**,PetscInt,Vec**);
extern PetscErrorCode KSPDefaultGetWork(KSP,PetscInt);

/*
       These allow the various Krylov methods to apply to either the linear system or its transpose.
*/
#define KSP_RemoveNullSpace(ksp,y) ((ksp->nullsp && ksp->pc_side == PC_LEFT) ? MatNullSpaceRemove(ksp->nullsp,y,PETSC_NULL) : 0)

#define KSP_MatMult(ksp,A,x,y)          (!ksp->transpose_solve) ? MatMult(A,x,y)                                                            : MatMultTranspose(A,x,y) 
#define KSP_MatMultTranspose(ksp,A,x,y) (!ksp->transpose_solve) ? MatMultTranspose(A,x,y)                                                   : MatMult(A,x,y) 
#define KSP_PCApply(ksp,x,y)            (!ksp->transpose_solve) ? (PCApply(ksp->pc,x,y) || KSP_RemoveNullSpace(ksp,y))                      : PCApplyTranspose(ksp->pc,x,y) 
#define KSP_PCApplyTranspose(ksp,x,y)   (!ksp->transpose_solve) ? PCApplyTranspose(ksp->pc,x,y)                                             : (PCApply(ksp->pc,x,y) || KSP_RemoveNullSpace(ksp,y)) 
#define KSP_PCApplyBAorAB(ksp,x,y,w)    (!ksp->transpose_solve) ? (PCApplyBAorAB(ksp->pc,ksp->pc_side,x,y,w) || KSP_RemoveNullSpace(ksp,y)) : PCApplyBAorABTranspose(ksp->pc,ksp->pc_side,x,y,w)
#define KSP_PCApplyBAorABTranspose(ksp,x,y,w)    (!ksp->transpose_solve) ? (PCApplyBAorABTranspose(ksp->pc,ksp->pc_side,x,y,w) || KSP_RemoveNullSpace(ksp,y)) : PCApplyBAorAB(ksp->pc,ksp->pc_side,x,y,w)

extern PetscLogEvent KSP_GMRESOrthogonalization, KSP_SetUp, KSP_Solve;


#endif
