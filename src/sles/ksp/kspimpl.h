/* $Id: kspimpl.h,v 1.50 2001/08/06 18:04:29 bsmith Exp $ */

#ifndef _KSPIMPL
#define _KSPIMPL

#include "petscksp.h"

typedef struct _KSPOps *KSPOps;

struct _KSPOps {
  int  (*buildsolution)(KSP,Vec,Vec*);       /* Returns a pointer to the solution, or
                                                calculates the solution in a 
				                user-provided area. */
  int  (*buildresidual)(KSP,Vec,Vec,Vec*);   /* Returns a pointer to the residual, or
				                calculates the residual in a 
				                user-provided area.  */
  int  (*solve)(KSP,int*);                   /* actual solver */
  int  (*setup)(KSP);
  int  (*setfromoptions)(KSP);
  int  (*publishoptions)(KSP);
  int  (*computeextremesingularvalues)(KSP,PetscReal*,PetscReal*);
  int  (*computeeigenvalues)(KSP,int,PetscReal*,PetscReal*,int *);
  int  (*destroy)(KSP);
  int  (*view)(KSP,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single KSP
*/
#define MAXKSPMONITORS 5 

/*
   Defines the KSP data structure.
*/
struct _p_KSP {
  PETSCHEADER(struct _KSPOps)
  /*------------------------- User parameters--------------------------*/
  int max_it;                     /* maximum number of iterations */
  PetscTruth    guess_zero,                  /* flag for whether initial guess is 0 */
                calc_sings,                  /* calculate extreme Singular Values */
                guess_knoll;                /* use initial guess of PCApply(ksp->B,b */
  PCSide pc_side;                  /* flag for left, right, or symmetric 
                                      preconditioning */
  PetscReal rtol,                     /* relative tolerance */
            atol,                     /* absolute tolerance */
            ttol,                     /* (not set by user)  */
            divtol;                   /* divergence tolerance */
  PetscReal rnorm0;                   /* initial residual norm (used for divergence testing) */
  PetscReal rnorm;                    /* current residual norm */
  KSPConvergedReason reason;     

  Vec vec_sol,vec_rhs;            /* pointer to where user has stashed 
                                      the solution and rhs, these are 
                                      never touched by the code, only 
                                      passed back to the user */ 
  PetscReal     *res_hist;            /* If !0 stores residual at iterations*/
  int           res_hist_len;         /* current size of residual history array */
  int           res_hist_max;         /* actual amount of data in residual_history */
  PetscTruth    res_hist_reset;       /* reset history to size zero for each new solve */

  /* --------User (or default) routines (most return -1 on error) --------*/
  int  (*monitor[MAXKSPMONITORS])(KSP,int,PetscReal,void*); /* returns control to user after */
  int  (*monitordestroy[MAXKSPMONITORS])(void*);         /* */
  void *monitorcontext[MAXKSPMONITORS];                  /* residual calculation, allows user */
  int  numbermonitors;                                   /* to, for instance, print residual norm, etc. */

  int        (*converged)(KSP,int,PetscReal,KSPConvergedReason*,void*);
  void       *cnvP; 

  PC         B;

  void       *data;                      /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  int        nwork;
  Vec        *work;

  int        setupcalled;

  int        its;       /* number of iterations so far computed */

  PetscTruth transpose_solve;    /* solve transpose system instead */

  KSPNormType normtype;          /* type of norm used for convergence tests */
};

#define KSPLogResidualHistory(ksp,norm) \
    {if (ksp->res_hist && ksp->res_hist_max > ksp->res_hist_len) \
     ksp->res_hist[ksp->res_hist_len++] = norm;}

#define KSPMonitor(ksp,it,rnorm) \
        { int _ierr,_i,_im = ksp->numbermonitors; \
          for (_i=0; _i<_im; _i++) {\
            _ierr = (*ksp->monitor[_i])(ksp,it,rnorm,ksp->monitorcontext[_i]);CHKERRQ(_ierr); \
	  } \
	}

EXTERN int KSPDefaultBuildSolution(KSP,Vec,Vec*);
EXTERN int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
EXTERN int KSPDefaultDestroy(KSP);
EXTERN int KSPDefaultGetWork(KSP,int);
EXTERN int KSPDefaultFreeWork(KSP);
EXTERN int KSPInitialResidual(KSP,Vec,Vec,Vec,Vec,Vec);
EXTERN int KSPUnwindPreconditioner(KSP,Vec,Vec);

/*
       These allow the various Krylov methods to apply to either the linear system
    or its transpose.
*/
#define KSP_MatMult(ksp,A,x,y)               (!ksp->transpose_solve) ?  MatMult(A,x,y)               : MatMultTranspose(A,x,y) 
#define KSP_MatMultTranspose(ksp,A,x,y)      (!ksp->transpose_solve) ?  MatMultTranspose(A,x,y)      : MatMult(A,x,y) 
#define KSP_PCApply(ksp,B,x,y)               (!ksp->transpose_solve) ?  PCApply(B,x,y,ksp->pc_side)               : PCApplyTranspose(B,x,y) 
#define KSP_PCApplyTranspose(ksp,B,x,y)      (!ksp->transpose_solve) ?  PCApplyTranspose(B,x,y)      : PCApply(B,x,y,ksp->pc_side) 
#define KSP_PCApplyBAorAB(ksp,pc,side,x,y,w) (!ksp->transpose_solve) ?  PCApplyBAorAB(pc,side,x,y,w) : PCApplyBAorABTranspose(pc,side,x,y,w)

#endif
