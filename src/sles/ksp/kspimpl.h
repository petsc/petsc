/* $Id: kspimpl.h,v 1.45 2000/05/05 22:17:26 balay Exp bsmith $ */

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
  int  (*printhelp)(KSP,char*);
  int  (*computeextremesingularvalues)(KSP,double*,double*);
  int  (*computeeigenvalues)(KSP,int,double*,double*,int *);
  int  (*destroy)(KSP);
  int  (*view)(KSP,Viewer);
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
                calc_res,                    /* calculate residuals at each iteration*/
                use_pres;                    /* use preconditioned residual */
  PCSide pc_side;                  /* flag for left, right, or symmetric 
                                      preconditioning */
  double rtol,                     /* relative tolerance */
         atol,                     /* absolute tolerance */
         ttol,                     /* (not set by user)  */
         divtol;                   /* divergence tolerance */
  double rnorm0;                   /* initial residual norm (used for divergence testing) */
  double rnorm;                    /* current residual norm */
  KSPConvergedReason reason;     

  Vec vec_sol,vec_rhs;            /* pointer to where user has stashed 
                                      the solution and rhs, these are 
                                      never touched by the code, only 
                                      passed back to the user */ 
  double     *res_hist;            /* If !0 stores residual at iterations*/
  int        res_hist_len;         /* current size of residual history array */
  int        res_hist_max;         /* actual amount of data in residual_history */
  PetscTruth res_hist_reset;       /* reset history to size zero for each new solve */

  /* --------User (or default) routines (most return -1 on error) --------*/
  int  (*monitor[MAXKSPMONITORS])(KSP,int,double,void*); /* returns control to user after */
  int  (*monitordestroy[MAXKSPMONITORS])(void*);         /* */
  void *monitorcontext[MAXKSPMONITORS];                  /* residual calculation, allows user */
  int  numbermonitors;                                   /* to, for instance, print residual norm, etc. */

  int  (*converged)(KSP,int,double,KSPConvergedReason*,void*);
  void       *cnvP; 

  PC         B;

  void       *data;                      /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  int        nwork;
  Vec        *work;

  int        setupcalled;

  int        its;       /* number of iterations so far computed */
  PetscTruth avoidnorms; /* does not compute residual norms when possible */

  PetscTruth transpose_solve;    /* solve transpose system instead */
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
EXTERN int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
EXTERN int KSPUnwindPreconditioner(KSP,Vec,Vec);

/*
       These allow the various Krylov methods to apply to either the linear system
    or its transpose.
*/
#define KSP_MatMult(ksp,A,x,y) (!ksp->transpose_solve) ?  MatMult(A,x,y) : MatMultTranspose(A,x,y) 
#define KSP_MatMultTranspose(ksp,A,x,y) (!ksp->transpose_solve) ?  MatMultTranspose(A,x,y) : MatMult(A,x,y) 
#define KSP_PCApply(ksp,A,x,y) (!ksp->transpose_solve) ?  PCApply(A,x,y) : PCApplyTranspose(A,x,y) 
#define KSP_PCApplyTranspose(ksp,A,x,y) (!ksp->transpose_solve) ?  PCApplyTranspose(A,x,y) : PCApply(A,x,y) 
#define KSP_PCApplyBAorAB(ksp,pc,side,x,y,work) (!ksp->transpose_solve) ? \
         PCApplyBAorAB(pc,side,x,y,work) : PCApplyBAorABTranspose(pc,side,x,y,work)

#endif
