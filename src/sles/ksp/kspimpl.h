/* $Id: kspimpl.h,v 1.40 1999/04/21 18:17:50 bsmith Exp balay $ */

#ifndef _KSPIMPL
#define _KSPIMPL

#include "ksp.h"

typedef struct _KSPOps *KSPOps;

struct _KSPOps {
  int  (*buildsolution)(KSP,Vec,Vec*);       /* Returns a pointer to the solution, or
                                                calculates the solution in a 
				                user-provided area. */
  int  (*buildresidual)(KSP,Vec,Vec,Vec*);   /* Returns a pointer to the residual, or
				                calculates the residual in a 
				                user-provided area.  */
  int  (*solve)(KSP,int*);                   /* actual solver */
  int  (*solvetrans)(KSP,int*);              /* actual solver */
  int  (*setup)(KSP);
  int  (*setfromoptions)(KSP);
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
  int max_it,                      /* maximum number of iterations */
      guess_zero,                  /* flag for whether initial guess is 0 */
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

  Vec vec_sol, vec_rhs;            /* pointer to where user has stashed 
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

  int  (*converged)(KSP,int,double,void*);
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
};

#define KSPLogResidualHistory(ksp,norm) \
    {if (ksp->res_hist && ksp->res_hist_max > ksp->res_hist_len) \
     ksp->res_hist[ksp->res_hist_len++] = norm;}

#define KSPMonitor(ksp,it,rnorm) \
        { int _ierr,_i,_im = ksp->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr = (*ksp->monitor[_i])(ksp,it,rnorm,ksp->monitorcontext[_i]);CHKERRQ(_ierr); \
	  } \
	}

extern int KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
extern int KSPDefaultDestroy(KSP);
extern int KSPDefaultGetWork(KSP,int);
extern int KSPDefaultFreeWork(KSP);
extern int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
extern int KSPUnwindPreconditioner(KSP,Vec,Vec);

#endif
