/* $Id: kspimpl.h,v 1.16 1996/01/12 03:51:34 bsmith Exp bsmith $ */

#ifndef _KSPIMPL
#define _KSPIMPL

#include "ksp.h"
/*
   Defines the KSP data structure.
*/
struct _KSP {
  PETSCHEADER
  /*------------------------- User parameters--------------------------*/
  int max_it,                      /* maximum number of iterations */
      guess_zero,                  /* flag for whether initial guess is 0 */
      calc_eigs,                   /* calculate extreme eigenvalues */
      calc_res,                    /* calculate residuals at each iteration*/
      use_pres;                    /* use preconditioned residual */
  PCSide pc_side;                  /* flag for left, right, or symmetric 
                                      preconditioning */
  double rtol,                     /* relative tolerance */
         atol,                     /* absolute tolerance */
         ttol,                     /* (not set by user)  */
         divtol;                   /* divergence tolerance */
  double rnorm0;                   /* initial residual norm 
				      (used for divergence testing) */

  Vec vec_sol, vec_rhs   ;         /* pointer to where user has stashed 
                                      the solution and rhs, these are 
                                      never touched by the code, only 
                                      passed back to the user */ 
  double *residual_history;        /* If !0 stores residual at iterations*/
  int    res_hist_size;            /* Size of residual history array */
  int    res_act_size;             /* actual amount of data in residual_history
				      */

  /* --------User (or default) routines (most return -1 on error) --------*/
  int  (*monitor)(KSP,int,double,void*); /* returns control to user after
                                      residual calculation, allows user to, for 
                                      instance, print residual norm, etc. */
  int (*converged)(KSP,int,double,void*);
  int (*buildsolution)(KSP,Vec,Vec*);  /* Returns a pointer to the solution, or
				      calculates the solution in a 
				      user-provided area. */
  int (*buildresidual)(KSP,Vec,Vec,Vec*); /* Returns a pointer to the residual, or
				      calculates the residual in a 
				      user-provided area.  */
  int (*adjust_work_vectors)(KSP,Vec*,int); /* should pre-allocate the vectors*/
  PC  B;    /* fit this framework just fine */

  /*------------ Major routines which act on KSPCtx-----------------*/
  int  (*solver)(KSP,int*);      /* actual solver */
  int  (*setup)(KSP);
  int  (*adjustwork)(KSP);
  void *data;                      /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  int  nwork;
  Vec *work;

  /* ------------Contexts for the user-defined functions-------------- */
  void *monP,       /* User Monitor */
       *cnvP;       /* Convergence tester */
  int setupcalled;
};

#define MONITOR(ksp,rnorm,it) if (ksp->monitor) { \
                                (*ksp->monitor)(ksp,it,rnorm,ksp->monP);\
                              }

extern int KSPCreate_Richardson(KSP);
extern int KSPCreate_Chebychev(KSP);
extern int KSPCreate_CG(KSP);
extern int KSPCreate_GMRES(KSP);
extern int KSPCreate_TCQMR(KSP);
extern int KSPCreate_BCGS(KSP);
extern int KSPCreate_CGS(KSP);
extern int KSPCreate_TFQMR(KSP);
extern int KSPCreate_LSQR(KSP);
extern int KSPCreate_PREONLY(KSP);
extern int KSPCreate_CR(KSP);
extern int KSPCreate_QCG(KSP);

extern int KSPiDefaultAdjustWork(KSP);
extern int KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
extern int KSPiDefaultDestroy(PetscObject);
extern int KSPCheckDef(KSP);
extern int KSPiDefaultGetWork(KSP,int);
extern int KSPiDefaultFreeWork(KSP);
extern int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
extern int KSPUnwindPre(KSP,Vec,Vec);

#endif
