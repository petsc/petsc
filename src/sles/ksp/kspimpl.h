
#ifndef _KSPIMPL
#define _KSPIMPL

#include "ptscimpl.h"
#include "ksp.h"

/*
   Iterative method context.
*/
struct _KSP {
  PETSCHEADER
  /*------------------------- User parameters--------------------------*/
  int max_it,                      /* maximum number of iterations */
      right_pre,                   /* flag for right preconditioning */
      guess_zero,                  /* flag for whether initial guess is 0 */
      calc_eigs,                   /* calculate extreme eigenvalues */
      calc_res,                    /* calculate residuals at each iteration*/
      use_pres;                    /* use preconditioned residual */

  double rtol,                     /* relative tolerance */
         atol,                     /* absolute tolerance */
         ttol,                     /* (not set by user)  */
         divtol;                   /* divergence tolerance */
  double rnorm0;                   /* initial residual norm 
				      (used for divergence testing) */

  Vec vec_sol, vec_rhs;         /* pointer to where user has stashed 
                                      the solution and rhs, these are 
                                      never touched by the code, only 
                                      passed back to the user */ 
  double *residual_history;        /* If !0 stores residual at iterations*/
  int    res_hist_size;            /* Size of residual history array */
  int    res_act_size;             /* actual amount of data in residual_history
				      */

  /* --------User (or default) routines (most return -1 on error) --------*/
  int  (*usr_monitor)(KSP,int,double,void*); /* returns control to user after
                                      residual calculation, allows user to, for 
                                      instance, print residual norm, etc. */
  int (*converged)(KSP,int,double,void*);
  int (*BuildSolution)(KSP,Vec,Vec*);  /* Returns a pointer to the solution, or
				      calculates the solution in a 
				      user-provided area. */
  int (*BuildResidual)(KSP,Vec,Vec,Vec*); /* Returns a pointer to the residual, or
				      calculates the residual in a 
				      user-provided area.  */
  int (*adjust_work_vectors)(KSP,Vec*,int); /* should pre-allocate the vectors*/
  PC  B;    /* fit this framework just fine */

  /*------------ Major routines which act on KSPCtx-----------------*/
  int  (*solver)(KSP,int*);      /* actual solver */
  int  (*setup)(KSP);
  int  (*adjustwork)(KSP);
  void *MethodPrivate;          /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  int  nwork;
  Vec *work;

  /* ------------Contexts for the user-defined functions-------------- */
  void *monP,       /* User Monitor */
       *cnvP;       /* Convergence tester */
  int setupcalled;
  /* ------Prefix for Names of setable parameters----------------------*/
  char  *prefix;
};

/*  
    Ugly macros used by implementations, should be phased out
*/

#define CONVERGED(itP,rn,it) (cerr=(*itP->converged)(itP,it,rn,itP->cnvP))
#define RCONV(itP,it)           ((cerr>0)?it:-it)

#define MONITOR(itP,rnorm,it) if (itP->usr_monitor) { \
                                (*itP->usr_monitor)(itP,it,rnorm,itP->monP);\
                              }


#if !defined(MAX)
#define  MAX(a,b)           ((a) > (b) ? (a) : (b))
#endif

int KSPCreate_Richardson(KSP);
int KSPCreate_Chebychev(KSP);
int KSPCreate_CG(KSP);
int KSPCreate_GMRES(KSP);
int KSPCreate_TCQMR(KSP);
int KSPCreate_BCGS(KSP);
int KSPCreate_CGS(KSP);
int KSPCreate_TFQMR(KSP);
int KSPCreate_LSQR(KSP);
int KSPCreate_PREONLY(KSP);
int KSPCreate_CR(KSP);

int KSPiDefaultAdjustWork(KSP);
int KSPDefaultBuildSolution(KSP,Vec,Vec*);
int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
int KSPiDefaultDestroy(PetscObject);
int KSPCheckDef(KSP);
int KSPiDefaultGetWork(KSP,int);
int KSPiDefaultFreeWork(KSP);
int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
int KSPUnwindPre(KSP,Vec,Vec);

#endif
