
#ifndef _KSPIMPL
#define _KSPIMPL

#include "ptscimpl.h"
#include "ksp.h"

#define KSP_COOKIE         0x202020

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
  int  (*usr_monitor)();            /* returns control to user after residual
                                      calculation, allows user to, for 
                                      instance, print residual norm, etc. */
  int (*converged)();
  int (*BuildSolution)();        /* Returns a pointer to the solution, or
				      calculates the solution in a 
				      user-provided area. */
  int (*BuildResidual)();        /* Returns a pointer to the residual, or
				      calculates the residual in a 
				      user-provided area.  */
  int (*adjust_work_vectors)();    /* should pre-allocate the vectors*/
  int  (*amult)(),                 /* application of matrix multiply A*/
       (*binv)(),                  /* application of preconditioner B*/
       (*matop)();                 /* application of AB or BA; if not set,
			                         amult and binv will be used */
  int  (*tamult)(),                /* transposes of the same         */
       (*tbinv)(),                  
       (*tmatop)();                

  /*------------ Major routines which act on KSPCtx-----------------*/
  KSPMETHOD  method;            /* type of solver */
  int  (*solver)();            /* actual solver */
  int  (*setup)();
  int  (*adjustwork)();
  void *MethodPrivate;          /* holder for misc stuff associated 
                                   with a particular iterative solver */

  /* ----------------Default work-area management -------------------- */
  int  nwork;
  Vec *work;

  /* ----------------Keep track of the amount of work----------------- */
  int  nmatop, namult, nbinv, nvectors, nscalar;

  /* ------------Contexts for the user-defined functions-------------- */
  void *amultP,     /* Amult operations */
       *binvP,      /* Binv operations */
       *monP,       /* User Monitor */
       *cnvP;       /* Convergence tester */
  int setupcalled;
  /* ------------Names of setable parameters--------------------------*/
  char      *namemethod, *namemax_it, *nameright_pre, *nameuse_pres,
            *namertol, *nameatol, *namedivtol, *namecalc_eigs,
            *namecalc_res;
};

/*  
    Ugly macros used by implementations, should be phased out
*/

#define CONVERGED(itP,rn,it) (cerr=(*itP->converged)(itP,it,rn,itP->cnvP))
#define RCONV(itP,it)           ((cerr>0)?it:-it)
#define PRE(itP,x,y)    {if (itP->binv) { (*itP->binv)(itP->binvP,x,y);} \
                        else { VecCopy(x,y); }}
#define TPRE(itP,x,y)   {if (itP->tbinv) { (*itP->tbinv)(itP->binvP,x,y);}\
                        else {VecCopy(x,y); }}
#define MATOP(itP,x,y,temp) {if (itP->matop) {\
                            (*itP->matop)(itP->amultP,itP->binvP,x,y);}\
                            else {\
                              if (itP->right_pre) {  \
				PRE(itP,x,temp);MM(itP,temp,y);} \
			      else { \
                                MM(itP,x,temp);PRE(itP,temp,y);}\
                              }}
#define TMATOP(itP,x,y,temp)  {  if (itP->tmatop) {\
                              (*itP->tmatop)(itP->amultP,itP->binvP,x,y);}\
                            } \
                            else {                      \
			      if (!itP->right_pre) {  \
                                TPRE(itP,x,temp);TMM(itP,temp,y);} \
			      else {            \
			        TMM(itP,x,temp);TPRE(itP,temp,y);}\
                              }}
#define MONITOR(itP,rnorm,it) if (itP->usr_monitor) { \
                                (*itP->usr_monitor)(itP,it,rnorm,itP->monP);\
                              }


#define MM(itP,x,y)            (*itP->amult)(itP->amultP,x,y) 
#define TMM(itP,x,y)           (*itP->tamult)(itP->amultP,x,y) 

#if !defined(MAX)
#define  MAX(a,b)           ((a) > (b) ? (a) : (b))
#endif

int KSPiRichardsonCreate();
int KSPiChebychevCreate();
int KSPiCGCreate();
int KSPiGMRESCreate();
int KSPiTCQMRCreate();
int KSPiBCGSCreate();
int KSPiCGSCreate();
int KSPiTFQMRCreate();
int KSPiLSQRCreate();
int KSPiPREONLYCreate();
int KSPiCRCreate();

int   KSPiDefaultAdjustWork();
int  KSPDefaultBuildSolution();
int KSPDefaultBuildResidual();
int     KSPiDefaultDestroy();

#endif
