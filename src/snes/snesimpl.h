/* $Id: snesimpl.h,v 1.3 1995/04/17 03:33:03 curfman Exp bsmith $ */

#ifndef __SNESIMPL_H
#define __SNESIMPL_H
#include "snes.h"
#include "ptscimpl.h"


/*
   Nonlinear solver context
 */
struct _SNES {
  PETSCHEADER
  char *prefix;

  /*  ----------------- User provided stuff ------------------------*/
  void  *user;		             /* User context */

  int   (*ComputeInitialGuess)(Vec,void*); /* Calculates an initial guess */
  Vec   vec_sol;                     /* Pointer to solution */
  void  *gusP;

  int   (*ComputeResidual)(Vec,Vec,void *);
  Vec   vec_res;                    /* Pointer to function or gradient */
  void  *resP;
  int   rsign;                      /* sign (+/-)  of residual */

  int   (*ComputeJacobian)(Vec,Mat*,void*);
  Mat   jacobian;                   /* Jacobian (or Hessian) matrix context */
  void  *jacP;
  SLES  sles;

  int   (*ComputeScaling)(Vec,Vec,void*);
  Vec   scaling;
  void  *scaP;

  /* ---------------- Petsc (or user) Provided stuff ---------------------*/
  int   (*Monitor)(SNES,int,double,void*);   
  void  *monP;		
  int   (*Converged)(SNES,double,double,double,void*);     
  void  *cnvP;		

  /* --- Routines and data that are unique to each particular solver --- */

  int   (*Setup)(SNES);         /* Sets up the nonlinear solver */
  int   (*Solver)(SNES,int*);        /* Actual nonlinear solver */
  int   (*SetFromOptions)(SNES);
  int   (*PrintHelp)(SNES);
  void  *data;     

  /* ------------------  Parameters -------------------------------------- */

  int      max_its;            /* Max number of iterations */
  int      max_funcs;          /* Max number of function evals (NLM only) */
  int      max_resids;         /* Max number of residual evaluations */
  int      nresids;            /* Number of residual evaluations */
  int      iter;               /* Global iteration number */
  double   norm;               /* Residual norm of current iterate (NLE)
				  or gradient norm of current iterate (NLM) */
  double   rtol;               /* Relative tolerance */
  double   atol;               /* Absolute tolerance */
  double   xtol;               /* Relative tolerance in solution */
  double   trunctol;

  /* ------------------- Default work-area management ------------------ */

  int      nwork;              
  Vec      *work;

  /* -------------------- Miscellaneous Information --------------------- */
  double   *conv_hist;         /* If !0, stores residual norm (NLE) or
				  gradient norm (NLM) at each iteration */
  int      conv_hist_len;      /* Amount of convergence history space */
};

#if !defined(MAX)
#define MAX(a,b)     ((a) > (b) ? (a) : (b))
#define MIN(a,b)     ((a) < (b) ? (a) : (b))
#endif

extern int SNESComputeInitialGuess(SNES,Vec);
extern int SNESComputeFunction(SNES,Vec, Vec);

#endif
