/* $Id: snesimpl.h,v 1.15 1995/07/16 21:59:22 curfman Exp bsmith $ */

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

  int   (*computeinitialguess)(SNES,Vec,void*); /* Calculates initial guess */
  Vec   vec_sol,vec_sol_always;     /* Pointer to solution */
  Vec   vec_sol_update_always;      /* Pointer to solution update */
  void  *gusP;

  int   (*computefunction)(SNES,Vec,Vec,void*);
  Vec   vec_func,vec_func_always;   /* Pointer to function or gradient */
  void  *funP;
  int   rsign;                      /* sign (+/-)  of residual */

  int   (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
  Mat   jacobian;                   /* Jacobian (or Hessian) matrix context */
  Mat   jacobian_pre;
  void  *jacP;
  SLES  sles;

  int   (*computescaling)(Vec,Vec,void*);
  Vec   scaling;
  void  *scaP;

  /* ---------------- Petsc (or user) Provided stuff ---------------------*/
  int   (*monitor)(SNES,int,double,void*);   
  void  *monP;		
  int   (*converged)(SNES,double,double,double,void*);     
  void  *cnvP;		

  /* --- Routines and data that are unique to each particular solver --- */

  int   (*setup)(SNES);         /* Sets up the nonlinear solver */
  int   (*solve)(SNES,int*);        /* Actual nonlinear solver */
  int   (*setfromoptions)(SNES);
  int   (*printhelp)(SNES);
  void  *data;     

  /* ------------------  Parameters -------------------------------------- */

  int      max_its;            /* Max number of iterations */
  int      max_funcs;          /* Max number of function evals (NLM only) */
  int      nfuncs;             /* Number of residual evaluations */
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
  int      nfailures;          /* Number of unsuccessful step attempts */

  /* ---------------------------- SUMS Stuff ---------------------------- */
  /* unconstrained minimization stuff ... For now we share everything else
     with the nonlinear equations code.  We should find a better way to deal 
     with this; the naming conventions are confusing.   Perhaps use unions? */

  int      (*computeumfunction)(SNES,Vec,Scalar*,void*);
  Scalar   fc;
  void     *umfunP;
  SNESType method_class;
  double   deltatol;          /* trust region convergence tolerance */
  double   fmin;              /* minimum tolerance for function value */
};

#if !defined(MAX)
#define MAX(a,b)     ((a) > (b) ? (a) : (b))
#define MIN(a,b)     ((a) < (b) ? (a) : (b))
#endif

#endif
