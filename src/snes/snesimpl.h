/* $Id: snesimpl.h,v 1.24 1995/08/21 16:03:02 curfman Exp curfman $ */

#ifndef __SNESIMPL_H
#define __SNESIMPL_H
#include "snes.h"

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

  int      (*computeumfunction)(SNES,Vec,double*,void*);
  double   fc;                /* function value */
  void     *umfunP;           /* function pointer */
  SNESType method_class;
  double   deltatol;          /* trust region convergence tolerance */
  double   fmin;              /* minimum tolerance for function value */
  int      set_method_called; /* flag indicating set_method has been called */
/*
   These are REALLY ugly and don't belong here, but since they must 
  be destroyed at the conclusion we have to put them somewhere.
 */
  int      ksp_ewconv;        /* flag indicating Eisenstat-Walker KSP 
                                 convergence criteria */
  void     *kspconvctx;       /* KSP convergence context */
  Mat      mfshell;           /* MatShell for matrix-free from command line */
};

/* Context for Eisenstat-Walker convergence criteria for KSP solvers */
typedef struct {
  int    version;             /* flag indicating version 1 or 2 of test */
  double rtol_0;              /* initial rtol */
  double rtol_last;           /* last rtol */
  double rtol_max;            /* maximum rtol */
  double gamma;               /* mult. factor for version 2 rtol computation */
  double alpha;               /* power for version 2 rtol computation */
  double alpha2;              /* power for safeguard */
  double threshold;           /* threshold for imposing safeguard */
  double lresid_last;         /* linear residual from last iteration */
  double norm_last;           /* function norm from last iteration */
} SNES_KSP_EW_ConvCtx;

int SNES_KSP_EW_Converged_Private(KSP,int,double,void*);
int SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES,KSP);
int SNESScaleStep_Private(SNES,Vec,double*,double*,double*,double*);

#endif
