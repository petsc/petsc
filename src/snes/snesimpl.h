/* $Id: snesimpl.h,v 1.34 1996/03/23 18:36:55 bsmith Exp curfman $ */

#ifndef __SNESIMPL_H
#define __SNESIMPL_H
#include "draw.h"
#include "snes.h"

/*
   Nonlinear solver context
 */

struct _SNES {
  PETSCHEADER

  /*  ----------------- User provided stuff ------------------------*/
  void  *user;		            /* user context */

  Vec   vec_sol,vec_sol_always;     /* pointer to solution */
  Vec   vec_sol_update_always;      /* pointer to solution update */

  int   (*computefunction)(SNES,Vec,Vec,void*);  /* function routine */
  Vec   vec_func,vec_func_always;   /* Pointer to function or gradient */
  void  *funP;                      /* user function context */

  int   (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
  Mat   jacobian;                   /* Jacobian (or Hessian) matrix */
  Mat   jacobian_pre;               /* preconditioner matrix */
  void  *jacP;                      /* user Jacobian context */
  SLES  sles;                       /* linear solver context */

  int   (*computescaling)(Vec,Vec,void*);  /* scaling routine */
  Vec   scaling;                           /* scaling vector */
  void  *scaP;                             /* scaling context */

  /* ---------------- Petsc (or user) Provided stuff ---------------------*/
  int   (*monitor)(SNES,int,double,void*); /* monitor routine */
  void  *monP;		                   /* monitor routine context */
  int   (*converged)(SNES,double,double,double,void*); /* converg. routine */
  void  *cnvP;		                   /* convergence context */

  /* --- Routines and data that are unique to each particular solver --- */

  int   (*setup)(SNES);             /* sets up the nonlinear solver */
  int   setup_called;               /* true if setup has been called */
  int   (*solve)(SNES,int*);        /* actual nonlinear solver */
  int   (*setfromoptions)(SNES);    /* sets options from database */
  int   (*printhelp)(SNES,char*);   /* prints help info */
  void  *data;                      /* implementationspecific data */

  /* ------------------  Parameters -------------------------------------- */

  int      max_its;            /* max number of iterations */
  int      max_funcs;          /* max number of function evals (NLM only) */
  int      nfuncs;             /* number of function evaluations */
  int      iter;               /* global iteration number */
  double   norm;               /* residual norm of current iterate
				  (or gradient norm of current iterate) */
  double   rtol;               /* relative tolerance */
  double   atol;               /* absolute tolerance */
  double   xtol;               /* relative tolerance in solution */
  double   trunctol;           /* truncation tolerance */

  /* ------------------- Default work-area management ------------------ */

  int      nwork;              
  Vec      *work;

  /* -------------------- Miscellaneous Information --------------------- */
  double   *conv_hist;         /* If !0, stores residual norm (or
				  gradient norm) at each iteration */
  int      conv_hist_len;      /* amount of convergence history space */
  int      nfailures;          /* number of unsuccessful step attempts */

  /* ---------------------------- SUMS Data ---------------------------- */
  /* unconstrained minimization info ... For now we share everything else
     with the nonlinear equations code.  We should find a better way to deal 
     with this; the naming conventions are confusing.  Perhaps use unions? */

  int             (*computeumfunction)(SNES,Vec,double*,void*);
  double          fc;                /* function value */
  void            *umfunP;           /* function pointer */
  SNESProblemType method_class;
  double          deltatol;          /* trust region convergence tolerance */
  double          fmin;              /* minimum tolerance for function value */
  int             set_method_called; /* flag indicating set_method has been called */
/*
   These are REALLY ugly and don't belong here, but since they must 
  be destroyed at the conclusion we have to put them somewhere.
 */
  int      ksp_ewconv;        /* flag indicating Eisenstat-Walker KSP 
                                 convergence criteria */
  void     *kspconvctx;       /* KSP convergence context */
  Mat      mfshell;           /* MatShell for matrix-free from command line */

  double   ttol;              /* used by default convergence test routine */

  DrawLG   xmonitor;     /* Where -snes_xmonitor context is stashed */
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

#define SNESMonitor(snes,it,rnorm) \
        if (snes->monitor) { \
          int _ierr; \
          _ierr = (*snes->monitor)(snes,it,rnorm,snes->monP); \
          CHKERRQ(_ierr); \
        }

int SNES_KSP_EW_Converged_Private(KSP,int,double,void*);
int SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES,KSP);
int SNESScaleStep_Private(SNES,Vec,double*,double*,double*,double*);

#endif
