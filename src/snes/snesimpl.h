/* $Id: snesimpl.h,v 1.59 2001/08/21 21:03:47 bsmith Exp $ */

#ifndef __SNESIMPL_H
#define __SNESIMPL_H

#include "petscsnes.h"

/*
   Nonlinear solver context
 */
#define MAXSNESMONITORS 5

struct _p_SNES {
  PETSCHEADER(int)

  /* Identifies this as a grid SNES structure */
  PetscTruth  isGSNES;                          /* This problem arises from an underlying grid */

  /*  ------------------------ User-provided stuff -------------------------------*/
  void  *user;		                        /* user-defined context */

  Vec   vec_sol,vec_sol_always;                 /* pointer to solution */
  Vec   vec_sol_update_always;                  /* pointer to solution update */

  int   (*computefunction)(SNES,Vec,Vec,void*); /* function routine */
  Vec   vec_func,vec_func_always;               /* pointer to function */
  void  *funP;                                  /* user-defined function context */

  int   (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
  Mat   jacobian;                               /* Jacobian matrix */
  Mat   jacobian_pre;                           /* preconditioner matrix */
  void  *jacP;                                  /* user-defined Jacobian context */
  KSP  ksp;                                   /* linear solver context */

  int   (*computescaling)(Vec,Vec,void*);       /* scaling routine */
  Vec   scaling;                                /* scaling vector */
  void  *scaP;                                  /* scaling context */

  /* ------------------------Boundary conditions-----------------------------------*/
  int (*applyrhsbc)(SNES, Vec, void *);         /* Applies boundary conditions to the rhs */
  int (*applysolbc)(SNES, Vec, void *);         /* Applies boundary conditions to the solution */

  /* ------------------------Time stepping hooks-----------------------------------*/
  int (*update)(SNES, int);                     /* General purpose function for update */

  /* ---------------- PETSc-provided (or user-provided) stuff ---------------------*/

  int   (*monitor[MAXSNESMONITORS])(SNES,int,PetscReal,void*); /* monitor routine */
  int   (*monitordestroy[MAXSNESMONITORS])(void*);          /* monitor context destroy routine */
  void  *monitorcontext[MAXSNESMONITORS];                   /* monitor context */
  int   numbermonitors;                                     /* number of monitors */
  int   (*converged)(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);      /* convergence routine */
  void  *cnvP;	                                            /* convergence context */
  SNESConvergedReason reason;

  /* --- Routines and data that are unique to each particular solver --- */

  int   (*setup)(SNES);             /* routine to set up the nonlinear solver */
  int   setupcalled;                /* true if setup has been called */
  int   (*solve)(SNES);             /* actual nonlinear solver */
  int   (*setfromoptions)(SNES);    /* sets options from database */
  int   (*printhelp)(SNES,char*);   /* prints help info */
  void  *data;                      /* implementation-specific data */

  /* --------------------------  Parameters -------------------------------------- */

  int      max_its;            /* max number of iterations */
  int      max_funcs;          /* max number of function evals */
  int      nfuncs;             /* number of function evaluations */
  int      iter;               /* global iteration number */
  int      linear_its;         /* total number of linear solver iterations */
  PetscReal   norm;            /* residual norm of current iterate */
  PetscReal   rtol;            /* relative tolerance */
  PetscReal   atol;            /* absolute tolerance */
  PetscReal   xtol;            /* relative tolerance in solution */
  PetscReal   deltatol;        /* trust region convergence tolerance */

  /* ------------------------ Default work-area management ---------------------- */

  int      nwork;              
  Vec      *work;

  /* ------------------------- Miscellaneous Information ------------------------ */

  PetscReal     *conv_hist;         /* If !0, stores function norm (or
                                    gradient norm) at each iteration */
  int        *conv_hist_its;     /* linear iterations for each Newton step */
  int        conv_hist_len;      /* size of convergence history array */
  int        conv_hist_max;      /* actual amount of data in conv_history */
  PetscTruth conv_hist_reset;    /* reset counter for each new SNES solve */
  int        numFailures;        /* number of unsuccessful step attempts */
  int        maxFailures;        /* maximum number of unsuccessful step attempts */

 /*
   These are REALLY ugly and don't belong here, but since they must 
  be destroyed at the conclusion we have to put them somewhere.
 */
  PetscTruth  ksp_ewconv;        /* flag indicating use of Eisenstat-Walker KSP convergence criteria */
  void        *kspconvctx;       /* KSP convergence context */

  PetscReal      ttol;           /* used by default convergence test routine */

  Vec         *vwork;            /* more work vectors for Jacobian approx */
  int         nvwork;
  int        (*destroy)(SNES);
  int        (*view)(SNES,PetscViewer);
};

/* Context for Eisenstat-Walker convergence criteria for KSP solvers */
typedef struct {
  int       version;             /* flag indicating version 1 or 2 of test */
  PetscReal rtol_0;              /* initial rtol */
  PetscReal rtol_last;           /* last rtol */
  PetscReal rtol_max;            /* maximum rtol */
  PetscReal gamma;               /* mult. factor for version 2 rtol computation */
  PetscReal alpha;               /* power for version 2 rtol computation */
  PetscReal alpha2;              /* power for safeguard */
  PetscReal threshold;           /* threshold for imposing safeguard */
  PetscReal lresid_last;         /* linear residual from last iteration */
  PetscReal norm_last;           /* function norm from last iteration */
} SNES_KSP_EW_ConvCtx;

#define SNESLogConvHistory(snes,res,its) \
  { if (snes->conv_hist && snes->conv_hist_max > snes->conv_hist_len) \
    { if (snes->conv_hist)     snes->conv_hist[snes->conv_hist_len]     = res; \
      if (snes->conv_hist_its) snes->conv_hist_its[snes->conv_hist_len] = its; \
      snes->conv_hist_len++;\
    }}

#define SNESMonitor(snes,it,rnorm) \
        { int _ierr,_i,_im = snes->numbermonitors; \
          for (_i=0; _i<_im; _i++) {\
            _ierr = (*snes->monitor[_i])(snes,it,rnorm,snes->monitorcontext[_i]);CHKERRQ(_ierr); \
	  } \
	}

int SNES_KSP_EW_Converged_Private(KSP,int,PetscReal,KSPConvergedReason*,void*);
int SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES,KSP);
int SNESScaleStep_Private(SNES,Vec,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

#endif
