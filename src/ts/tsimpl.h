  /* $Id: tsimpl.h,v 1.25 2001/09/07 20:12:01 bsmith Exp $ */

#ifndef __TSIMPL_H
#define __TSIMPL_H
#include "petscts.h"

/*
    Timesteping context. 
      General case: U_t = F(t,U) <-- the right-hand-side function
      Linear  case: U_t = A(t) U <-- the right-hand-side matrix
      Linear (no time) case: U_t = A U <-- the right-hand-side matrix
*/

/*
     Maximum number of monitors you can run with a single TS
*/
#define MAXTSMONITORS 5 

struct _TSOps {
  int (*rhsmatrix)(TS, PetscReal, Mat *, Mat *, MatStructure *, void *),
      (*rhsfunction)(TS, PetscReal, Vec, Vec, void *),
      (*rhsjacobian)(TS, PetscReal, Vec, Mat *, Mat *, MatStructure *, void *),
      (*applymatrixbc)(TS, Mat, Mat, void *),
      (*rhsbc)(TS, PetscReal, Vec, void *),
      (*applyrhsbc)(TS, Vec, void *),
      (*applysolbc)(TS, Vec, void *),
      (*prestep)(TS),
      (*update)(TS, double, double *),
      (*poststep)(TS),
      (*reform)(TS),
      (*reallocate)(TS),
      (*setup)(TS),
      (*step)(TS,int *, PetscReal *),
      (*setfromoptions)(TS),
      (*printhelp)(TS, char *),
      (*destroy)(TS),
      (*view)(TS, PetscViewer);
};

struct _p_TS {
  PETSCHEADER(struct _TSOps)
  TSProblemType problem_type;
  Vec           vec_sol, vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  int  (*monitor[MAXTSMONITORS])(TS,int,PetscReal,Vec,void*); /* returns control to user after */
  int  (*mdestroy[MAXTSMONITORS])(void*);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  int  numbermonitors;                                 /* to, for instance, print residual norm, etc. */

  /* Identifies this as a grid TS structure */
  PetscTruth  isGTS;                                 /* This problem arises from an underlying grid */
  PetscTruth *isExplicit;                            /* Indicates which fields have explicit time dependence */
  int        *Iindex;                                /* The index of the identity for each time dependent field */

  /* ---------------------Linear Iteration---------------------------------*/
  SLES sles;
  Mat  A, B;                                         /* user provided matrix and preconditioner */

  /* ---------------------Nonlinear Iteration------------------------------*/
  SNES  snes;
  void *funP;
  void *jacP;
  void *bcP;


  /* --- Data that is unique to each particular solver --- */
  int   setupcalled;            /* true if setup has been called */
  void *data;                   /* implementationspecific data */
  void *user;                   /* user context */

  /* ------------------  Parameters -------------------------------------- */
  int       max_steps;              /* max number of steps */
  PetscReal max_time;               /* max time allowed */
  PetscReal time_step;              /* current time increment */
  PetscReal time_step_old;          /* previous time increment */
  PetscReal initial_time_step;      /* initial time increment */
  int       steps;                  /* steps taken so far */
  PetscReal ptime;                  /* time taken so far */
  int       linear_its;             /* total number of linear solver iterations */
  int       nonlinear_its;          /* total number of nonlinear solver iterations */

  /* ------------------- Default work-area management ------------------ */
  int  nwork;              
  Vec *work;
};

EXTERN int TSMonitor(TS,int,PetscReal,Vec);
EXTERN int TSComputeRHSBoundaryConditions(TS,PetscReal,Vec);

#endif
