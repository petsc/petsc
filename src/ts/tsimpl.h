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

struct _p_TS {
  PETSCHEADER(int)

  TSProblemType problem_type;

  Vec           vec_sol,vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  int  (*monitor[MAXTSMONITORS])(TS,int,PetscReal,Vec,void*); /* returns control to user after */
  int  (*mdestroy[MAXTSMONITORS])(void*);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  int  numbermonitors;                       /* to, for instance, print residual norm, etc. */

  int           (*rhsmatrix)(TS,PetscReal,Mat*,Mat*,MatStructure *,void*);
  Mat           A,B;        /* user provided matrix and preconditioner */

  int           (*rhsfunction)(TS,PetscReal,Vec,Vec,void*); 
  void          *funP;
  int           (*rhsjacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure *,void*);
  void          *jacP;
  int           (*rhsbc)(TS,PetscReal,Vec,void*); 
  void          *bcP;

  /* ---------Inner nonlinear or linear solvers ---------------------------*/

  SNES          snes;
  SLES          sles;

  /* --- Routines and data that are unique to each particular solver --- */

  int           (*setup)(TS);            /* sets up the nonlinear solver */
  int           setupcalled;            /* true if setup has been called */
  int           (*step)(TS,int*,PetscReal*); /* stepping routine */      
  int           (*setfromoptions)(TS);    /* sets options from database */
  void          *data;                    /* implementationspecific data */

  void          *user;                    /* user context */

  /* ------------------  Parameters -------------------------------------- */

  int           max_steps;          /* max number of steps */
  PetscReal     max_time;
  PetscReal     time_step;
  PetscReal     initial_time_step;
  int           steps;              /* steps taken so far */
  PetscReal     ptime;              /* time taken so far */
  int           linear_its;         /* total number of linear solver iterations */
  int           nonlinear_its;      /* total number of nonlinear solver iterations */

  /* ------------------- Default work-area management ------------------ */

  int           nwork;              
  Vec           *work;
  int           (*destroy)(TS);
  int           (*view)(TS,PetscViewer);
};

EXTERN int TSMonitor(TS,int,PetscReal,Vec);
EXTERN int TSComputeRHSBoundaryConditions(TS,PetscReal,Vec); 

#endif
