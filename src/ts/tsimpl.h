  /* $Id: tsimpl.h,v 1.19 2000/01/11 21:02:54 bsmith Exp balay $ */

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
  int  (*monitor[MAXTSMONITORS])(TS,int,double,Vec,void*); /* returns control to user after */
  int  (*mdestroy[MAXTSMONITORS])(void*);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  int  numbermonitors;                       /* to, for instance, print residual norm, etc. */

  int           (*rhsmatrix)(TS,double,Mat*,Mat*,MatStructure *,void*);
  Mat           A,B;        /* user provided matrix and preconditioner */
  Mat           Ashell;     /* if user provided a Shell matrix */

  int           (*rhsfunction)(TS,double,Vec,Vec,void*); 
  void          *funP;
  int           (*rhsjacobian)(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);
  void          *jacP;
  int           (*rhsbc)(TS,double,Vec,void*); 
  void          *bcP;

  /* ---------Inner nonlinear or linear solvers ---------------------------*/

  SNES          snes;
  SLES          sles;

  /* --- Routines and data that are unique to each particular solver --- */

  int           (*setup)(TS);            /* sets up the nonlinear solver */
  int           setupcalled;            /* true if setup has been called */
  int           (*step)(TS,int*,double*); /* stepping routine */      
  int           (*setfromoptions)(TS);    /* sets options from database */
  int           (*printhelp)(TS,char*);   /* prints help info */
  void          *data;                    /* implementationspecific data */

  void          *user;                    /* user context */

  /* ------------------  Parameters -------------------------------------- */

  int           max_steps;          /* max number of steps */
  double        max_time;
  double        time_step;
  double        initial_time_step;
  int           steps;              /* steps taken so far */
  double        ptime;              /* time taken so far */
  int           linear_its;         /* total number of linear solver iterations */
  int           nonlinear_its;      /* total number of nonlinear solver iterations */

  /* ------------------- Default work-area management ------------------ */

  int           nwork;              
  Vec           *work;
  int           (*destroy)(TS);
  int           (*view)(TS,Viewer);
};

extern int TSMonitor(TS,int,double,Vec);
extern int TSComputeRHSBoundaryConditions(TS,double,Vec); 

#endif
