/* $Id: tsimpl.h,v 1.5 1996/03/26 16:21:14 balay Exp curfman $ */

#ifndef __TSIMPL_H
#define __TSIMPL_H
#include "ts.h"

/*
    Time steping context. 
      
      General case: U_t = F(t,U) <-- the right hand side function.
      Linear  case: U_t = A(t) U. <-- the right hand side matrix.
      Linear (no time) case: U_t = A U. <-- the right hand side matrix
*/

struct _TS {
  PETSCHEADER

  TSProblemType problem_type;

  Vec           vec_sol, vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  int           (*monitor)(TS,int,double,Vec,void*); /* monitor routine */
  void          *monP;		            /* monitor routine context */

  int           (*rhsmatrix)(TS,double,Mat*,Mat*,MatStructure *,void*);
  Mat           A,B;        /* user provided matrix and preconditioner */
  Mat           Ashell;     /* if user provided a Shell matrix */

  int           (*rhsfunction)(TS,double,Vec,Vec,void*); 
  void          *funP;
  int           (*rhsjacobian)(TS,double,Vec,Mat*,Mat*,MatStructure *,void*);
  void          *jacP;

  /* ---------Inner nonlinear or linear solvers ---------------------------*/

  SNES          snes;
  SLES          sles;

  /* --- Routines and data that are unique to each particular solver --- */

  int           (*setup)(TS);            /* sets up the nonlinear solver */
  int           setup_called;            /* true if setup has been called */
  int           (*step)(TS,int*,double*); /* stepping routine */      
  int           (*setfromoptions)(TS);    /* sets options from database */
  int           (*printhelp)(TS);         /* prints help info */
  void          *data;                    /* implementationspecific data */

  void          *user;                    /* user context */
  /* ------------------  Parameters -------------------------------------- */

  int           max_steps;          /* max number of steps */
  double        max_time;
  double        time_step;
  int           steps;              /* steps taken so far */
  double        ptime;              /* time taken so far */

  /* ------------------- Default work-area management ------------------ */

  int           nwork;              
  Vec           *work;
};


extern int TSMonitor(TS,int,Scalar,Vec);
#endif
