/* $Id: tsimpl.h,v 1.1 1996/01/01 19:52:21 bsmith Exp bsmith $ */

#ifndef __TSIMPL_H
#define __TSIMPL_H
#include "ts.h"

/*
    Time steping context. 
      
      General case: U_t = F(t,U) <-- the right hand side function.
      Linear  case: U_t = A(t) U. <-- the right hand side matrix.
*/

struct _TS {
  PETSCHEADER

  TSProblemType problem_type;

  Vec   vec_sol, vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  int   (*monitor)(TS,int,Scalar,Vec,void*); /* monitor routine */
  void  *monP;		                     /* monitor routine context */

  int   (*rhsfunction)(TS,Scalar,Vec,Vec,void*); 
  void  *funP;
  int   (*rhsmatrix)(TS,Scalar,Mat*,Mat*,MatStructure *,void*);       
  Mat   A,B;                        /* user provided matrix and preconditioner */
  int   (*rhsjacobian)(TS,Scalar,Vec,Mat*,Mat*,MatStructure *,void*);
  void  *jacP;

  /* ---------Inner nonlinear or linear solvers ---------------------------*/

  SNES     snes;
  SLES     sles;

  /* --- Routines and data that are unique to each particular solver --- */

  int   (*setup)(TS);               /* sets up the nonlinear solver */
  int   setup_called;               /* true if setup has been called */
  int   (*step)(TS,int*,Scalar*);      
  int   (*setfromoptions)(TS);      /* sets options from database */
  int   (*printhelp)(TS);           /* prints help info */
  void  *data;                      /* implementationspecific data */

  void  *user;                      /* user context */
  /* ------------------  Parameters -------------------------------------- */

  int      max_steps;          /* max number of steps */
  Scalar   max_time;
  Scalar   time_step;
  int      steps;              /* steps taken so far */
  Scalar   ptime;              /* time taken so far */

  /* ------------------- Default work-area management ------------------ */

  int      nwork;              
  Vec      *work;

};


#endif
