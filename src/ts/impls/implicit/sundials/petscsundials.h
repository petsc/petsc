
/*
    Provides a PETSc interface to SUNDIALS. Alan Hindmarsh's parallel ODE
   solver developed at LLNL.
*/

#if !defined(__PETSCSUNDIALS_H)
#define __PETSCSUNDIALS_H

#include "src/ts/tsimpl.h"              /*I   "petscts.h"   I*/
#include "src/ksp/pc/pcimpl.h"         /*I   "petscpc.h"   I*/
#include "src/mat/matimpl.h"

/*
   Include files specific for SUNDIALS
*/
#if defined(PETSC_HAVE_SUNDIALS)

EXTERN_C_BEGIN
#include "sundialstypes.h"
#include "cvode.h"
#include "nvector.h"
#include "nvector_serial.h"
#include "nvector_parallel.h"
#include "iterative.h" 
#include "cvspgmr.h"
EXTERN_C_END

typedef struct {
  Vec  update;    /* work vector where new solution is formed */
  Vec  func;      /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;       /* work vector for RHS; vec_sol/dt */
  Vec  w1,w2;     /* work space vectors for function evaluation */
  PetscTruth  exact_final_time; /* force Sundials to interpolate solution to exactly final time
                                   requested by user (default) */
  /* PETSc peconditioner objects used by SUNDIALS */
  Mat  pmat;                         /* preconditioner Jacobian */
  PC   pc;                           /* the PC context */
  int  cvode_type;                   /* the SUNDIALS method, BDF or ADAMS   */
  TSSundialsGramSchmidtType gtype; 
  int                    restart;
  double                 linear_tol;

  /* Variables used by Sundials */
  MPI_Comm comm_sundials;
  double   reltol;
  double   abstol;          /* only for using SS flag in SUNDIALS */
  N_Vector y;               /* current solution */
  void     *mem;            /* time integrater context */
  int      nonlinear_solves,linear_solves; /* since creation of object */
} TS_Sundials;
#endif

#endif




