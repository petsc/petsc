/*$Id: petscpvode.h,v 1.13 2000/05/08 15:09:13 balay Exp $*/

/*
    Provides a PETSc interface to PVODE. Alan Hindmarsh's parallel ODE
   solver developed at LLNL.
*/

#if !defined(__PETSCPVODE_H)
#define __PETSCPVODE_H

#include "src/ts/tsimpl.h"              /*I   "petscts.h"   I*/
#include "src/sles/pc/pcimpl.h"         /*I   "petscpc.h"   I*/
#include "src/mat/matimpl.h"

/*
   Include files specific for PVODE
*/
#if defined(PETSC_HAVE_PVODE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)

EXTERN_C_BEGIN
#include "sundialstypes.h"
#include "cvode.h"
#include "nvector.h"
#include "nvector_parallel.h"
#include "iterativ.h"
#include "cvspgmr.h"
EXTERN_C_END

typedef struct {
  Vec  update;    /* work vector where new solution is formed */
  Vec  func;      /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;       /* work vector for RHS; vec_sol/dt */

  Vec  w1,w2;     /* work space vectors for function evaluation */


  PetscTruth  exact_final_time; /* force PVode to interpolate solution to exactly final time
                                   requested by user (default) */

  /*
     PETSc peconditioner objects used by PVODE
  */
 
  Mat  pmat;                         /* preconditioner Jacobian */
  PC   pc;                           /* the PC context */
  int  cvode_type;                   /* the PVODE method, BDF  or ADAMS   */
  TSPVodeGramSchmidtType gtype; 
  int                    restart;
  double                 linear_tol;

  /*
     Variables used by PVode 
  */

  MPI_Comm comm_pvode;
  long int iopt[OPT_SIZE];
  double   ropt[OPT_SIZE];
  double   reltol;
  double   abstol;          /* only for using SS flag in PVODE */
  N_Vector y;               /* current solution */
  void     *mem;            /* time integrater context */

  int      nonlinear_solves,linear_solves; /* since creation of object */
} TS_PVode;

#endif

#endif




