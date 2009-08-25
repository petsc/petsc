
/*
    Provides a PETSc interface to SUNDIALS. Alan Hindmarsh's parallel ODE
   solver developed at LLNL.
*/

#if !defined(__PETSCSUNDIALS_H)
#define __PETSCSUNDIALS_H

#include "private/tsimpl.h"       /*I   "petscts.h"   I*/
#include "private/pcimpl.h"               /*I   "petscpc.h"   I*/
#include "private/matimpl.h"

/*
   Include files specific for SUNDIALS
*/
#if defined(PETSC_HAVE_SUNDIALS)

EXTERN_C_BEGIN
#include "cvode/cvode.h"                  /* prototypes for CVODE fcts. */
#include "cvode/cvode_spgmr.h"            /* prototypes and constants for CVSPGMR solver */
#include "nvector/nvector_parallel.h"     /* definition N_Vector and macro NV_DATA_P  */
EXTERN_C_END

typedef struct {
  Vec        update;    /* work vector where new solution is formed */
  Vec        func;      /* work vector where F(t[i],u[i]) is stored */
  Vec        rhs;       /* work vector for RHS; vec_sol/dt */
  Vec        w1,w2;     /* work space vectors for function evaluation */
  PetscTruth exact_final_time; /* force Sundials to interpolate solution to exactly final time
                                   requested by user (default) */
  /* PETSc peconditioner objects used by SUNDIALS */
  Mat  pmat;                         /* preconditioner Jacobian */
  PC   pc;                           /* the PC context */
  int  cvode_type;                   /* the SUNDIALS method, BDF or ADAMS  */
  TSSundialsGramSchmidtType gtype; 
  int                       restart;
  double                    linear_tol;

  /* Variables used by Sundials */
  MPI_Comm    comm_sundials;
  double      reltol;
  double      abstol;        /* only for using SS flag in SUNDIALS */
  N_Vector    y;             /* current solution */
  void        *mem;
  PetscTruth  monitorstep;   /* flag for monitor internal steps; itask=V_ONE_STEP or itask=CV_NORMAL*/
} TS_Sundials;
#endif

#endif




