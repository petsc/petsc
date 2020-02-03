
/*
    Provides a PETSc interface to SUNDIALS. Alan Hindmarsh's parallel ODE
   solver developed at LLNL.
*/

#if !defined(PETSC_SUNDIALS_H)
#define PETSC_SUNDIALS_H

#include <petsc/private/tsimpl.h>       /*I   "petscts.h"   I*/
#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>

/*
   Include files specific for SUNDIALS
*/
#if defined(PETSC_HAVE_SUNDIALS)

EXTERN_C_BEGIN
#include <cvode/cvode.h>                  /* prototypes for CVODE fcts. */
#include <cvode/cvode_spgmr.h>            /* prototypes and constants for CVSPGMR solver */
#include <nvector/nvector_parallel.h>     /* definition N_Vector and macro NV_DATA_P  */
EXTERN_C_END

typedef struct {
  Vec update;           /* work vector where new solution is formed */
  Vec ydot;             /* work vector the time derivative is stored */
  Vec w1,w2;            /* work space vectors for function evaluation */

  /* PETSc peconditioner objects used by SUNDIALS */
  PetscInt                  cvode_type;   /* the SUNDIALS method, BDF or ADAMS  */
  TSSundialsGramSchmidtType gtype;
  PetscReal                 linear_tol;
  PetscReal                 mindt,maxdt;

  /* Variables used by Sundials */
  MPI_Comm  comm_sundials;
  PetscReal reltol;
  PetscReal abstol;          /* only for using SS flag in SUNDIALS */
  N_Vector  y;               /* current solution */
  void      *mem;
  PetscBool monitorstep;     /* flag for monitor internal steps; itask=V_ONE_STEP or itask=CV_NORMAL*/
  PetscInt  maxl;            /* max dimension of the Krylov subspace to be used */
  PetscInt  maxord;          /* max order of BDF / Adams method */
} TS_Sundials;
#endif

#endif




