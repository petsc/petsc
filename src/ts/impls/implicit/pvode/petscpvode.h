#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: petscpvode.h,v 1.6 1998/01/17 17:38:20 bsmith Exp bsmith $"; 
#endif

/*
    Provides a PETSc interface to PVODE. Alan Hindmarsh's parallel ODE
   solver developed at LLNL.
*/

#if !defined(__PETSCPVODE_H)
#define __PETSCPVODE_H

#include <math.h>
#include "src/ts/tsimpl.h"              /*I   "ts.h"   I*/
#include "src/pc/pcimpl.h"              /*I   "pc.h"   I*/
#include "pinclude/pviewer.h"
#include "src/mat/matimpl.h"

/*
   Include files specific for PVODE
*/
#if defined(HAVE_PVODE) && !defined(__cplusplus) 
#include "llnltyps.h"
#include "cvode.h"
#include "vector.h"
#include "iterativ.h"
#include "cvspgmr.h"


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

  long int iopt[OPT_SIZE];
  double   ropt[OPT_SIZE];
  double   reltol;
  double   abstol;          /* only for using SS flag in PVODE */
  N_Vector y;               /* current solution */
  void     *mem;            /* time integrater context */
} TS_PVode;

#endif

#endif




