#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: petsccvode.h,v 1.2 1997/07/10 18:24:54 lixu Exp lixu $"; 
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
#include "src/snes/snesimpl.h" 
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

  /*
     PETSc peconditioner objects used by PVODE
  */
 
  Mat  pmat;          /* actually, it's not the preconditioner, but the Jacobian */
  PC   pc;            /* the PC context */
  int  cvode_method;  /* the PVODE method, BDF  or ADAMS   */
 

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

#define Ith(v,i)    N_VIth(v,i-1)         /* the ith local component of v */

#endif

#endif




