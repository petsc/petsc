/* "$Id: snesmfj.h,v 1.4 1999/03/24 04:29:52 curfman Exp bsmith $"; */
/*
    This file should be included in NEW routines that compute the
    differencing parameter for finite difference based matrix-free
    methods.  For example, such routines can compute h for use in
    Jacobian-vector products of the form

                       F(x+ha) - F(x)
          F'(u)a  ~=  ----------------
                            h
*/

#if !defined(__SNESMFJ_H__)
#define __SNESMFJ_H__

#include "include/snes.h"         /*I  "snes.h"   I*/

/*
    Table of functions that manage the computation and understanding
    of the parameter for finite difference based matrix-free computations
*/
typedef struct {
  int (*compute)(MatSNESMFCtx,Vec,Vec,Scalar *);
  int (*view)(MatSNESMFCtx,Viewer);
  int (*destroy)(MatSNESMFCtx);
  int (*printhelp)(MatSNESMFCtx);
  int (*setfromoptions)(MatSNESMFCtx);
} MFOps;

struct _p_MatSNESMFCtx {    /* context for default matrix-free SNES */
  MFOps            *ops;                   /* function table */
  MPI_Comm         comm;                   /* communicator */
  SNES             snes;                   /* nonlinear solver */
  Vec              w;                      /* work vector */
  PCNullSpace      sp;                     /* null space context; not currently used  */
  double           error_rel;              /* square root of relative error in computing function */
  Scalar           currenth;               /* last differencing parameter h used */
  Scalar           *historyh;              /* history of differencing parameter h */
  int              ncurrenth, maxcurrenth; 
  void             *hctx;
  char             type_name[256];
  Mat              mat;                          /* back reference to shell matrix that contains this */
  /*
        The next three are used only if user called MatSNESMFSetFunction()
  */
  int              (*func)(SNES,Vec,Vec,void*);  /* function used for matrix free */
  void             *funcctx;                     /* the context for the function */
  Vec              funcvec;                      /* location to store func(u) */
};

extern FList MatSNESMFList;
extern int   MatSNESMFRegisterAllCalled;

#endif
