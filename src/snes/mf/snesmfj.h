/* "$Id: snesmfj.h,v 1.2 1998/11/20 15:30:39 bsmith Exp bsmith $"; */
/*
       This file should be included for those programming NEW routines
    to compute the h factor for finite difference based matrix free methods
*/

#if !defined(__SNESMFJ_H__)
#define __SNESMFJ_H__

#include "include/snes.h"         /*I  "snes.h"   I*/

/*
        Table of functions that manage the computation and understanding
    of the h parameter for finite difference based matrix-free computations
*/
typedef struct {
  int (*compute)(MatSNESMFCtx,Vec,Vec,Scalar *);
  int (*view)(MatSNESMFCtx,Viewer);
  int (*destroy)(MatSNESMFCtx);
  int (*printhelp)(MatSNESMFCtx);
  int (*setfromoptions)(MatSNESMFCtx);
} MFOps;

struct _p_MatSNESMFCtx {    /* context for default matrix-free SNES */
  MFOps            *ops;
  MPI_Comm         comm;
  SNES             snes;                   
  Vec              w;                      /* work vector */
  PCNullSpace      sp;                     /* null space context; not currently used  */
  double           error_rel;              /* square root of relative error in computing function */
  Scalar           currenth;               /* last differencing parameter h used */
  Scalar           *historyh;              /* history of differencing parameter h */
  int              ncurrenth,maxcurrenth;
  void             *hctx;
  char             type_name[256];
  Mat              mat;                    /* back reference to shell matrix that contains this */
};

extern FList MatSNESMFList;
extern int   MatSNESMFRegisterAllCalled;

#endif
