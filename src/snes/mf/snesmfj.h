/* "$Id: snesmfj.h,v 1.15 2001/07/20 21:25:24 bsmith Exp bsmith $"; */
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

#include "include/petscsnes.h"         /*I  "petscsnes.h"   I*/

/*
    Table of functions that manage the computation and understanding
    of the parameter for finite difference based matrix-free computations
*/
struct _MFOps {
  int (*compute)(MatSNESMFCtx,Vec,Vec,PetscScalar *);
  int (*view)(MatSNESMFCtx,PetscViewer);
  int (*destroy)(MatSNESMFCtx);
  int (*setfromoptions)(MatSNESMFCtx);
};

struct _p_MatSNESMFCtx {    /* context for default matrix-free SNES */
  PETSCHEADER(struct _MFOps)
  SNES             snes;                   /* nonlinear solver */
  Vec              w;                      /* work vector */
  MatNullSpace     sp;                     /* null space context */
  double           error_rel;              /* square root of relative error in computing function */
  Scalar           currenth;               /* last differencing parameter h used */
  Scalar           *historyh;              /* history of differencing parameter h */
  int              ncurrenth,maxcurrenth; 
  void             *hctx;
  Mat              mat;                    /* back reference to shell matrix that contains this */
  int              recomputeperiod;        /* how often the h is recomputed; default to 1 */
  int              count;                  /* used by recomputeperiod */
  /*
        The next three are used only if user called MatSNESMFSetFunction()
  */
  int              (*func)(SNES,Vec,Vec,void*);  /* function used for matrix free */
  void             *funcctx;                     /* the context for the function */
  Vec              funcvec;                      /* location to store func(u) */
  Vec              current_f;                    /* location of F(u); used with F(u+h) */
  Vec              current_u;                    /* location of u; used with F(u+h) */

  PetscTruth       usesnes;                      /* if false indicates that one should (*func) 
                                                    instead of SNES even if snes is present */

  int              (*funci)(int,Vec,PetscScalar*,void*);  /* Evaluates func_[i]() */
  int              (*funcisetbase)(Vec,void*);       /* Sets base for future evaluations of func_[i]() */
};

EXTERN PetscFList      MatSNESMPetscFList;
EXTERN PetscTruth MatSNESMFRegisterAllCalled;

#endif
