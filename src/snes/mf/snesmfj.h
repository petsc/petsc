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
  PetscErrorCode (*compute)(MatSNESMFCtx,Vec,Vec,PetscScalar *);
  PetscErrorCode (*view)(MatSNESMFCtx,PetscViewer);
  PetscErrorCode (*destroy)(MatSNESMFCtx);
  PetscErrorCode (*setfromoptions)(MatSNESMFCtx);
};

struct _p_MatSNESMFCtx {    /* context for default matrix-free SNES */
  PETSCHEADER(struct _MFOps);
  SNES             snes;                   /* nonlinear solver */
  Vec              w;                      /* work vector */
  MatNullSpace     sp;                     /* null space context */
  PetscReal        error_rel;              /* square root of relative error in computing function */
  PetscScalar      currenth;               /* last differencing parameter h used */
  PetscScalar      *historyh;              /* history of differencing parameter h */
  PetscInt         ncurrenth,maxcurrenth; 
  void             *hctx;
  Mat              mat;                    /* back reference to shell matrix that contains this */
  PetscInt         recomputeperiod;        /* how often the h is recomputed; default to 1 */
  PetscInt         count;                  /* used by recomputeperiod */
  void             *checkhctx;             /* optional context used by MatSNESMFSetCheckh() */
  PetscErrorCode (*checkh)(Vec,Vec,PetscScalar*,void*);
  /*
        The next three are used only if user called MatSNESMFSetFunction()
  */
  PetscErrorCode (*func)(SNES,Vec,Vec,void*);  /* function used for matrix free */
  void             *funcctx;                     /* the context for the function */
  Vec              funcvec;                      /* location to store func(u) */
  Vec              current_f;                    /* location of F(u); used with F(u+h) */
  Vec              current_u;                    /* location of u; used with F(u+h) */

  PetscTruth       usesnes;                      /* if false indicates that one should (*func) 
                                                    instead of SNES even if snes is present */

  PetscErrorCode (*funci)(PetscInt,Vec,PetscScalar*,void*);  /* Evaluates func_[i]() */
  PetscErrorCode (*funcisetbase)(Vec,void*);            /* Sets base for future evaluations of func_[i]() */

  PetscScalar      vscale,vshift;
};

EXTERN PetscFList MatSNESMPetscFList;
EXTERN PetscTruth MatSNESMFRegisterAllCalled;

#endif
