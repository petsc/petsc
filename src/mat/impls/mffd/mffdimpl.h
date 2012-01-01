/*
    This file should be included in NEW routines that compute the
    differencing parameter for finite difference based matrix-free
    methods.  For example, such routines can compute h for use in
    Jacobian-vector products of the form

                       F(x+ha) - F(x)
          F'(u)a  ~=  ----------------
                            h
*/

#if !defined(__MFFD_H__)
#define __MFFD_H__

#include <petscmat.h>         /*I  "petscmat.h"   I*/

/*
    Table of functions that manage the computation and understanding
    of the parameter for finite difference based matrix-free computations
*/
struct _MFOps {
  PetscErrorCode (*compute)(MatMFFD,Vec,Vec,PetscScalar *,PetscBool * zeroa);
  PetscErrorCode (*view)(MatMFFD,PetscViewer);
  PetscErrorCode (*destroy)(MatMFFD);
  PetscErrorCode (*setfromoptions)(MatMFFD);
};

struct _p_MatMFFD {    /* context for default matrix-free SNES */
  PETSCHEADER(struct _MFOps);
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
  PetscErrorCode   (*checkh)(void*,Vec,Vec,PetscScalar*);
  void             *checkhctx;             /* optional context used by MatMFFDSetCheckh() */

  PetscErrorCode   (*func)(void*,Vec,Vec);  /* function used for matrix free */
  void             *funcctx;                     /* the context for the function */
  Vec              current_f;                    /* location of F(u); used with F(u+h) */
  PetscBool        current_f_allocated;
  Vec              current_u;                    /* location of u; used with F(u+h) */

  PetscErrorCode   (*funci)(void*,PetscInt,Vec,PetscScalar*);  /* Evaluates func_[i]() */
  PetscErrorCode   (*funcisetbase)(void*,Vec);            /* Sets base for future evaluations of func_[i]() */

  PetscScalar      vscale,vshift;              /* diagonal scale and shift by scalars */
  Vec              dlscale,drscale,dshift;              /* diagonal scale and shift by vectors */
};

extern PetscFList MatMFFDList;
extern PetscBool  MatMFFDRegisterAllCalled;

#endif
