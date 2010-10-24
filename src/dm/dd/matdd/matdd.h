#if !defined(__MATDD_H)
#define __MATDD_H

#include "private/matimpl.h"
#include "../src/vec/vec/impls/dvecimpl.h"
#include "../src/vec/vec/impls/dd/vecdd.h"


struct _MatDDOps {
  PetscErrorCode (*setvalues            )(Mat M, PetscInt r, PetscInt c, PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv);
  PetscErrorCode (*setvalueslocal       )(Mat M, PetscInt r, PetscInt c, PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv);
  PetscErrorCode (*multrow              )(Mat M, PetscInt i, Vec x, Vec y);
  PetscErrorCode (*multaddcol           )(Mat M, PetscInt j, Vec x, Vec u, Vec y);
  PetscErrorCode (*multtransposeaddrow  )(Mat M, PetscInt i, Vec x, Vec u, Vec y);
  PetscErrorCode (*multtransposecol     )(Mat M, PetscInt j, Vec x, Vec y);
  PetscErrorCode (*multaddblock         )(Mat M, PetscInt i, PetscInt j, Vec x, Vec u, Vec y);
  PetscErrorCode (*multtransposeaddblock)(Mat M, PetscInt i, Vec x, Vec u, Vec y);

};

typedef struct {
  struct _MatDDOps *ops;
  DDLayout rmapdd, cmapdd;
  Vec outvec, invec;
  PetscTruth setup;
  Mat scatter, gather;
} Mat_DD; 


#endif
