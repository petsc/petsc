/*
  Code for solving DAEs in semi-explicit form
*/
#include <petsc-private/tsimpl.h>

typedef struct {
  Vec            U,V;
  PetscErrorCode (*setfromoptions)(TS);
  PetscErrorCode (*solve)(TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS,PetscViewer);
  PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec,void*);
  void           *fctx,*Fctx;
  void           *data;
}TS_DAESimple;

extern PetscErrorCode TSDestroy_DAESimple(TS);
extern PetscErrorCode TSReset_DAESimple(TS);
extern PetscErrorCode TSSetFromOptions_DAESimple(TS);
extern PetscErrorCode TSSetUp_DAESimple(TS);
extern PetscErrorCode TSSolve_DAESimple(TS);
