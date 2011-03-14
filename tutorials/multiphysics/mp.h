
#include "petscdmmg.h"
#include <petscdmcomposite.h>

typedef struct {
  PetscScalar u,v,omega;
} Field1;

typedef struct {
  PetscScalar temp;
} Field2;

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */
} AppCtx;

extern PetscErrorCode FormInitialGuessLocal1(DMDALocalInfo*,Field1**);
extern PetscErrorCode FormFunctionLocal1(DMDALocalInfo*,Field1**,Field2**,Field1**,void*);

extern PetscErrorCode FormInitialGuessLocal2(DMDALocalInfo*,Field2**,AppCtx*);
extern PetscErrorCode FormFunctionLocal2(DMDALocalInfo*,Field1**,Field2**,Field2**,void*);

