
#include "petscdmmg.h"

typedef struct {
  PetscScalar u,v,omega;
} Field1;

typedef struct {
  PetscScalar temp;
} Field2;

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */

  PetscInt     nsolve;
  Field1       **x1;  /* passing local ghosted vector array of Physics 1 */
  Field2       **x2;  /* passing local ghosted vector array of Physics 2 */
} AppCtx;

extern PetscErrorCode FormInitialGuessLocal1(DALocalInfo*,Field1**);
extern PetscErrorCode FormFunctionLocal1(DALocalInfo*,Field1**,Field2**,Field1**,void*);

extern PetscErrorCode FormInitialGuessLocal2(DALocalInfo*,Field2**,AppCtx*);
extern PetscErrorCode FormFunctionLocal2(DALocalInfo*,Field1**,Field2**,Field2**,void*);

