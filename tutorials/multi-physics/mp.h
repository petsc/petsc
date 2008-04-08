
#include "petscdmmg.h"

typedef struct {
  PetscScalar u,v,omega;
} Field1;

typedef struct {
  PetscScalar temp;
} Field2;

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */
  DMMG         *dmmg1,*dmmg2,*dmmg_comp;             /* used by MySolutionView() */
  DMComposite  pack;
} AppCtx;

extern PetscErrorCode FormInitialLocalGuess1(DALocalInfo*,Field1**,Field1**,void*);
extern PetscErrorCode FormFunctionLocal1(DALocalInfo*,Field1**,Field2**,Field1**,void*);

extern PetscErrorCode FormInitialLocalGuess2(DALocalInfo*,Field2**,Field2**,void*);
extern PetscErrorCode FormFunctionLocal2(DALocalInfo*,Field1**,Field2**,Field2**,void*);

