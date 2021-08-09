#include <petscts.h>

/* Simple C struct that allows us to access the two velocity (x and y directions) values easily in the code */
typedef struct {
  PetscScalar u,v;
} Field;

/* Data structure to store the model parameters */
typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool aijpc;
  Vec       U;
  Mat       A;
  TS        ts;
} AppCtx;

/* User-supplied functions for TS */
PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
