#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct AppCtx AppCtx;
struct AppCtx {
  PetscReal param;          /* test problem parameter */
  PetscInt  m,n;            /* MMS3 parameters */
  PetscErrorCode (*mms_solution)(AppCtx*,const DMDACoor2d*,PetscScalar*);
  PetscErrorCode (*mms_forcing)(AppCtx*,const DMDACoor2d*,PetscScalar*);
};

PETSC_EXTERN PetscErrorCode FormFunctionLocalVec(DMDALocalInfo *info,Vec x,Vec f,AppCtx *user);
PETSC_EXTERN PetscErrorCode FormObjectiveLocalVec(DMDALocalInfo *info,Vec x,PetscReal *obj,AppCtx *user);
PETSC_EXTERN PetscErrorCode FormJacobianLocalVec(DMDALocalInfo *info,Vec x,Mat jac,Mat jacpre,AppCtx *user);
