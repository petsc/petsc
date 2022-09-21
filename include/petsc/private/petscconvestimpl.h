#ifndef PETSCCEIMPL_H
#define PETSCCEIMPL_H

#include <petscconvest.h>
#include <petsc/private/petscimpl.h>

typedef struct _PetscConvEstOps *PetscConvEstOps;
struct _PetscConvEstOps {
  PetscErrorCode (*setfromoptions)(PetscConvEst);
  PetscErrorCode (*setup)(PetscConvEst);
  PetscErrorCode (*view)(PetscConvEst, PetscViewer);
  PetscErrorCode (*destroy)(PetscConvEst);
  PetscErrorCode (*setsolver)(PetscConvEst, PetscObject);
  PetscErrorCode (*initguess)(PetscConvEst, PetscInt, DM, Vec);
  PetscErrorCode (*computeerror)(PetscConvEst, PetscInt, DM, Vec, PetscReal[]);
  PetscErrorCode (*getconvrate)(PetscConvEst, PetscReal[]);
};

struct _p_PetscConvEst {
  PETSCHEADER(struct _PetscConvEstOps);
  /* Inputs */
  DM          idm;      /* Initial grid */
  PetscObject solver;   /* Solver */
  PetscReal   r;        /* The refinement factor (spatial check requires r = 2) */
  PetscInt    Nr;       /* The number of refinements */
  PetscInt    Nf;       /* The number of fields in the DM */
  PetscBool   noRefine; /* Debugging flag to disable refinement */
  PetscErrorCode (**initGuess)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  void **ctxs;
  /* Outputs */
  PetscLogEvent event;
  PetscBool     monitor;
  PetscInt     *dofs;
  PetscReal    *errors;
};

#endif
