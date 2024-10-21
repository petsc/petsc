#pragma once

#include <petscdmadaptor.h>
#include <petsc/private/petscimpl.h>

#define MAXDMADAPTORMONITORS 16

typedef struct _DMAdaptorOps *DMAdaptorOps;
struct _DMAdaptorOps {
  PetscErrorCode (*setfromoptions)(DMAdaptor);
  PetscErrorCode (*setup)(DMAdaptor);
  PetscErrorCode (*view)(DMAdaptor, PetscViewer);
  PetscErrorCode (*destroy)(DMAdaptor);
  PetscErrorCode (*transfersolution)(DMAdaptor, DM, Vec, DM, Vec, void *);
  PetscErrorCode (*mixedsetup)(DMAdaptor, DM);
  PetscErrorCode (*computeerrorindicator)(DMAdaptor, Vec, Vec);
  PetscErrorCode (*computecellerrorindicator)(DMAdaptor, PetscInt, PetscInt, const PetscScalar *, const PetscScalar *, const PetscFVCellGeom *, PetscReal *, void *);
};

struct _p_DMAdaptor {
  PETSCHEADER(struct _DMAdaptorOps);
  void *data;

  /* Inputs */
  DM        idm;                   /* Initial grid */
  SNES      snes;                  /* Solver */
  VecTagger refineTag, coarsenTag; /* Criteria for adaptivity */
  /*   control */
  DMAdaptationCriterion adaptCriterion;
  PetscBool             femType;
  PetscInt              numSeq;           /* Number of sequential adaptations */
  PetscInt              Nadapt;           /* Target number of vertices */
  PetscReal             refinementFactor; /* N_adapt = r^dim N_orig */
  /*   FVM support */
  PetscBool          computeGradient;
  DM                 cellDM, gradDM;
  Vec                cellGeom, faceGeom, cellGrad; /* Local vectors */
  const PetscScalar *cellGeomArray, *cellGradArray;
  // Monitors
  PetscErrorCode (*monitor[MAXDMADAPTORMONITORS])(DMAdaptor, PetscInt, DM, DM, PetscInt, PetscReal[], Vec, void *);
  PetscCtxDestroyFn *monitordestroy[MAXDMADAPTORMONITORS];
  void              *monitorcontext[MAXDMADAPTORMONITORS];
  PetscInt           numbermonitors;
  /* Auxiliary objects */
  PetscLimiter limiter;
  PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  void **exactCtx;
};
