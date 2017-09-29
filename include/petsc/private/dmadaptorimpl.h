#if !defined(_DMADAPTORIMPL_H)
#define _DMADAPTORIMPL_H

#include <petscdmadaptor.h>
#include <petsc/private/petscimpl.h>

typedef struct _DMAdaptorOps *DMAdaptorOps;
struct _DMAdaptorOps {
  PetscErrorCode (*setfromoptions)(DMAdaptor);
  PetscErrorCode (*setup)(DMAdaptor);
  PetscErrorCode (*view)(DMAdaptor,PetscViewer);
  PetscErrorCode (*destroy)(DMAdaptor);
  PetscErrorCode (*computesolution)(DM,Vec,void*);
  PetscErrorCode (*computeerrorindicator)(DMAdaptor,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscFVCellGeom*,PetscReal*,void*);
};

struct _p_DMAdaptor
{
  PETSCHEADER(struct _DMAdaptorOps);
  /* Inputs */
  DM                 idm;  /* Initial grid */
  SNES               snes; /* Solver */
  VecTagger          refineTag, coarsenTag; /* Criteria for adaptivity */
  /*   FVM support */
  PetscBool          computeGradient;
  DM                 cellDM, gradDM;
  Vec                cellGeom, faceGeom, cellGrad; /* Local vectors */
  const PetscScalar *cellGeomArray, *cellGradArray;
  /* Outputs */
  DM                 odm;  /* Output grid */
  PetscBool          monitor;
  /* Auxiliary objects */
  PetscLimiter       limiter;
};

#endif
