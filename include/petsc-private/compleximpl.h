#if !defined(_COMPLEXIMPL_H)
#define _COMPLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmcomplex.h> /*I      "petscdmcomplex.h"    I*/
#include "petsc-private/dmimpl.h"

PETSC_EXTERN PetscLogEvent DMCOMPLEX_Distribute;

typedef struct _n_DMLabel *DMLabel;
struct _n_DMLabel {
  char     *name;           /* Label name */
  PetscInt  numStrata;      /* Number of integer values */
  PetscInt *stratumValues;  /* Value of each stratum */
  PetscInt *stratumOffsets; /* Offset of each stratum */
  PetscInt *stratumSizes;   /* Size of each stratum */
  PetscInt *points;         /* Points for each stratum, sorted after setup */
  DMLabel   next;           /* Linked list */
};

typedef struct {
  PetscInt             refct;
  PetscInt             dim;              /* Topological mesh dimension */

  /* Sieve */
  PetscSection         coneSection;      /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;      /* Cached for fast lookup */
  PetscInt            *cones;            /* Cone for each point */
  PetscInt            *coneOrientations; /* Orientation of each cone point, means cone traveral should start on point 'o', and if negative start on -(o+1) and go in reverse */
  PetscSection         supportSection;   /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize;   /* Cached for fast lookup */
  PetscInt            *supports;         /* Cone for each point */
  PetscReal            refinementLimit;  /* Maximum volume for refined cell */

  PetscInt            *facesTmp;            /* Work space for faces operation */

  /* Submesh */
  IS                   subpointMap;      /* map[submesh point] = original mesh point, original points are sorted so we can use PetscFindInt() */

  /* Labels and numbering */
  DMLabel              labels;           /* Linked list of labels */
  IS                   globalVertexNumbers;
  IS                   globalCellNumbers;

  /* Output */
  PetscInt             vtkCellMax, vtkVertexMax; /* Allow exclusion of some points in the VTK output */
  PetscInt             vtkCellHeight;            /* The height of cells for output, default is 0 */
  PetscReal            scale[NUM_PETSC_UNITS];   /* The scale for each SI unit */

  /* FEM (should go in another DM) */
  PetscErrorCode (*integrateResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                         const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]);
  PetscErrorCode (*integrateJacobianActionFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[], const PetscScalar[],
                                               const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                               void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                               void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                               void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                               void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]);
  PetscErrorCode (*integrateJacobianFEM)(PetscInt, PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                         const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]);

  /* Debugging */
  PetscBool            printSetValues;
  PetscInt             printFEM;
} DM_Complex;

#endif /* _COMPLEXIMPL_H */
