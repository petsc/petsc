#if !defined(_PLEXIMPL_H)
#define _PLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscbt.h>
#include "petsc-private/dmimpl.h"

PETSC_EXTERN PetscLogEvent DMPLEX_Distribute, DMPLEX_Stratify;

/* This is an integer map, in addition it is also a container class
   Design points:
     - Low storage is the most important design point
     - We want flexible insertion and deletion
     - We can live with O(log) query, but we need O(1) iteration over strata
*/
struct _n_DMLabel {
  PetscInt  refct;
  char     *name;           /* Label name */
  PetscInt  numStrata;      /* Number of integer values */
  PetscInt *stratumValues;  /* Value of each stratum */
  PetscInt *stratumOffsets; /* Offset of each stratum */
  PetscInt *stratumSizes;   /* Size of each stratum */
  PetscInt *points;         /* Points for each stratum, sorted after setup */
  DMLabel   next;           /* Linked list */
  PetscInt  pStart, pEnd;   /* Bounds for index lookup */
  PetscBT   bt;             /* A bit-wise index */
};

typedef struct {
  PetscInt             refct;
  PetscInt             dim;               /* Topological mesh dimension */

  /* Sieve */
  PetscSection         coneSection;       /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;       /* Cached for fast lookup */
  PetscInt            *cones;             /* Cone for each point */
  PetscInt            *coneOrientations;  /* Orientation of each cone point, means cone traveral should start on point 'o', and if negative start on -(o+1) and go in reverse */
  PetscSection         supportSection;    /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize;    /* Cached for fast lookup */
  PetscInt            *supports;          /* Cone for each point */
  PetscBool            refinementUniform; /* Flag for uniform cell refinement */
  PetscReal            refinementLimit;   /* Maximum volume for refined cell */
  PetscInt             hybridPointMax[8]; /* Allow segregation of some points, each dimension has a divider (used in VTK output and refinement) */

  PetscInt            *facesTmp;          /* Work space for faces operation */

  /* Submesh */
  DMLabel              subpointMap;       /* Label each original mesh point in the submesh with its depth, subpoint are the implicit numbering */

  /* Labels and numbering */
  DMLabel              labels;            /* Linked list of labels */
  IS                   globalVertexNumbers;
  IS                   globalCellNumbers;

  /* Preallocation */
  PetscInt             preallocCenterDim; /* Dimension of the points which connect adjacent points for preallocation */

  /* Output */
  PetscInt             vtkCellHeight;            /* The height of cells for output, default is 0 */
  PetscReal            scale[NUM_PETSC_UNITS];   /* The scale for each SI unit */

  /* FEM (should go in another DM) */
  PetscErrorCode (*integrateResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                         const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]), PetscScalar[]);
  PetscErrorCode (*integrateBdResidualFEM)(PetscInt, PetscInt, PetscInt, PetscQuadrature[], const PetscScalar[],
                                           const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                                           void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                           void (*)(const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]), PetscScalar[]);
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
} DM_Plex;

PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll_VTU(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexVTKGetCellType(DM,PetscInt,PetscInt,PetscInt*);
PETSC_EXTERN PetscErrorCode VecView_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex(Vec,PetscViewer);

#endif /* _PLEXIMPL_H */
