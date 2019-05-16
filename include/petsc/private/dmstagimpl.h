#if !defined(DMSTAGIMPL_H_)
#define DMSTAGIMPL_H_

#include <petscdmstag.h> /*I "petscdmstag.h" I*/
#include <petsc/private/dmimpl.h>

#define DMSTAG_MAX_DIM 3
#define DMSTAG_MAX_STRATA DMSTAG_MAX_DIM + 1

/* This value is 1 + 3^DMSTAG_MAX DIM */
#define DMSTAG_NUMBER_LOCATIONS 28

typedef struct {
  /* Fields which may require being set before DMSetUp() is called, set by DMStagInitialize().
     Some may be adjusted by DMSetUp() */
  PetscInt          N[DMSTAG_MAX_DIM];            /* Global dimensions (elements)      */
  PetscInt          n[DMSTAG_MAX_DIM];            /* Local dimensions (elements)       */
  PetscInt          dof[DMSTAG_MAX_STRATA];       /* Dof per point for each stratum    */
  DMStagStencilType stencilType;                  /* Elementwise stencil type          */
  PetscInt          stencilWidth;                 /* Elementwise ghost width           */
  DMBoundaryType    boundaryType[DMSTAG_MAX_DIM]; /* Physical domain ghosting type     */
  PetscInt          nRanks[DMSTAG_MAX_DIM];       /* Ranks in each direction           */

  /* Additional fields populated by DMSetUp() */
  PetscInt          nGhost[DMSTAG_MAX_DIM];       /* Local dimensions (w/ ghosts)      */
  PetscInt          start[DMSTAG_MAX_DIM];        /* First element number              */
  PetscInt          startGhost[DMSTAG_MAX_DIM];   /* First element number (w/ ghosts)  */
  PetscMPIInt       rank[DMSTAG_MAX_DIM];         /* Location in grid of ranks         */
  PetscMPIInt       *neighbors;                   /* dim^3 local ranks                 */
  PetscInt          *l[DMSTAG_MAX_DIM];           /* Elements/rank in each direction   */
  VecScatter        gtol;                         /* Global --> Local                  */
  VecScatter        ltog_injective;               /* Local  --> Global, injective      */
  PetscInt          *locationOffsets;             /* Offsets for points in loc. rep.   */

  /* Coordinates */
  DMType            coordinateDMType;             /* DM type to create for coordinates */

  /* Convenience (easily computed from the above) */
  PetscInt          entriesPerElement;            /* Entries stored with each element   */
  PetscInt          entries;                      /* Local number of entries            */
  PetscInt          entriesGhost;                 /* Local numbers of entries w/ ghosts */
  PetscBool         firstRank[DMSTAG_MAX_DIM];    /* First rank in this dim?            */
  PetscBool         lastRank[DMSTAG_MAX_DIM];     /* Last rank in this dim?             */
} DM_Stag;

PETSC_INTERN PetscErrorCode DMStagDuplicateWithoutSetup(DM,MPI_Comm,DM*);
PETSC_INTERN PetscErrorCode DMStagInitialize(DMBoundaryType,DMBoundaryType,DMBoundaryType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencilType,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM);
PETSC_INTERN PetscErrorCode DMSetUp_Stag_1d(DM);
PETSC_INTERN PetscErrorCode DMSetUp_Stag_2d(DM);
PETSC_INTERN PetscErrorCode DMSetUp_Stag_3d(DM);
PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_1d(DM);
PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_2d(DM);
PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_3d(DM);
PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_1d(DM,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_2d(DM,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_3d(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);

#endif
