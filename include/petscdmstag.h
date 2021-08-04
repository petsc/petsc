#if !defined(DMSTAG_H_)
#define DMSTAG_H_

#include <petscdm.h>
#include <petscdmproduct.h>

/*E
  DMStagStencilLocation - enumerated type denoted a location relative to an element in a DMStag grid

  The interpretation of these values is dimension-dependent.

  Level: beginner

.seealso: DMSTAG, DMStagStencil, DMStagGetLocationSlot()
E*/
typedef enum {
DMSTAG_NULL_LOCATION=0,
DMSTAG_BACK_DOWN_LEFT,
DMSTAG_BACK_DOWN,
DMSTAG_BACK_DOWN_RIGHT,
DMSTAG_BACK_LEFT,
DMSTAG_BACK,
DMSTAG_BACK_RIGHT,
DMSTAG_BACK_UP_LEFT,
DMSTAG_BACK_UP,
DMSTAG_BACK_UP_RIGHT,
DMSTAG_DOWN_LEFT,
DMSTAG_DOWN,
DMSTAG_DOWN_RIGHT,
DMSTAG_LEFT,
DMSTAG_ELEMENT,
DMSTAG_RIGHT,
DMSTAG_UP_LEFT,
DMSTAG_UP,
DMSTAG_UP_RIGHT,
DMSTAG_FRONT_DOWN_LEFT,
DMSTAG_FRONT_DOWN,
DMSTAG_FRONT_DOWN_RIGHT,
DMSTAG_FRONT_LEFT,
DMSTAG_FRONT,
DMSTAG_FRONT_RIGHT,
DMSTAG_FRONT_UP_LEFT,
DMSTAG_FRONT_UP,
DMSTAG_FRONT_UP_RIGHT
} DMStagStencilLocation;
PETSC_EXTERN const char *const DMStagStencilLocations[]; /* Corresponding strings (see stagstencil.c) */

/*S
  DMStagStencil - data structure representing a degree of freedom on a DMStag grid

  Data structure (C struct), analogous to describing a degree of freedom associated with a DMStag object,
  in terms of a global element index in each of up to three directions, a "location" as defined by DMStagStencilLocation,
  and a component number. Primarily for use with DMStagMatSetValuesStencil() (compare with use of MatStencil with MatSetValuesStencil()).

  Note:
  The component (c) field must always be set, even if there is a single component at a given location (in which case c should be set to 0).

Level: beginner

.seealso: DMSTAG, DMStagMatSetValuesStencil(), DMStagVecSetValuesStencil(), DMStagStencilLocation, DMStagSetStencilWidth(), DMStagSetStencilType(), DMStagVecGetValuesStencil()
S*/
typedef struct {
  DMStagStencilLocation loc;
  PetscInt              i,j,k,c;
} DMStagStencil;

/*E
  DMStagStencilType - Elementwise stencil type, determining which neighbors participate in communication

  Level: beginner

.seealso: DMSTAG, DMStagCreate1d(), DMStagCreate2d(), DMStagCreate3d(), DMStagStencil
E*/

typedef enum{DMSTAG_STENCIL_NONE=0,DMSTAG_STENCIL_STAR,DMSTAG_STENCIL_BOX} DMStagStencilType;
PETSC_EXTERN const char *const DMStagStencilTypes[]; /* Corresponding strings (see stagstencil.c) */

PETSC_EXTERN PetscErrorCode DMCreate_Stag(DM);
PETSC_EXTERN PetscErrorCode DMStagCreate1d(MPI_Comm,DMBoundaryType,PetscInt,PetscInt,PetscInt,DMStagStencilType,PetscInt,const PetscInt[],DM*);
PETSC_EXTERN PetscErrorCode DMStagCreate2d(MPI_Comm,DMBoundaryType,DMBoundaryType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencilType,PetscInt,const PetscInt[],const PetscInt[],DM*);
PETSC_EXTERN PetscErrorCode DMStagCreate3d(MPI_Comm,DMBoundaryType,DMBoundaryType,DMBoundaryType,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencilType,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],DM*);
PETSC_EXTERN PetscErrorCode DMStagCreateCompatibleDMStag(DM,PetscInt,PetscInt,PetscInt,PetscInt,DM*);
PETSC_EXTERN PetscErrorCode DMStagGetBoundaryTypes(DM,DMBoundaryType*,DMBoundaryType*,DMBoundaryType*);
PETSC_EXTERN PetscErrorCode DMStagGetCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetDOF(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetEntries(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetEntriesPerElement(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetGhostCorners(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetGlobalSizes(DM,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetIsFirstRank(DM,PetscBool*,PetscBool*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMStagGetIsLastRank(DM,PetscBool*,PetscBool*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMStagGetLocalSizes(DM,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetLocationDOF(DM,DMStagStencilLocation,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetLocationSlot(DM,DMStagStencilLocation,PetscInt,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetNumRanks(DM,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetOwnershipRanges(DM,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_EXTERN PetscErrorCode DMStagGetProductCoordinateArrays(DM,void*,void*,void*);
PETSC_EXTERN PetscErrorCode DMStagGetProductCoordinateArraysRead(DM,void*,void*,void*);
PETSC_EXTERN PetscErrorCode DMStagGetProductCoordinateLocationSlot(DM,DMStagStencilLocation,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagGetStencilType(DM,DMStagStencilType*);
PETSC_EXTERN PetscErrorCode DMStagGetStencilWidth(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagMatGetValuesStencil(DM,Mat,PetscInt,const DMStagStencil*,PetscInt,const DMStagStencil*,PetscScalar*);
PETSC_EXTERN PetscErrorCode DMStagMatSetValuesStencil(DM,Mat,PetscInt,const DMStagStencil*,PetscInt,const DMStagStencil*,const PetscScalar*,InsertMode);
PETSC_EXTERN PetscErrorCode DMStagMigrateVec(DM,Vec,DM,Vec);
PETSC_EXTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective(DM);
PETSC_EXTERN PetscErrorCode DMStagRestoreProductCoordinateArrays(DM,void*,void*,void*);
PETSC_EXTERN PetscErrorCode DMStagRestoreProductCoordinateArraysRead(DM,void*,void*,void*);
PETSC_EXTERN PetscErrorCode DMStagSetBoundaryTypes(DM,DMBoundaryType,DMBoundaryType,DMBoundaryType);
PETSC_EXTERN PetscErrorCode DMStagSetCoordinateDMType(DM,DMType);
PETSC_EXTERN PetscErrorCode DMStagSetDOF(DM,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMStagSetGlobalSizes(DM,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMStagSetNumRanks(DM,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMStagSetOwnershipRanges(DM,PetscInt const *,PetscInt const *,PetscInt const *);
PETSC_EXTERN PetscErrorCode DMStagSetStencilType(DM,DMStagStencilType);
PETSC_EXTERN PetscErrorCode DMStagSetStencilWidth(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMStagSetUniformCoordinates(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode DMStagSetUniformCoordinatesProduct(DM,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode DMStagStencilToIndexLocal(DM,PetscInt,PetscInt,const DMStagStencil*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMStagVecGetArray(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMStagVecGetArrayRead(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMStagVecGetValuesStencil(DM,Vec,PetscInt,const DMStagStencil*,PetscScalar*);
PETSC_EXTERN PetscErrorCode DMStagVecRestoreArray(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMStagVecRestoreArrayRead(DM,Vec,void*);
PETSC_EXTERN PetscErrorCode DMStagVecSetValuesStencil(DM,Vec,PetscInt,const DMStagStencil*,const PetscScalar*,InsertMode);
PETSC_EXTERN PetscErrorCode DMStagVecSplitToDMDA(DM,Vec,DMStagStencilLocation,PetscInt,DM*,Vec*);

PETSC_DEPRECATED_FUNCTION("Use DMStagGetProductCoordinateArraysRead() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagGet1dCoordinateArraysDOFRead(DM dm,void *ax,void *ay,void *az) {return DMStagGetProductCoordinateArraysRead(dm,ax,ay,az);}
PETSC_DEPRECATED_FUNCTION("Use DMStagGetProductCoordinateLocationSlot() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagGet1dCoordinateLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt *s) {return DMStagGetProductCoordinateLocationSlot(dm,loc,s);}
PETSC_DEPRECATED_FUNCTION("Use DMStagGetStencilType() (since version 3.11)") PETSC_STATIC_INLINE PetscErrorCode DMStagGetGhostType(DM dm,DMStagStencilType *s) {return DMStagGetStencilType(dm,s);}
PETSC_DEPRECATED_FUNCTION("Use DMStagRestoreProductCoordinateArraysRead() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagRestore1dCoordinateArraysDOFRead(DM dm,void *ax,void *ay,void *az) {return DMStagRestoreProductCoordinateArraysRead(dm,ax,ay,az);}
PETSC_DEPRECATED_FUNCTION("Use DMStagSetStencilType() (since version 3.11)") PETSC_STATIC_INLINE PetscErrorCode DMStagSetGhostType(DM dm,DMStagStencilType *s) {return DMStagGetStencilType(dm,s);}
PETSC_DEPRECATED_FUNCTION("Use DMStagVecGetArray() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagVecGetArrayDOF(DM dm,Vec v,void *a) {return DMStagVecGetArray(dm,v,a);}
PETSC_DEPRECATED_FUNCTION("Use DMStagVecGetArrayRead() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagVecGetArrayDOFRead(DM dm,Vec v,void *a) {return DMStagVecGetArrayRead(dm,v,a);}
PETSC_DEPRECATED_FUNCTION("Use DMStagVecRestoreArray() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagVecRestoreArrayDOF(DM dm,Vec v,void *a) {return DMStagVecRestoreArray(dm,v,a);}
PETSC_DEPRECATED_FUNCTION("Use DMStagVecRestoreArrayRead() (since version 3.13") PETSC_STATIC_INLINE PetscErrorCode DMStagVecRestoreArrayDOFRead(DM dm,Vec v,void *a) {return DMStagVecRestoreArrayRead(dm,v,a);}

#endif
