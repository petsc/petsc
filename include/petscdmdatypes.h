#if !defined(_PETSCDMDATYPES_H)
#define _PETSCDMDATYPES_H

#include <petscdmtypes.h>

/*E
    DMDAStencilType - Determines if the stencil extends only along the coordinate directions, or also
      to the northeast, northwest etc

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate(), DMDASetStencilType()
E*/
typedef enum { DMDA_STENCIL_STAR,DMDA_STENCIL_BOX } DMDAStencilType;

/*E
    DMDABoundaryType - Describes the choice for fill of ghost cells on physical domain boundaries.

   Level: beginner

   A boundary may be of type DMDA_BOUNDARY_NONE (no ghost nodes), DMDA_BOUNDARY_GHOST (ghost nodes
   exist but aren't filled, you can put values into them and then apply a stencil that uses those ghost locations),
   DMDA_BOUNDARY_MIRROR (not yet implemented for 3d), or DMDA_BOUNDARY_PERIODIC
   (ghost nodes filled by the opposite edge of the domain).

   Note: This is information for the boundary of the __PHYSICAL__ domain. It has nothing to do with boundaries between
     processes, that width is always determined by the stencil width, see DMDASetStencilWidth().

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum { DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_MIRROR, DMDA_BOUNDARY_PERIODIC } DMDABoundaryType;

/*E
    DMDAInterpolationType - Defines the type of interpolation that will be returned by
       DMCreateInterpolation.

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(), DMDACreate()
E*/
typedef enum { DMDA_Q0, DMDA_Q1 } DMDAInterpolationType;

/*E
    DMDAElementType - Defines the type of elements that will be returned by
       DMDAGetElements()

   Level: beginner

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMCreateInterpolation(), DMDASetInterpolationType(),
          DMDASetElementType(), DMDAGetElements(), DMDARestoreElements(), DMDACreate()
E*/
typedef enum { DMDA_ELEMENT_P1, DMDA_ELEMENT_Q1 } DMDAElementType;

/*S
     DMDALocalInfo - C struct that contains information about a structured grid and a processors logical
              location in it.

   Level: beginner

  Concepts: distributed array

  Developer note: Then entries in this struct are int instead of PetscInt so that the elements may
                  be extracted in Fortran as if from an integer array

.seealso:  DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DM, DMDAGetLocalInfo(), DMDAGetInfo()
S*/
typedef struct {
  PetscInt         dim,dof,sw;
  PetscInt         mx,my,mz;    /* global number of grid points in each direction */
  PetscInt         xs,ys,zs;    /* starting point of this processor, excluding ghosts */
  PetscInt         xm,ym,zm;    /* number of grid points on this processor, excluding ghosts */
  PetscInt         gxs,gys,gzs;    /* starting point of this processor including ghosts */
  PetscInt         gxm,gym,gzm;    /* number of grid points on this processor including ghosts */
  DMDABoundaryType bx,by,bz; /* type of ghost nodes at boundary */
  DMDAStencilType  st;
  DM               da;
} DMDALocalInfo;

#endif
