#if !defined(_PETSCDMTYPES_H)
#define _PETSCDMTYPES_H

/*S
     DM - Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers

   Level: intermediate

  Concepts: grids, grid refinement

   Notes: The DMDACreate() based object and the DMCompositeCreate() based object are examples of DMs

.seealso:  DMCompositeCreate(), DMDACreate(), DMSetType(), DMType
S*/
typedef struct _p_DM* DM;

/*E
  DMBoundaryType - Describes the choice for fill of ghost cells on physical domain boundaries.

  Level: beginner

  A boundary may be of type DM_BOUNDARY_NONE (no ghost nodes), DM_BOUNDARY_GHOST (ghost vertices/cells
  exist but aren't filled, you can put values into them and then apply a stencil that uses those ghost locations),
  DM_BOUNDARY_MIRROR (not yet implemented for 3d), DM_BOUNDARY_PERIODIC (ghost vertices/cells filled by the opposite
  edge of the domain), or DM_BOUNDARY_TWIST (like periodic, only glued backwards like a Mobius strip).

  Note: This is information for the boundary of the __PHYSICAL__ domain. It has nothing to do with boundaries between
  processes, that width is always determined by the stencil width, see DMDASetStencilWidth().

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum {DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_MIRROR, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_TWIST} DMBoundaryType;

#endif
