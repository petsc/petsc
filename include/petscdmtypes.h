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

  A boundary may be of type DM_BOUNDARY_NONE (no ghost nodes), DM_BOUNDARY_GHOSTED (ghost vertices/cells
  exist but aren't filled, you can put values into them and then apply a stencil that uses those ghost locations),
  DM_BOUNDARY_MIRROR (the ghost value is the same as the value 1 grid point in; that is the 0th grid point in the real mesh acts like a mirror to define the ghost point value; 
  not yet implemented for 3d), DM_BOUNDARY_PERIODIC (ghost vertices/cells filled by the opposite
  edge of the domain), or DM_BOUNDARY_TWIST (like periodic, only glued backwards like a Mobius strip).

  Note: This is information for the boundary of the __PHYSICAL__ domain. It has nothing to do with boundaries between
  processes, that width is always determined by the stencil width, see DMDASetStencilWidth().

  Note: If the physical grid points have values  0 1 2 3 with DM_BOUNDARY_MIRROR then the local vector with ghost points has the values 1 0 1 2 3 2

  Developer notes: Should DM_BOUNDARY_MIRROR have the same meaning with DMDA_Q0, that is a staggered grid? In that case should the ghost point have the same value
  as the 0th grid point where the physical boundary serves as the mirror?

  References: http://scicomp.stackexchange.com/questions/5355/writing-the-poisson-equation-finite-difference-matrix-with-neumann-boundary-cond

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum {DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_MIRROR, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_TWIST} DMBoundaryType;

/*E
  DMPointLocationType - Describes the method to handle point location failure

  Level: beginner

  If a search using DM_POINTLOCATION_NONE fails, the failure is signaled with a negative cell number. On the
  other hand, if DM_POINTLOCATION_NEAREST is used, on failure, the (approximate) nearest point in the mesh is
  used, replacing the given point in the input vector. DM_POINTLOCATION_REMOVE returns values only for points
  which were located.

.seealso: DMLocatePoints()
E*/
typedef enum {DM_POINTLOCATION_NONE, DM_POINTLOCATION_NEAREST, DM_POINTLOCATION_REMOVE} DMPointLocationType;

/*S
  PetscPartitioner - PETSc object that manages a graph partitioner

  Level: intermediate

  Concepts: partition, mesh

.seealso: PetscPartitionerCreate(), PetscPartitionerSetType(), PetscPartitionerType
S*/
typedef struct _p_PetscPartitioner *PetscPartitioner;

#endif
