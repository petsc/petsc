#if !defined(PETSCDMTYPES_H)
#define PETSCDMTYPES_H

/*S
     DM - Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers

   Level: intermediate

   Notes:
    The DMDACreate() based object and the DMCompositeCreate() based object are examples of DMs

.seealso:  DMCompositeCreate(), DMDACreate(), DMSetType(), DMType
S*/
typedef struct _p_DM* DM;

/*E
  DMBoundaryType - Describes the choice for fill of ghost cells on physical domain boundaries.

  Level: beginner

  A boundary may be of type DM_BOUNDARY_NONE (no ghost nodes), DM_BOUNDARY_GHOSTED (ghost vertices/cells
  exist but aren't filled; you can put values into them and then apply a stencil that uses those ghost locations),
  DM_BOUNDARY_MIRROR (the ghost value is the same as the value 1 grid point in; that is, the 0th grid point in the real mesh acts like a mirror to define the ghost point value;
  not yet implemented for 3d), DM_BOUNDARY_PERIODIC (ghost vertices/cells filled by the opposite
  edge of the domain), or DM_BOUNDARY_TWIST (like periodic, only glued backwards like a Mobius strip).

  Notes:
  This is information for the boundary of the __PHYSICAL__ domain. It has nothing to do with boundaries between
  processes. That width is always determined by the stencil width; see DMDASetStencilWidth().

  If the physical grid points have values 0 1 2 3 with DM_BOUNDARY_MIRROR then the local vector with ghost points has the values 1 0 1 2 3 2 .

  Developer Notes:
    Should DM_BOUNDARY_MIRROR have the same meaning with DMDA_Q0, that is a staggered grid? In that case should the ghost point have the same value
  as the 0th grid point where the physical boundary serves as the mirror?

  References:
  https://scicomp.stackexchange.com/questions/5355/writing-the-poisson-equation-finite-difference-matrix-with-neumann-boundary-cond

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum {DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_MIRROR, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_TWIST} DMBoundaryType;
/*E
  DMBoundaryConditionType - indicates what type of boundary condition is to be imposed

  Note: This flag indicates the type of function which will define the condition:
$ DM_BC_ESSENTIAL       - A Dirichlet condition using a function of the coordinates
$ DM_BC_ESSENTIAL_FIELD - A Dirichlet condition using a function of the coordinates and auxiliary field data
$ DM_BC_ESSENTIAL_BD_FIELD - A Dirichlet condition using a function of the coordinates, facet normal, and auxiliary field data
$ DM_BC_NATURAL         - A Neumann condition using a function of the coordinates
$ DM_BC_NATURAL_FIELD   - A Neumann condition using a function of the coordinates and auxiliary field data
$ DM_BC_NATURAL_RIEMANN - A flux condition which determines the state in ghost cells
The user can check whether a boundary condition is essential using (type & DM_BC_ESSENTIAL), and similarly for
natural conditions (type & DM_BC_NATURAL)

  Level: beginner

.seealso: DMAddBoundary(), DMGetBoundary()
E*/
typedef enum {DM_BC_ESSENTIAL = 1, DM_BC_ESSENTIAL_FIELD = 5, DM_BC_NATURAL = 2, DM_BC_NATURAL_FIELD = 6, DM_BC_ESSENTIAL_BD_FIELD = 9, DM_BC_NATURAL_RIEMANN = 10} DMBoundaryConditionType;

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

/*E
  DMAdaptationStrategy - Describes the strategy used for adaptive solves

  Level: beginner

  DM_ADAPTATION_INITIAL will refine a mesh based on an initial guess. DM_ADAPTATION_SEQUENTIAL will refine the
  mesh based on a sequence of solves, much like grid sequencing. DM_ADAPTATION_MULTILEVEL will use the sequence
  of constructed meshes in a multilevel solve, much like the Systematic Upscaling of Brandt.

.seealso: DMAdaptorSolve()
E*/
typedef enum {DM_ADAPTATION_INITIAL, DM_ADAPTATION_SEQUENTIAL, DM_ADAPTATION_MULTILEVEL} DMAdaptationStrategy;

/*E
  DMAdaptationCriterion - Describes the test used to decide whether to coarsen or refine parts of the mesh

  Level: beginner

  DM_ADAPTATION_REFINE will uniformly refine a mesh, much like grid sequencing. DM_ADAPTATION_LABEL will adapt
  the mesh based upon a label of the cells filled with DMAdaptFlag markers. DM_ADAPTATION_METRIC will try to
  mesh the manifold described by the input metric tensor uniformly. PETSc can also construct such a metric based
  upon an input primal or a gradient field.

.seealso: DMAdaptorSolve()
E*/
typedef enum {DM_ADAPTATION_NONE, DM_ADAPTATION_REFINE, DM_ADAPTATION_LABEL, DM_ADAPTATION_METRIC} DMAdaptationCriterion;

/*E
  DMAdaptFlag - Marker in the label prescribing adaptation

  Level: beginner

.seealso: DMAdaptLabel()
E*/
typedef enum {DM_ADAPT_DETERMINE = PETSC_DETERMINE, DM_ADAPT_KEEP = 0, DM_ADAPT_REFINE, DM_ADAPT_COARSEN, DM_ADAPT_COARSEN_LAST, DM_ADAPT_RESERVED_COUNT} DMAdaptFlag;

/*E
  DMDirection - Indicates a coordinate direction

  Level: beginner

.seealso: DMDAGetRay(), DMDAGetProcessorSubset(), DMPlexShearGeometry()
E*/
typedef enum {DM_X, DM_Y, DM_Z} DMDirection;

/*E
DMEnclosureType - The type of enclosure relation between one DM and another

Level: beginner

For example, one DM dmA may be the boundary of another dmB, in which case it would be labeled DM_ENC_SUBMESH. If
the situation is reversed, and dmA has boundary dmB, it would be labeled DM_ENC_SUPERMESH. Likewise, if dmA was
a subregion of dmB, it would be labeled DM_ENC_SUBMESH. If no relation can be determined, DM_ENC_NONE is used.
If a relation is not yet known, then DM_ENC_UNKNOWN is used.

.seealso: DMGetEnclosureRelation()
E*/
typedef enum {DM_ENC_EQUALITY, DM_ENC_SUPERMESH, DM_ENC_SUBMESH, DM_ENC_NONE, DM_ENC_UNKNOWN} DMEnclosureType;

/*E
  DMPolytopeType - This describes the polytope represented by each cell.

  Level: beginner

  While most operations only need the topology information in the Plex, we must sometimes have the
  user specify a polytope. For instance, when interpolating from a cell-vertex mesh, the type of
  polytope can be ambiguous. Also, Plex allows different symmetries of prism cell with the same
  constituent points. Normally these types are autoamtically inferred and the user does not specify
  them.

.seealso: DMPlexComputeCellTypes()
E*/
typedef enum {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_POINT_PRISM_TENSOR, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_SEG_PRISM_TENSOR, DM_POLYTOPE_TETRAHEDRON, DM_POLYTOPE_HEXAHEDRON, DM_POLYTOPE_TRI_PRISM, DM_POLYTOPE_TRI_PRISM_TENSOR, DM_POLYTOPE_QUAD_PRISM_TENSOR, DM_POLYTOPE_FV_GHOST, DM_POLYTOPE_INTERIOR_GHOST, DM_POLYTOPE_UNKNOWN, DM_NUM_POLYTOPES} DMPolytopeType;
PETSC_EXTERN const char *const DMPolytopeTypes[];

/*E
  PetscUnit - The seven fundamental SI units

  Level: beginner

.seealso: DMPlexGetScale(), DMPlexSetScale()
E*/
typedef enum {PETSC_UNIT_LENGTH, PETSC_UNIT_MASS, PETSC_UNIT_TIME, PETSC_UNIT_CURRENT, PETSC_UNIT_TEMPERATURE, PETSC_UNIT_AMOUNT, PETSC_UNIT_LUMINOSITY, NUM_PETSC_UNITS} PetscUnit;

/*S
    DMField - PETSc object for defining a field on a mesh topology

    Level: intermediate
S*/
typedef struct _p_DMField* DMField;

#endif
