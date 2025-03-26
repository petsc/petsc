#pragma once

/* MANSEC = DM */
/* SUBMANSEC = DMPlex */

/*E
  DMPlexShape - The domain shape used for automatic mesh creation.

  Values:
+ `DM_SHAPE_BOX`         - The tensor product of intervals in dimension d
. `DM_SHAPE_BOX_SURFACE` - The surface of a box in dimension d+1
. `DM_SHAPE_BALL`        - The d-dimensional ball
. `DM_SHAPE_SPHERE`      - The surface of the (d+1)-dimensional ball
. `DM_SHAPE_CYLINDER`    - The tensor product of the interval and disk
. `DM_SHAPE_SCHWARZ_P`   - The Schwarz-P triply periodic minimal surface
. `DM_SHAPE_GYROID`      - The Gyroid triply periodic minimal surface
. `DM_SHAPE_DOUBLET`     - The mesh of two cells of a specified type
. `DM_SHAPE_ANNULUS`     - The area between two concentric spheres in dimension d
- `DM_SHAPE_HYPERCUBIC`  - The skeleton of the tensor product of the intervals

  Level: beginner

.seealso: [](ch_dmbase), `DMPLEX`, `DMPlexGetCellRefiner()`, `DMPlexSetCellRefiner()`, `DMRefine()`, `DMPolytopeType`, `DMPlexCoordMap`
E*/
typedef enum {
  DM_SHAPE_BOX,
  DM_SHAPE_BOX_SURFACE,
  DM_SHAPE_BALL,
  DM_SHAPE_SPHERE,
  DM_SHAPE_CYLINDER,
  DM_SHAPE_SCHWARZ_P,
  DM_SHAPE_GYROID,
  DM_SHAPE_DOUBLET,
  DM_SHAPE_ANNULUS,
  DM_SHAPE_HYPERCUBIC,
  DM_SHAPE_ZBOX,
  DM_SHAPE_UNKNOWN
} DMPlexShape;
PETSC_EXTERN const char *const DMPlexShapes[];

/*E
  DMPlexCoordMap - The coordinate mapping used for automatic mesh creation.

  Values:
+ `DM_COORD_MAP_NONE`     - The identity map
. `DM_COORD_MAP_SHEAR`    - The shear (additive) map along some dimension
. `DM_COORD_MAP_FLARE`    - The flare (multiplicative) map along some dimension
. `DM_COORD_MAP_ANNULUS`  - The map from a rectangle to an annulus
. `DM_COORD_MAP_SHELL`    - The map from a rectangular solid to an spherical shell
- `DM_COORD_MAP_SINUSOID` - The map from a flat rectangle to a sinusoidal surface

  Level: beginner

.seealso: [](ch_dmbase), `DMPLEX`, `DMPlexGetCellRefiner()`, `DMPlexSetCellRefiner()`, `DMRefine()`, `DMPolytopeType`, `DMPlexShape`
E*/
typedef enum {
  DM_COORD_MAP_NONE,
  DM_COORD_MAP_SHEAR,
  DM_COORD_MAP_FLARE,
  DM_COORD_MAP_ANNULUS,
  DM_COORD_MAP_SHELL,
  DM_COORD_MAP_SINUSOID,
  DM_COORD_MAP_UNKNOWN
} DMPlexCoordMap;
PETSC_EXTERN const char *const DMPlexCoordMaps[];

/*E
  DMPlexCSRAlgorithm - The algorithm for building the adjacency graph in CSR format, usually for a mesh partitioner

  Values:
+ `DM_PLEX_CSR_MAT`     - Use `MatPartitioning` by first making a matrix
. `DM_PLEX_CSR_GRAPH`   - Use the original `DMPLEX` and communicate along the boundary
- `DM_PLEX_CSR_OVERLAP` - Build an overlapped `DMPLEX` and then locally compute

  Level: beginner

.seealso: [](ch_dmbase), `DMPLEX`, `DMPlexCreatePartitionerGraph()`, `PetscPartitionerDMPlexPartition()`, `DMPlexDistribute()`
E*/
typedef enum {
  DM_PLEX_CSR_MAT,
  DM_PLEX_CSR_GRAPH,
  DM_PLEX_CSR_OVERLAP
} DMPlexCSRAlgorithm;
PETSC_EXTERN const char *const DMPlexCSRAlgorithms[];

typedef struct _n_DMPlexPointQueue *DMPlexPointQueue;
struct _n_DMPlexPointQueue {
  PetscInt  size;   /* Size of the storage array */
  PetscInt *points; /* Array of mesh points */
  PetscInt  front;  /* Index of the front of the queue */
  PetscInt  back;   /* Index of the back of the queue */
  PetscInt  num;    /* Number of enqueued points */
};
