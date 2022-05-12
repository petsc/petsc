#if !defined(PETSCDMPLEXTYPES_H)
#define PETSCDMPLEXTYPES_H

/* SUBMANSEC = DMPlex */

/*E
  DMPlexShape - The domain shape used for automatic mesh creation.

  Existing shapes include
$ DM_SHAPE_BOX         - The tensor product of intervals in dimension d
$ DM_SHAPE_BOX_SURFACE - The surface of a box in dimension d+1
$ DM_SHAPE_BALL        - The d-dimensional ball
$ DM_SHAPE_SPHERE      - The surface of the (d+1)-dimensional ball
$ DM_SHAPE_CYLINDER    - The tensor product of the interval and disk
$ DM_SHAPE_SCHWARZ_P   - The Schwarz-P triply periodic minimal surface
$ DM_SHAPE_GYROID      - The Gyroid triply periodic minimal surface
$ DM_SHAPE_DOUBLET     - The mesh of two cells of a specified type

  Level: beginner

.seealso: `DMPlexGetCellRefiner()`, `DMPlexSetCellRefiner()`, `DMRefine()`, `DMPolytopeType`
E*/
typedef enum {DM_SHAPE_BOX, DM_SHAPE_BOX_SURFACE, DM_SHAPE_BALL, DM_SHAPE_SPHERE, DM_SHAPE_CYLINDER, DM_SHAPE_SCHWARZ_P, DM_SHAPE_GYROID, DM_SHAPE_DOUBLET, DM_SHAPE_UNKNOWN} DMPlexShape;
PETSC_EXTERN const char * const DMPlexShapes[];

/*E
  DMPlexCSRAlgorithm - The algorithm for building the adjacency graph in CSR format, usually for a mesh partitioner

  Existing shapes include
$ DM_PLEX_CSR_MAT     - Use MatPartition by first making a matrix
$ DM_PLEX_CSR_GRAPH   - Use the original Plex and communicate along the boundary
$ DM_PLEX_CSR_OVERLAP - Build an overlapped Plex and then locally compute

  Level: beginner

.seealso: `DMPlexCreatePartitionerGraph()`, `PetscPartitionerDMPlexPartition()`, `DMPlexDistribute()`
E*/
typedef enum {DM_PLEX_CSR_MAT, DM_PLEX_CSR_GRAPH, DM_PLEX_CSR_OVERLAP} DMPlexCSRAlgorithm;
PETSC_EXTERN const char * const DMPlexCSRAlgorithms[];

typedef struct _p_DMPlexPointQueue *DMPlexPointQueue;
struct _p_DMPlexPointQueue {
  PetscInt  size;   /* Size of the storage array */
  PetscInt *points; /* Array of mesh points */
  PetscInt  front;  /* Index of the front of the queue */
  PetscInt  back;   /* Index of the back of the queue */
  PetscInt  num;    /* Number of enqueued points */
};

#endif
