#if !defined(PETSCDMPLEXTYPES_H)
#define PETSCDMPLEXTYPES_H

/*E
  DMPlexShape - The domain shape used for automatic mesh creation.

  Existing shapes include
$ DM_SHAPE_BOX         - The tensor product of intervals in dimension d
$ DM_SHAPE_BOX_SURFACE - The surface of a box in dimension d+1
$ DM_SHAPE_BALL        - The d-dimensional ball
$ DM_SHAPE_SPHERE      - The surface of the (d+1)-dimensional ball
$ DM_SHAPE_CYLINDER    - The tensor product of the interval and disk

  Level: beginner

.seealso: DMPlexGetCellRefiner(), DMPlexSetCellRefiner(), DMRefine(), DMPolytopeType
E*/
typedef enum {DM_SHAPE_BOX, DM_SHAPE_BOX_SURFACE, DM_SHAPE_BALL, DM_SHAPE_SPHERE, DM_SHAPE_CYLINDER, DM_SHAPE_UNKNOWN} DMPlexShape;
PETSC_EXTERN const char * const DMPlexShapes[];

/*E
  DMPlexCSRAlgorithm - The algorithm for building the adjacency graph in CSR format, usually for a mesh partitioner

  Existing shapes include
$ DM_PLEX_CSR_MAT     - Use MatPartition by first making a matrix
$ DM_PLEX_CSR_GRAPH   - Use the original Plex and communicate along the boundary
$ DM_PLEX_CSR_OVERLAP - Build an overlapped Plex and then locally compute

  Level: beginner

.seealso: DMPlexCreatePartitionerGraph(), PetscPartitionerDMPlexPartition(), DMPlexDistribute()
E*/
typedef enum {DM_PLEX_CSR_MAT, DM_PLEX_CSR_GRAPH, DM_PLEX_CSR_OVERLAP} DMPlexCSRAlgorithm;
PETSC_EXTERN const char * const DMPlexCSRAlgorithms[];

#endif
