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

#endif
