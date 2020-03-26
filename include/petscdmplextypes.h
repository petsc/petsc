#if !defined(PETSCDMPLEXTYPES_H)
#define PETSCDMPLEXTYPES_H

/*S
  DMPlexCellRefiner - Object encapsulating the refinement strategy for a DMPlex

  Level: developer

.seealso:  DMPlexCellRefinerCreate(), DMType
S*/
typedef struct _p_DMPlexCellRefiner *DMPlexCellRefiner;


/*E
  DMPlexCellRefinerType - This describes the strategy used to refine cells.

  Level: beginner

  The strategy gives a prescription for refining each cell type. Existing strategies include
$ DM_REFINER_REGULAR - Divide cells into smaller cells of the same type
$ DM_REFINER_TO_BOX - Divide all cells into box cells
$ DM_REFINER_TO_SIMPLEX - Divide all cells into simplices

.seealso: DMPlexGetCellRefiner(), DMPlexSetCellRefiner(), DMRefine(), DMPolytopeType
E*/
typedef enum {DM_REFINER_REGULAR, DM_REFINER_TO_BOX, DM_REFINER_TO_SIMPLEX} DMPlexCellRefinerType;
PETSC_EXTERN const char * const DMPlexCellRefinerTypes[];

#endif
