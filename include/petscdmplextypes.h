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
$ REFINER_REGULAR - Divide cells into smaller cells of the same type
$ REFINER_TO_HEX - Divide all cells into box cells
$ REFINER_TO_SIMPLEX - Divide all cells into simplices

.seealso: DMPlexGetCellRefiner(), DMPlexSetCellRefiner(), DMRefine(), DMPolytopeType
E*/
typedef enum {REFINER_REGULAR, REFINER_TO_HEX, REFINER_TO_SIMPLEX} DMPlexCellRefinerType;
PETSC_EXTERN const char * const DMPlexCellRefinerTypes[];

#endif
