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
$ DM_REFINER_REGULAR      - Divide cells into smaller cells of the same type
$ DM_REFINER_TO_BOX       - Divide all cells into box cells
$ DM_REFINER_TO_SIMPLEX   - Divide all cells into simplices
$ DM_REFINER_ALFELD2D     - Alfeld barycentric refinement of triangles
$ DM_REFINER_ALFELD3D     - Alfeld barycentric refinement of tetrahedra
$ DM_REFINER_POWELL_SABIN - Powell-Sabin barycentric refinement of simplices (unfinished)
$ DM_REFINER_BL           - Refine only tensor cells in the tensor direction, often used to refine boundary layers

.seealso: DMPlexGetCellRefiner(), DMPlexSetCellRefiner(), DMRefine(), DMPolytopeType
E*/
typedef enum {DM_REFINER_REGULAR, DM_REFINER_TO_BOX, DM_REFINER_TO_SIMPLEX, DM_REFINER_ALFELD2D, DM_REFINER_ALFELD3D, DM_REFINER_POWELL_SABIN, DM_REFINER_BL} DMPlexCellRefinerType;
PETSC_EXTERN const char * const DMPlexCellRefinerTypes[];

#endif
