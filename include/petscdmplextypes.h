#if !defined(_PETSCDMPLEXTYPES_H)
#define _PETSCDMPLEXTYPES_H

/*E
  DMPlexCellType - Common mesh celltypes

  Level: beginner

  Plex can handle any cell shape, but sometimes we have to determine things about a mesh that the user
  does not specify, and for this we have to make assumptions about the mesh. One very common assumption
  is that all cells in the mesh take a certain form. For example, in order to interpolate a mesh (create
  edges and faces automatically) we might assume that all cells are simples, or are tensor product cells.

.seealso: DMDASetBoundaryType(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDACreate()
E*/
typedef enum {DM_PLEX_CELLTYPE_SIMPLEX, DM_PLEX_CELLTYPE_TENSOR, DM_PLEX_CELLTYPE_UNKNOWN} DMPlexCellType;

#endif
