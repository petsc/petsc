#ifndef PETSCISTYPES_H
#define PETSCISTYPES_H

/* SUBMANSEC = IS */

/*S
     IS - Abstract PETSc object used for efficient indexing into vector and matrices

   Level: beginner

.seealso: `ISType`, `ISCreateGeneral()`, `ISCreateBlock()`, `ISCreateStride()`, `ISGetIndices()`, `ISDestroy()`
S*/
typedef struct _p_IS *IS;

/*S
   ISLocalToGlobalMapping - mappings from a
      local ordering (on individual MPI processes) of 0 to n-1 to a global PETSc ordering (across collections of MPI processes)
      used by a vector or matrix.

   Level: intermediate

   Note:
   Mapping from local to global is scalable; but global
   to local may not be if the range of global values represented locally
   is very large. `ISLocalToGlobalMappingType` provides alternative ways of efficiently applying `ISGlobalToLocalMappingApply()

   Developer Note:
   `ISLocalToGlobalMapping` is actually a private object; it is included
   here for the inline function `ISLocalToGlobalMappingApply()` to allow it to be inlined since
   it is used so often.

.seealso: `ISLocalToGlobalMappingCreate()`, `ISLocalToGlobalMappingApply()`, `ISLocalToGlobalMappingDestroy()`, `ISGlobalToLocalMappingApply()`
S*/
typedef struct _p_ISLocalToGlobalMapping *ISLocalToGlobalMapping;

/*S
     ISColoring - sets of IS's that define a coloring of something, such as a graph defined by a sparse matrix

   Level: intermediate

    Notes:
    One should not access the *is records below directly because they may not yet
    have been created. One should use `ISColoringGetIS()` to make sure they are
    created when needed.

    When the coloring type is `IS_COLORING_LOCAL` the coloring is in the local ordering of the unknowns.
    That is the matching the local (ghosted) vector; a local to global mapping must be applied to map
    them to the global ordering.

    Developer Note:
    This is not a `PetscObject`

.seealso: `IS`, `MatColoringCreate()`, `MatColoring`, `ISColoringCreate()`, `ISColoringGetIS()`, `ISColoringView()`
S*/
typedef struct _n_ISColoring *ISColoring;

/*S
     PetscLayout - defines layout of vectors and matrices (that is the "global" numbering of vector and matrix entries) across MPI processes (which rows are owned by which processes)

   Level: developer

   Notes:
   PETSc vectors (`Vec`) have a global number associated with each vector entry. The first MPI process that shares the vector owns the first `n0` entries of the vector,
   the second MPI process the next `n1` entries, etc. A `PetscLayout` is a way of managing this information, for example the number of locally owned entries is provided
   by `PetscLayoutGetLocalSize()` and the range of indices for a given MPI process is provided by `PetscLayoutGetRange()`.

   Each PETSc `Vec` contains a `PetscLayout` object which can be obtained with `VecGetLayout()`. For convinence `Vec` provides an API to access the layout information directly,
   for example with `VecGetLocalSize()` and `VecGetOwnershipRange()`.

   Similarly PETSc matrices have layouts, these are discussed in [](chapter_matrices).

.seealso: `PetscLayoutCreate()`, `PetscLayoutDestroy()`, `PetscLayoutGetRange()`, `PetscLayoutGetLocalSize()`, `PetscLayoutGetSize()`,
          `PetscLayoutGetBlockSize()`, `PetscLayoutGetRanges()`,  `PetscLayoutFindOwner()`,  `PetscLayoutFindOwnerIndex()`,
          `VecGetLayout()`, `VecGetLocalSize()`, `VecGetOwnershipRange()`
S*/
typedef struct _n_PetscLayout *PetscLayout;

#endif
