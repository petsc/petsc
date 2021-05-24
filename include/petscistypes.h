#if !defined(PETSCISTYPES_H)
#define PETSCISTYPES_H

/*S
     IS - Abstract PETSc object that allows indexing.

   Level: beginner

.seealso:  ISCreateGeneral(), ISCreateBlock(), ISCreateStride(), ISGetIndices(), ISDestroy()
S*/
typedef struct _p_IS* IS;

/*S
   ISLocalToGlobalMapping - mappings from an arbitrary
      local ordering from 0 to n-1 to a global PETSc ordering
      used by a vector or matrix.

   Level: intermediate

   Note: mapping from local to global is scalable; but global
  to local may not be if the range of global values represented locally
  is very large.

   Note: the ISLocalToGlobalMapping is actually a private object; it is included
  here for the inline function ISLocalToGlobalMappingApply() to allow it to be inlined since
  it is used so often.

.seealso:  ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingDestroy()
S*/
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;

/*S
     ISColoring - sets of IS's that define a coloring
              of the underlying indices

   Level: intermediate

    Notes:
        One should not access the *is records below directly because they may not yet
    have been created. One should use ISColoringGetIS() to make sure they are
    created when needed.

        When the coloring type is IS_COLORING_LOCAL the coloring is in the local ordering of the unknowns.
    That is the matching the local (ghosted) vector; a local to global mapping must be applied to map
    them to the global ordering.

    Developer Note: this is not a PetscObject

.seealso:  ISColoringCreate(), ISColoringGetIS(), ISColoringView()
S*/
typedef struct _n_ISColoring* ISColoring;

/*S
     PetscLayout - defines layout of vectors and matrices across processes (which rows are owned by which processes)

   Level: developer

.seealso:  PetscLayoutCreate(), PetscLayoutDestroy()
S*/
typedef struct _n_PetscLayout* PetscLayout;

#endif
