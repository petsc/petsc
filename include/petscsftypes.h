#if !defined(_PETSCSFTYPES_H)
#define _PETSCSFTYPES_H

/*S
   PetscSF - PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.

   Level: intermediate

  Concepts: star forest

       PetscSF uses the concept of star forests to indicate and determine the communication patterns concisely and efficiently.
  A star  http://en.wikipedia.org/wiki/Star_(graph_theory) forest is simply a collection of trees of height 1. The leave nodes represent
  "ghost locations" for the root nodes.

.seealso: PetscSFCreate(), VecScatter, VecScatterCreate()
S*/
typedef struct _p_PetscSF* PetscSF;

/*S
   PetscSFNode - specifier of owner and index

   Level: beginner

  Concepts: indexing, stride, distribution

.seealso: PetscSFSetGraph()
S*/
typedef struct {
  PetscInt rank;                /* Rank of owner */
  PetscInt index;               /* Index of node on rank */
} PetscSFNode;

#endif
