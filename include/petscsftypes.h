#ifndef PETSCSFTYPES_H
#define PETSCSFTYPES_H

/* SUBMANSEC = PetscSF */

/*S
   PetscSF - PETSc object for setting up and managing the communication of certain entries of arrays and `Vec` between MPI ranks.

   Level: intermediate

       `PetscSF` uses the concept of star forests to indicate and determine the communication patterns concisely and efficiently.
  A star  https://en.wikipedia.org/wiki/Star_(graph_theory) forest is simply a collection of trees of height 1. The leave nodes represent
  "ghost locations" for the root nodes.

.seealso: `PetscSFCreate()`, `VecScatter`, `VecScatterCreate()`
S*/
typedef struct _p_PetscSF *PetscSF;

/*J
    PetscSFType - String with the name of a `PetscSF` type

   Level: beginner

.seealso: `PetscSFSetType()`, `PetscSF`
J*/
typedef const char *PetscSFType;
#define PETSCSFBASIC      "basic"
#define PETSCSFNEIGHBOR   "neighbor"
#define PETSCSFALLGATHERV "allgatherv"
#define PETSCSFALLGATHER  "allgather"
#define PETSCSFGATHERV    "gatherv"
#define PETSCSFGATHER     "gather"
#define PETSCSFALLTOALL   "alltoall"
#define PETSCSFWINDOW     "window"

/*S
   PetscSFNode - specifier of owner and index

   Level: beginner

  Sample Usage:
.vb
    PetscSFNode    *remote;
    PetscCall(PetscMalloc1(nleaves,&remote));
    for (i=0; i<size; i++) {
      remote[i].rank = i;
      remote[i].index = rank;
    }
.ve

  Sample Fortran Usage:
.vb
    type(PetscSFNode) remote(6)
    remote(1)%rank  = modulo(rank+size-1,size)
    remote(1)%index = 1 * stride
.ve

.seealso: `PetscSF`, `PetscSFSetGraph()`
S*/
typedef struct {
  PetscInt rank;  /* Rank of owner */
  PetscInt index; /* Index of node on rank */
} PetscSFNode;

/*S
     VecScatter - Object used to manage communication of data
       between vectors in parallel. Manages both scatters and gathers

   Level: beginner

.seealso: `Vec`, `PetscSF`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`
S*/
typedef PetscSF VecScatter;

/*J
    VecScatterType - String with the name of a PETSc vector scatter type

   Level: beginner

.seealso: `PetscSFType`, `VecScatterSetType()`, `VecScatter`, `VecScatterCreate()`, `VecScatterDestroy()`
J*/
typedef PetscSFType VecScatterType;
#endif
