#pragma once

/* MANSEC = Vec */
/* SUBMANSEC = PetscSF */

/*S
   PetscSF - PETSc object for managing the communication of certain entries of arrays and `Vec` between MPI processes.

   Level: intermediate

  `PetscSF` uses the concept of star forests to indicate and determine the communication patterns concisely and efficiently.
  A star  <https://en.wikipedia.org/wiki/Star_(graph_theory)> forest is simply a collection of trees of height 1. The leave nodes represent
  "ghost locations" for the root nodes.

  The standard usage paradigm for `PetscSF` is to provide the communication pattern with `PetscSFSetGraph()` or `PetscSFSetGraphWithPattern()` and
  then perform the communication using `PetscSFBcastBegin()` and `PetscSFBcastEnd()`, `PetscSFReduceBegin()` and `PetscSFReduceEnd()`.

.seealso: [](sec_petscsf), `PetscSFCreate()`, `PetscSFSetGraph()`, `PetscSFSetGraphWithPattern()`, `PetscSFBcastBegin()`, `PetscSFBcastEnd()`,
          `PetscSFReduceBegin()`, `PetscSFReduceEnd()`, `VecScatter`, `VecScatterCreate()`
S*/
typedef struct _p_PetscSF *PetscSF;

/*J
  PetscSFType - String with the name of a `PetscSF` type. Each `PetscSFType` uses different mechanisms to perform the communication.

  Level: beginner

  Available Types:
+ `PETSCSFBASIC`      - use MPI sends and receives
. `PETSCSFNEIGHBOR`   - use MPI_Neighbor operations
. `PETSCSFALLGATHERV` - use MPI_Allgatherv operations
. `PETSCSFALLGATHER`  - use MPI_Allgather operations
. `PETSCSFGATHERV`    - use MPI_Igatherv and MPI_Iscatterv operations
. `PETSCSFGATHER`     - use MPI_Igather and MPI_Iscatter operations
. `PETSCSFALLTOALL`   - use MPI_Ialltoall operations
- `PETSCSFWINDOW`     - use MPI_Win operations

  Note:
  Some `PetscSFType` only provide specialized code for a subset of the `PetscSF` operations and use `PETSCSFBASIC` for the others.

.seealso: [](sec_petscsf), `PetscSFSetType()`, `PetscSF`
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
   PetscSFNode - specifier of MPI rank owner and local index for array or `Vec` entry locations that are to be communicated with a `PetscSF`

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

  Notes:
  Use  `MPIU_SF_NODE` when performing MPI operations on arrays of `PetscSFNode`

  Generally the values of `rank` should be in $[ 0,size)$  and the value of `index` greater than or equal to 0, but there are some situations that violate this.

.seealso: [](sec_petscsf), `PetscSF`, `PetscSFSetGraph()`
S*/
typedef struct {
  PetscInt rank;  /* MPI rank of owner */
  PetscInt index; /* Index of node on rank */
} PetscSFNode;

#define MPIU_SF_NODE MPIU_2INT

typedef enum {
  PETSCSF_ROOT2LEAF = 0,
  PETSCSF_LEAF2ROOT = 1
} PetscSFDirection;
typedef enum {
  PETSCSF_BCAST  = 0,
  PETSCSF_REDUCE = 1,
  PETSCSF_FETCH  = 2
} PetscSFOperation;
/* When doing device-aware MPI, a backend refers to the SF/device interface */
typedef enum {
  PETSCSF_BACKEND_INVALID = 0,
  PETSCSF_BACKEND_CUDA    = 1,
  PETSCSF_BACKEND_HIP     = 2,
  PETSCSF_BACKEND_KOKKOS  = 3
} PetscSFBackend;
typedef struct _n_PetscSFLink *PetscSFLink;

/*S
  VecScatter - Object used to manage communication of data
  between vectors in parallel or between parallel and sequential vectors. Manages both scatters and gathers

  Level: beginner

  Note:
  This is an alias for `PetscSF`.

.seealso: [](sec_petscsf), `Vec`, `PetscSF`, `VecScatterCreate()`, `VecScatterBegin()`, `VecScatterEnd()`
S*/
typedef PetscSF VecScatter;

/*J
  VecScatterType - String with the name of a PETSc vector scatter type

  Level: beginner

  Note:
  This is an alias for `PetscSFType`

.seealso: [](sec_petscsf), `PetscSFType`, `VecScatterSetType()`, `VecScatter`, `VecScatterCreate()`, `VecScatterDestroy()`
J*/
typedef PetscSFType VecScatterType;
