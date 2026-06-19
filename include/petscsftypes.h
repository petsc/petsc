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

/*E
   PetscSFDirection - Direction in which a `PetscSF` communication is performed

   Values:
+   `PETSCSF_ROOT2LEAF` - data flows from roots to leaves (broadcast direction)
-   `PETSCSF_LEAF2ROOT` - data flows from leaves to roots (reduce/fetch direction)

   Level: developer

.seealso: [](sec_petscsf), `PetscSF`, `PetscSFOperation`, `PetscSFBcastBegin()`, `PetscSFReduceBegin()`
E*/
typedef enum {
  PETSCSF_ROOT2LEAF = 0,
  PETSCSF_LEAF2ROOT = 1
} PetscSFDirection;

/*E
   PetscSFOperation - Identifies the high-level operation being performed by a `PetscSF` communication

   Values:
+   `PETSCSF_BCAST`  - broadcast from roots to leaves
.   `PETSCSF_REDUCE` - reduce from leaves to roots with an `MPI_Op`
-   `PETSCSF_FETCH`  - fetch-and-op: each leaf receives the current root value and contributes its own

   Level: developer

.seealso: [](sec_petscsf), `PetscSF`, `PetscSFDirection`, `PetscSFBcastBegin()`, `PetscSFReduceBegin()`, `PetscSFFetchAndOpBegin()`
E*/
typedef enum {
  PETSCSF_BCAST  = 0,
  PETSCSF_REDUCE = 1,
  PETSCSF_FETCH  = 2
} PetscSFOperation;

/*E
   PetscSFBackend - Device backend used by a `PetscSF` to pack, unpack, and exchange data when doing device-aware communication

   Values:
+   `PETSCSF_BACKEND_INVALID` - no backend has been selected (the default for a host-only SF)
.   `PETSCSF_BACKEND_CUDA`    - use CUDA-aware pack/unpack and MPI
.   `PETSCSF_BACKEND_HIP`     - use HIP-aware pack/unpack and MPI
-   `PETSCSF_BACKEND_KOKKOS`  - use the Kokkos backend (which itself may dispatch to CUDA, HIP, or OpenMP)

   Level: developer

.seealso: [](sec_petscsf), `PetscSF`, `PetscSFLink`, `PetscSFDirection`, `PetscSFOperation`
E*/
/* When doing device-aware MPI, a backend refers to the SF/device interface */
typedef enum {
  PETSCSF_BACKEND_INVALID = 0,
  PETSCSF_BACKEND_CUDA    = 1,
  PETSCSF_BACKEND_HIP     = 2,
  PETSCSF_BACKEND_KOKKOS  = 3
} PetscSFBackend;
/*S
  PetscSFLink - Opaque internal scratch object used by `PetscSF` to pair a packed buffer with the appropriate pack/unpack and MPI operations for a given root-data layout and `PetscSFBackend`

  Level: developer

.seealso: `PetscSF`, `PetscSFBackend`, `PetscSFBcastBegin()`, `PetscSFReduceBegin()`
S*/
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
