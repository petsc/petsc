#pragma once

#include <petsc/private/viewerimpl.h>
#include <cgnstypes.h>

typedef struct {
  char           *filename_template;
  char           *filename;
  PetscFileMode   btype;
  int             file_num;
  const PetscInt *node_l2g;
  int             base, zone;
  PetscInt        num_local_nodes, nStart, nEnd;
  PetscInt        eStart, eEnd;
  PetscScalar    *nodal_field;
  PetscSegBuffer  output_steps;
  PetscSegBuffer  output_times;
  PetscInt        batch_size;
} PetscViewer_CGNS;

#define PetscCallCGNS(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)

#if !defined(PRIdCGSIZE)
  #if CG_SIZEOF_SIZE == 32
    // cgsize_t is defined as int
    #define MPIU_CGSIZE MPI_INT
    #define PRIdCGSIZE  "d"
  #else
    #if defined(_WIN32)
      // cgsize_t is defined as __int64, which is synonymous with long long
      #define MPIU_CGSIZE MPI_LONG_LONG
      #define PRIdCGSIZE  "lld"
    #else
      // cgsize_t is defined as long
      #define MPIU_CGSIZE MPI_LONG
      #define PRIdCGSIZE  "ld"
    #endif
  #endif
#else
  #if CG_SIZEOF_SIZE == 32
    // cgsize_t is defined as int32_t
    #define MPIU_CGSIZE MPI_INT32_T
  #else
    // cgsize_t is defined as int64_t
    #define MPIU_CGSIZE MPI_INT64_T
  #endif
#endif

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer, PetscInt);
