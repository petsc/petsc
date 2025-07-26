#pragma once

#include <petsc/private/viewerimpl.h>
#include <cgnstypes.h>
#include <cgnslib.h>

PETSC_EXTERN PetscLogEvent PETSC_VIEWER_CGNS_Open, PETSC_VIEWER_CGNS_Close, PETSC_VIEWER_CGNS_ReadMeta, PETSC_VIEWER_CGNS_WriteMeta, PETSC_VIEWER_CGNS_ReadData, PETSC_VIEWER_CGNS_WriteData;

typedef struct {
  char           *filename_template;
  char           *filename;
  PetscFileMode   btype;
  int             file_num;
  const PetscInt *node_l2g;
  int             base, zone;
  CGNS_ENUMT(GridLocation_t) grid_loc;
  PetscInt       num_local_nodes, nStart, nEnd;
  PetscInt       eStart, eEnd;
  PetscScalar   *nodal_field;
  PetscSegBuffer output_steps;
  PetscSegBuffer output_times;
  PetscInt       previous_output_step;
  PetscInt       batch_size;

  // Solution reading information
  PetscInt solution_index;              // User set solution index
  int      solution_file_index;         // CGNS file solution index for direct access
  int      solution_file_pointer_index; // CGNS file solution index for FlowSolutionPointers (and other related arrays), index by 1
  char    *solution_name;
} PetscViewer_CGNS;

#define PetscCallCGNS(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)

#define PetscCallCGNSOpen(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_Open, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_Open, o1, o2, 0, 0)); \
  } while (0)

#define PetscCallCGNSClose(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_Close, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_Close, o1, o2, 0, 0)); \
  } while (0)

#define PetscCallCGNSRead(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_ReadMeta, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_ReadMeta, o1, o2, 0, 0)); \
  } while (0)

#define PetscCallCGNSReadData(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_ReadData, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_ReadData, o1, o2, 0, 0)); \
  } while (0)

#define PetscCallCGNSWrite(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_WriteMeta, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_WriteMeta, o1, o2, 0, 0)); \
  } while (0)

#define PetscCallCGNSWriteData(ierr, o1, o2) \
  do { \
    PetscCall(PetscLogEventBegin(PETSC_VIEWER_CGNS_WriteData, o1, o2, 0, 0)); \
    PetscCallCGNS((ierr)); \
    PetscCall(PetscLogEventEnd(PETSC_VIEWER_CGNS_WriteData, o1, o2, 0, 0)); \
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

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSRegisterLogEvents_Internal();
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer, PetscInt);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSGetSolutionFileIndex_Internal(PetscViewer, int *);
