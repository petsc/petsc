#ifndef viewercgnsimpl_h
#define viewercgnsimpl_h

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

PETSC_EXTERN PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer, PetscInt);

#endif
