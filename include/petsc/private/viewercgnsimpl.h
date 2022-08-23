#ifndef viewercgnsimpl_h
#define viewercgnsimpl_h

#include <petsc/private/viewerimpl.h>
#include <cgnstypes.h>

typedef struct {
  char           *filename;
  PetscFileMode   btype;
  int             file_num;
  PetscBool       parallel;
  const PetscInt *node_l2g;
  int             base, zone;
  PetscInt        num_local_nodes, nStart, nEnd;
  PetscScalar    *nodal_field;
  PetscSegBuffer  output_times;
} PetscViewer_CGNS;

#define PetscCallCGNS(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)

#endif
