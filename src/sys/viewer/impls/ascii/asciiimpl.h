
#include <petsc-private/viewerimpl.h>  /*I     "petscsys.h"   I*/
#include <stdarg.h>

typedef struct {
  FILE          *fd;
  PetscFileMode mode;           /* The mode in which to open the file */
  PetscInt      tab;            /* how many times text is tabbed in from left */
  PetscInt      tab_store;      /* store tabs value while tabs are turned off */
  PetscViewer   bviewer;        /* if PetscViewer is a singleton, this points to mother */
  PetscViewer   sviewer;        /* if PetscViewer has a singleton, this points to singleton */
  char          *filename;
  PetscBool     storecompressed; 
  PetscBool     closefile;       
  PetscBool     allowsynchronized; /* allow synchronized writes from any process to the viewer */
} PetscViewer_ASCII;

typedef struct PetscViewerLink_t PetscViewerLink;
struct PetscViewerLink_t {
  PetscViewer              viewer;
  struct PetscViewerLink_t *next;
};

extern PetscMPIInt Petsc_Viewer_keyval;

EXTERN_C_BEGIN
extern PetscMPIInt  MPIAPI Petsc_DelViewer(MPI_Comm,PetscMPIInt,void*,void*);
EXTERN_C_END
