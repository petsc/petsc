

#include "petscsys.h"
#include "silo.h"
#include "private/viewerimpl.h"

typedef struct {
  DBfile *file_pointer; /* The PDB file for Silo */
  char   *meshName;     /* The name for the current mesh */
  char   *objName;      /* The name for the next object passed to Silo */
} Viewer_Silo;
