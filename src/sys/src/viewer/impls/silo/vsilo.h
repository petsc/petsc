/* $Id: vsilo.h,v 1.2 1999/11/20 21:47:40 knepley Exp $ */
/* 
   This is the definition of the SILO viewer structure.
   Note: each viewer has a different data structure.
*/

#include "petscconfig.h"
#include "petsc.h"
#include "petscsys.h" 
#ifdef HAVE_SILO
  #include "silo.h"
#endif

#include "src/sys/src/viewer/viewerimpl.h"

typedef struct {
#ifdef HAVE_SILO
  DBfile *file_pointer; /* The PDB file for Silo */
#endif
  char   *meshName;     /* The name for the current mesh */
  char   *objName;      /* The name for the next object passed to Silo */
} Viewer_Silo;
