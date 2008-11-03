/* 
     This is the definition of the socket viewer structure. This starts the same as the PetscViewer_Binary() so the
   binary read/writes can be called directly on it.
*/

#include "../src/sys/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include "petscsys.h" 

typedef struct {
  int           port;
#if defined(PETSC_HAVE_MPIIO)
  PetscTruth    MPIIO;
#endif
} PetscViewer_Socket;

#define PETSCSOCKETDEFAULTPORT    5005

/* different types of matrix which may be communicated */
#define DENSEREAL      0
#define SPARSEREAL     1
#define DENSECHARACTER 2
#define DENSEINT       3

/* Note: DENSEREAL and DENSECHARACTER are stored exactly the same way */
/* DENSECHARACTER simply has a flag set which tells that it should be */
/* interpreted as a string not a numeric vector                       */



