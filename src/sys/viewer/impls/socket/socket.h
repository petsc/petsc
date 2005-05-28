/* 
     This is the definition of the Matlab viewer structure. Note: 
  each viewer has a different data structure.
*/

#include "src/sys/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include "petscsys.h" 

typedef struct {
  int           port;
} PetscViewer_Socket;

#define DEFAULTPORT    5005

/* different types of matrix which may be communicated */
#define DENSEREAL      0
#define SPARSEREAL     1
#define DENSECHARACTER 2
#define DENSEINT       3

/* Note: DENSEREAL and DENSECHARACTER are stored exactly the same way */
/* DENSECHARACTER simply has a flag set which tells that it should be */
/* interpreted as a string not a numeric vector                       */



