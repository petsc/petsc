/* $Id: matlab.h,v 1.14 1998/12/03 04:04:50 bsmith Exp bsmith $ */
/* 
     This is the definition of the Matlab viewer structure. Note: 
  each viewer has a different data structure.
*/

#include "src/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include "sys.h" 

typedef struct {
  int           port;
} Viewer_Socket;

#define DEFAULTPORT    5005

/* different types of matrix which may be communicated */
#define DENSEREAL      0
#define SPARSEREAL     1
#define DENSECHARACTER 2
#define DENSEINT       3

/* Note: DENSEREAL and DENSECHARACTER are stored exactly the same way */
/* DENSECHARACTER simply has a flag set which tells that it should be */
/* interpreted as a string not a numeric vector                       */



