/* 
     This is the include file, it contains definitions 
  needed by different components of the package.

     Note: this is not the only viewer that can be defined!
*/

#include "petsc.h"
#include "ptscimpl.h"

struct _Viewer {
  PETSCHEADER
  int         port;
};


/* different types of matrix which may be communicated */
#define DENSEREAL      0
#define SPARSEREAL     1
#define DENSECHARACTER 2

/* Note: DENSEREAL and DENSECHARACTER are stored exactly the same way */
/* DENSECHARACTER simply has a flag set which tells that it should be */
/* interpreted as a string not a numeric vector                       */

