/* $Id: matlab.h,v 1.8 1996/03/10 17:29:22 bsmith Exp bsmith $ */
/* 
     This is the definition of the Matlab viewer structure. Note: 
  each viewer has a different data structure.
*/

#include "petsc.h"
#include "pinclude/pviewer.h"
#include "sysio.h"

struct _Viewer {
  VIEWERHEADER
  int           port;
};

#define DEFAULTPORT    5005

/* different types of matrix which may be communicated */
#define DENSEREAL      0
#define SPARSEREAL     1
#define DENSECHARACTER 2

/* Note: DENSEREAL and DENSECHARACTER are stored exactly the same way */
/* DENSECHARACTER simply has a flag set which tells that it should be */
/* interpreted as a string not a numeric vector                       */



