/* $Id: psimpl.h,v 1.2 1999/11/24 21:52:54 bsmith Exp balay $ */

/*
      Defines the internal data structures for the Postscript
   implementation of the graphics functionality in PETSc.
*/

#include "src/sys/src/draw/drawimpl.h"
#include "petscsys.h"

#if !defined(_PSIMPL_H)
#define _PSIMPL_H

typedef struct {
    Viewer   ps_file;
    double   xl,xr,yl,yr;
    int      ixl,ixr,iyl,iyr;
    int      currentcolor;
} Draw_PS;

#endif
