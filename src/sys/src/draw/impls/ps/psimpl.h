/* $Id: psimpl.h,v 1.5 2001/04/10 19:34:15 bsmith Exp $ */

/*
      Defines the internal data structures for the Postscript
   implementation of the graphics functionality in PETSc.
*/

#include "src/sys/src/draw/drawimpl.h"
#include "petscsys.h"

#if !defined(_PSIMPL_H)
#define _PSIMPL_H

typedef struct {
    PetscViewer ps_file;
    double      xl,xr,yl,yr;
    int         ixl,ixr,iyl,iyr;
    int         currentcolor;
} PetscDraw_PS;

#endif
