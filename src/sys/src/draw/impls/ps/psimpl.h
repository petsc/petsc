/* $Id: psimpl.h,v 1.3 2000/05/05 22:13:39 balay Exp bsmith $ */

/*
      Defines the internal data structures for the Postscript
   implementation of the graphics functionality in PETSc.
*/

#include "src/sys/src/draw/drawimpl.h"
#include "petscsys.h"

#if !defined(_PSIMPL_H)
#define _PSIMPL_H

typedef struct {
    PetscViewer   ps_file;
    double   xl,xr,yl,yr;
    int      ixl,ixr,iyl,iyr;
    int      currentcolor;
} PetscDraw_PS;

#endif
