
/*  
    Private Krylov Context Structure (KSP) for the Bad limited memory Broyden method applied to a linear equation

*/

#if !defined(__BROYDENIMPL_H)
#define __BROYDENIMPL_H

/*
        Defines the basic KSP object
*/
#include "private/kspimpl.h"

typedef struct {
  Vec      *v,*w;
  PetscInt msize;   /* maximum size of space */
  PetscInt csize;   /* current size of space */
} KSP_Broyden;

#endif
