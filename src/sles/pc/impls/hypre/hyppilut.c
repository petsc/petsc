/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/
/*
    Creates hypre ijvector from PETSc vector
*/

#include "src/sles/pc/pcimpl.h"          /*I "petscvec.h" I*/
#include "HYPRE.h"
#include "IJ_mv.h"
