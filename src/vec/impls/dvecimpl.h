/* $Id: dvecimpl.h,v 1.3 1995/06/07 17:28:55 bsmith Exp bsmith $ */
/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "vecimpl.h"

typedef struct { int n; Scalar *array; } Vec_Seq;


#endif
