/* $Id: dvecimpl.h,v 1.5 1996/08/04 23:11:06 bsmith Exp bsmith $ */
/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "src/vec/vecimpl.h"

typedef struct { 
  VECHEADER
} Vec_Seq;


#endif
