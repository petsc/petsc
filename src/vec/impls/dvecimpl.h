/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors
  Includes declarations of all Dv* functions 

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "vecimpl.h"

typedef struct { int n; Scalar *array; } DvVector;


#endif
