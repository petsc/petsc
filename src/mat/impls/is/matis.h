/*$Id: is.h,v 1.3 2000/08/24 22:42:00 bsmith Exp $*/

#if !defined(__is_h)
#define __is_h

#include "src/mat/matimpl.h"

typedef struct {
  Mat                    A;             /* the local Neumann matrix */
  VecScatter             ctx;           /* update ghost points for matrix vector product */
  Vec                    x,y;           /* work space for ghost values for matrix vector product */
  ISLocalToGlobalMapping mapping;
  int                    rstart,rend;   /* local row ownership */
  PetscTruth             pure_neumann;
} Mat_IS;

#endif




