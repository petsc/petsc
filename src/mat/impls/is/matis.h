
#if !defined(__is_h)
#define __is_h

#include <petsc-private/matimpl.h>

typedef struct {
  Mat                    A;             /* the local Neumann matrix */
  VecScatter             ctx;           /* update ghost points for matrix vector product */
  Vec                    x,y;           /* work space for ghost values for matrix vector product */
  ISLocalToGlobalMapping mapping;
  int                    rstart,rend;   /* local row ownership */
  PetscBool              pure_neumann;
} Mat_IS;

#endif




