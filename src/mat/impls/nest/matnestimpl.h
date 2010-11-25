
#ifndef MatNest_impl_h
#define MatNest_impl_h

#include <private/matimpl.h>

typedef struct {
  PetscInt           nr,nc;        /* nr x nc blocks */
  Mat                **m;
  PetscBool          setup_called;
  IS                 *is_row,*is_col;
  PetscInt           *row_len,*col_len;
} Mat_Nest;

#endif

