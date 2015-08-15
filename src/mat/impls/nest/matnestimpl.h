
#if !defined(MatNest_impl_h)
#define MatNest_impl_h

#include <petsc/private/matimpl.h>

struct MatNestISPair {
  IS *row,*col;
};

typedef struct {
  PetscInt             nr,nc;      /* nr x nc blocks */
  Mat                  **m;
  struct MatNestISPair isglobal;
  struct MatNestISPair islocal;
  Vec                  *left,*right;
  PetscInt             *row_len,*col_len;
  PetscBool            splitassembly;
} Mat_Nest;

#endif
