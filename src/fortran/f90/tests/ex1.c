#ifdef PETSC_HAVE_FORTRAN_CAPS
#define fortran_routine_ FORTRAN_ROUTINE
#define c_routine_ C_ROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define fortran_routine_ fortran_routine
#define c_routine_ c_routine
#endif

#include <stdio.h>
#include "src/fortran/f90/zf90.h"
typedef struct {
  int a;
  array1d b;
  int c;
} abc;

extern void fortran_routine_(abc *x);

void c_routine_(abc *x)
{
  double *data = (double*)b->data;
  x->a = 2;
  data[0] = 22.0
  x->c = 222;
  fortran_routine_(x); 
}
