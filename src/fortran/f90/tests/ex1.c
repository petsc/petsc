/*$Id: ex1.c,v 1.3 2000/08/23 18:42:50 balay Exp balay $*/

#include <stdio.h>
#include "src/fortran/f90/zf90.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define fortran_routine_ FORTRAN_ROUTINE
#define c_routine_ C_ROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define fortran_routine_ fortran_routine
#define c_routine_ c_routine
#endif

typedef struct {
  int a;
  array1d b;
  int c;
} abc;

EXTERN_C_BEGIN

extern void fortran_routine_(abc *x);

void c_routine_(abc *x)
{
  double *data = (double*)x->b.addr;

  printf("From C: %d %5.2e %d\n",x->a,data[0],x->c);
  fflush(stdout);
  x->a = 2;

  data[0] = 22.0;
  x->c = 222;
  fortran_routine_(x); 
}

EXTERN_C_END
