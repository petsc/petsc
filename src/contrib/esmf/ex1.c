/*$Id: ex1.c,v 1.5 2000/08/30 21:18:39 balay Exp $*/

#include <stdio.h>
#include "petscf90.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define fortran_routine_ FORTRAN_ROUTINE
#define c_routine_ C_ROUTINE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define fortran_routine_ fortran_routine
#define c_routine_ c_routine
#endif

typedef struct {
  int a;
  void *b;
} part1;

typedef struct {
  int c;
} part2;

EXTERN_C_BEGIN

extern void fortran_routine_(void *x);

void c_routine_(void *in)
{
  double     *data;
  part2      *y;
  part1      *x  = (part1 *)in;
  F90Array1d ptr = (F90Array1d)&(x->b);

  F90Array1dAccess(ptr,(void **)&data);
  F90Array1dGetNextRecord(ptr,(void**)&y);
  printf("From C: %d %5.2e %d\n",x->a,data[0],y->c);
  fflush(stdout);
  x->a = 2;

  data[0] = 22.0;
  y->c = 222;
  fortran_routine_(x); 
}

EXTERN_C_END
