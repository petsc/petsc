
#include "sysio.h"
#include "is.h"
#include "vec.h"

int VecCreateInitialVector(n,argc,argv,V)
int  argc,n;
char **argv;
Vec  *V;
{
#if !defined(PETSC_COMPLEX)
  if (SYArgHasName(&argc,argv,0,"-blas")) {
    PISfprintf(0,stdout,"Using BLAS sequential vectors\n");
    return VecCreateSequentialBLAS(n,V);
  }
  PISfprintf(0,stdout,"Using standard sequential vectors\n");
  return VecCreateSequential(n,V);
#else
  PISfprintf(0,stdout,"Using Complex sequential vectors\n");
  return VecCreateComplexSequential(n,V);
#endif 
}
 
