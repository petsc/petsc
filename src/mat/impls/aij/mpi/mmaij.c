
/*
   Support for the parallel AIJ matrix vector multiply
*/
#include "mpiaij.h"
#include "vec/vecimpl.h"
#include "../seq/aij.h"

int MPIAIJSetUpMultiply(Mat mat)
{
  Matimpiaij *aij = (Matimpiaij *) mat->data;
  Matiaij    *B = (Matiaij *) (aij->B->data);  
  int        N = aij->N,i,j,*indices;

  /* For the first stab we make an array as long as the number of columns */
  indices = (int *) MALLOC( N*sizeof(int) ); CHKPTR(indices);
  MEMSET(indices,0,N*sizeof(int));

  return 0;
}
