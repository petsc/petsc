
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "mat.h"
#include "comm.h"

/*@C
      MatCreateInitialMatrix - Reads from command line to determine 
           what type of matrix to create.

  Input Parameters:
.   m,n - matrix dimensions
 
  Output Parameter:
.   V - location to stash resulting matrix.
@*/
int MatCreateInitialMatrix(int m,int n,Mat *V)
{
  if (OptionsHasName(0,0,"-dense")) {
    fprintf(stdout,"Using BLAS+LAPACK sequential dense matrices\n");
    return MatCreateSequentialDense(m,n,V);
  }
  fprintf(stdout,"Using standard sequential AIJ matrices\n");
  return MatCreateSequentialAIJ(m,n,10,0,V);
}
 
