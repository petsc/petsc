#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.6 1995/03/06 04:00:47 bsmith Exp bsmith $";
#endif

#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "mat.h"

/*@C
      MatCreateInitialMatrix - Reads from command line to determine 
           what type of matrix to create. Also uses MPI matrices if 
           number processors in MPI_COMM_WORLD greater then one.

  Input Parameters:
.   m,n - matrix dimensions
 
  Output Parameter:
.   V - location to stash resulting matrix.
@*/
int MatCreateInitialMatrix(int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(MPI_COMM_WORLD,&numtid);
  if (OptionsHasName(0,0,"-dense_mat")) {
    return MatCreateSequentialDense(m,n,V);
  }
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    return MatCreateMPIAIJ(MPI_COMM_WORLD,-1,-1,m,n,5,0,0,0,V);
  }
  return MatCreateSequentialAIJ(m,n,10,0,V);
}
 
