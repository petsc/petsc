#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.9 1995/04/07 01:42:09 curfman Exp curfman $";
#endif

#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "mat.h"

/*@C
      MatCreateInitialMatrix - Reads from command line to determine 
           what type of matrix to create.  Also uses MPI matrices if 
           number processors in MPI_COMM_WORLD is greater then one.

  Input Parameters:
.   m,n - global matrix dimensions
 
  Output Parameter:
.   V - location to stash resulting matrix
@*/
int MatCreateInitialMatrix(int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(MPI_COMM_WORLD,&numtid);
  if (OptionsHasName(0,0,"-dense_mat")) {
    return MatCreateSequentialDense(m,n,V);
  }
  if (OptionsHasName(0,0,"-row_mat")) {
    return MatCreateSequentialRow(m,n,10,0,V);
  }
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    if (OptionsHasName(0,0,"-row_mat")) {
      return MatCreateMPIRow(MPI_COMM_WORLD,-1,-1,m,n,5,0,0,0,V);
    }
    if (OptionsHasName(0,0,"-rowbs_mat")) {
      return MatCreateMPIRowbs(MPI_COMM_WORLD,-1,m,5,0,0,V);
    }
    return MatCreateMPIAIJ(MPI_COMM_WORLD,-1,-1,m,n,5,0,0,0,V);
  }
  return MatCreateSequentialAIJ(m,n,10,0,V);
}
 
