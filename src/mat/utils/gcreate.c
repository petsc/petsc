#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.12 1995/04/15 17:23:18 curfman Exp curfman $";
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
.   comm - MPI communicator
 
  Output Parameter:
.   V - location to stash resulting matrix
@*/
int MatCreateInitialMatrix(MPI_Comm comm,int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (OptionsHasName(0,0,"-dense_mat")) {
    return MatCreateSequentialDense(comm,m,n,V);
  }
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    if (OptionsHasName(0,0,"-row_mat")) {
      return MatCreateMPIRow(comm,PETSC_DECIDE,PETSC_DECIDE, m,n,5,0,0,0,V);
    }
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
    if (OptionsHasName(0,0,"-rowbs_mat")) {
      return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,5,0,0,V);
    }
#endif
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE, m,n,5,0,0,0,V);
  }
  if (OptionsHasName(0,0,"-row_mat")) {
    return MatCreateSequentialRow(comm,m,n,10,0,V);
  }
  return MatCreateSequentialAIJ(comm,m,n,10,0,V);
}
 
