#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.14 1995/04/27 01:04:14 curfman Exp curfman $";
#endif

#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "mat.h"

/*@C
   MatCreateInitialMatrix - Creates a matrix, where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.

   Input Parameters:
.  m - number of global rows
.  n - number of global columns
.  comm - MPI communicator
 
   Output Parameter:
.  V - location to stash resulting matrix

   Options Database Keywords:
$  -dense_mat : dense type, uses MatCreateSequentialDense()
$  -row_mat   : row type, uses MatCreateSequentialRow()
$               and MatCreateMPIRow()
$  -rowbs_mat : rowbs type (for parallel symmetric matrices),
$               uses MatCreateMPIRowbs()
$  -bdiag_mat : block diagonal type, uses 
$               MatCreateSequentialBDiag() and
$               MatCreateMPIBDiag()
$
$  -mpi_objects : uses MPI matrix, even for one processor

   Notes:
   The default matrix type is AIJ, using MatCreateSequentialAIJ() and
   MatCreateMPIAIJ().

.keywords: matrix, create, initial

.seealso: MatCreateSequentialAIJ((), MatCreateMPIAIJ(), 
          MatCreateSequentialRow(), MatCreateMPIRow(), 
          MatCreateSequentialDense(), MatCreateSequentialBDiag(),
          MatCreateMPIRowbs()
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
 
