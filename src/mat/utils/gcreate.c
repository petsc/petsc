#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.17 1995/05/02 16:35:56 curfman Exp bsmith $";
#endif

#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "mat.h"

/*@C
   MatCreate - Creates a matrix, where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.

   Input Parameters:
.  m - number of global rows
.  n - number of global columns
.  comm - MPI communicator
 
   Output Parameter:
.  V - location to stash resulting matrix

   Options Database Keywords:
$  -mat_dense : dense type, uses MatCreateSequentialDense()
$  -mat_row   : row type, uses MatCreateSequentialRow()
$               and MatCreateMPIRow()
$  -mat_rowbs : rowbs type (for parallel symmetric matrices),
$               uses MatCreateMPIRowbs()
$  -mat_bdiag : block diagonal type, uses 
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
int MatCreate(MPI_Comm comm,int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (OptionsHasName(0,0,"-mat_dense")) {
    return MatCreateSequentialDense(comm,m,n,V);
  }
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    if (OptionsHasName(0,0,"-mat_row")) {
      return MatCreateMPIRow(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
    }
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
    if (OptionsHasName(0,0,"-mat_rowbs")) {
      return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,5,0,0,V);
    }
#endif
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  if (OptionsHasName(0,0,"-mat_row")) {
    return MatCreateSequentialRow(comm,m,n,10,0,V);
  }
  if (OptionsHasName(0,0,"-mat_bdiag")) {
    int nb = 1, ndiag = 0, ndiag2, i, *d, ierr;
    OptionsGetInt(0,0,"-mat_bdiag_bsize",&nb);
    OptionsGetInt(0,0,"-mat_bdiag_ndiag",&ndiag);
    if (ndiag) {
      d = (int *)MALLOC( ndiag * sizeof(int) ); CHKPTR(d);
      ndiag2 = ndiag;
      OptionsGetIntArray(0,0,"-mat_bdiag_dvals",d,&ndiag2);
      if (ndiag2 != ndiag) 
        SETERR(1,"Incompatible number of diagonals and diagonal values.");
    } else SETERR(1,"Must set diagonals before creating matrix.");
    ierr = MatCreateSequentialBDiag(comm,m,n,ndiag,nb,d,0,V); 
    if (d) FREE(d);
    return ierr;
  }
  return MatCreateSequentialAIJ(comm,m,n,10,0,V);
}
 
