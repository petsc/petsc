
#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.34 1995/08/03 01:14:17 curfman Exp bsmith $";
#endif

#include "sys.h"
#include "mat.h"       /*I "mat.h"  I*/

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
$  -mat_aij   : AIJ type, uses MatCreateSequentialAIJ()
$               and MatCreateMPIAIJ()
$  -mat_dense : dense type, uses MatCreateSequentialDense()
$  -mat_row   : row type, uses MatCreateSequentialRow()
$               and MatCreateMPIRow()
$  -mat_rowbs : rowbs type (for parallel symmetric matrices only),
$               uses MatCreateMPIRowbs()
$  -mat_bdiag : block diagonal type, uses 
$               MatCreateSequentialBDiag() and
$               MatCreateMPIBDiag()
$
$  -mpi_objects : uses MPI matrix (parallel format), even for one processor

   Notes:
   The default matrix type is AIJ, using MatCreateSequentialAIJ() and
   MatCreateMPIAIJ(). 

.keywords: matrix, create, initial

.seealso: MatCreateSequentialAIJ((), MatCreateMPIAIJ(), 
          MatCreateSequentialRow(), MatCreateMPIRow(), 
          MatCreateSequentialBDiag(),MatCreateMPIBDiag(),
          MatCreateSequentialDense(), MatCreateMPIRowbs(), MatConvert()
 @*/
int MatCreate(MPI_Comm comm,int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (OptionsHasName(0,"-help")) {
    MPIU_printf(comm,"MatCreate() options: -mat_dense, -mat_row -mat_rowbs\n");
    MPIU_printf(comm,"                     -mat_bdiag, -mpi_objects\n"); 
  }
  if (OptionsHasName(0,"-mat_dense")) {
    return MatCreateSequentialDense(comm,m,n,V);
  }
  if (OptionsHasName(0,"-mat_bdiag")) {
    int nb = 1, ndiag = 0, ndiag2,  *d, ierr;
    OptionsGetInt(0,"-mat_bdiag_bsize",&nb);
    OptionsGetInt(0,"-mat_bdiag_ndiag",&ndiag);
    if (!ndiag) SETERRQ(1,"MatCreate:Must set diagonals before creating mat");
    d = (int *)PETSCMALLOC( ndiag * sizeof(int) ); CHKPTRQ(d);
    ndiag2 = ndiag;
    OptionsGetIntArray(0,"-mat_bdiag_dvals",d,&ndiag2);
    if (ndiag2 != ndiag) { 
      SETERRQ(1,"MatCreate:Incompatible number of diags and diagonal vals");
    }
    if (numtid > 1 || OptionsHasName(0,"-mpi_objects"))
      ierr = MatCreateMPIBDiag(comm,PETSC_DECIDE,m,n,ndiag,nb,d,0,V); 
    else
      ierr = MatCreateSequentialBDiag(comm,m,n,ndiag,nb,d,0,V); 
    CHKERRQ(ierr);
    if (d) PETSCFREE(d);
    MPIU_printf(comm,"Options with -mat_bdiag: -mat_bdiag_bsize block_size\n");
    MPIU_printf(comm,"  -mat_bdiag_ndiag number_diags \n"); 
    MPIU_printf(comm,"  -mat_bdiag_dvals d1,d2,d3...  \n"); 
    MPIU_printf(comm,"(for example)  -mat_bdiag_dvals -5,-1,0,1,5  \n"); 
    return ierr;
  }
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
  if (OptionsHasName(0,"-mat_rowbs")) {
    return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,5,0,0,V);
  }
#endif
  if (numtid > 1 || OptionsHasName(0,"-mpi_objects")) {
    if (OptionsHasName(0,"-mat_row")) {
      return MatCreateMPIRow(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
    }
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  if (OptionsHasName(0,"-mat_row")) {
    return MatCreateSequentialRow(comm,m,n,10,0,V);
  }
  return MatCreateSequentialAIJ(comm,m,n,10,0,V);
}

#include "matimpl.h"
/*@C
   MatGetName - Gets the matrix type name (as a string) from the matrix.

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of matrix type

.keywords: matrix, get, name

.seealso:  MatGetType()
@*/
int MatGetName(Mat mat,char **name)
{
  /* Note:  Be sure that this list corresponds to the enum in mat.h */
  int  itype = (int)mat->type;
  char *matname[9];
  matname[0] = "MATDENSE";
  matname[1] = "MATAIJ";
  matname[2] = "MATMPIAIJ";
  matname[3] = "MATSHELL";
  matname[4] = "MATROW";
  matname[5] = "MATMPIROW";
  matname[6] = "MATMPIROW_BS";
  matname[7] = "MATBDIAG";
  matname[8] = "MATMPIBDIAG";
  if (itype < 0 || itype > 8) *name = "unknown matrix type";
  else                        *name = matname[itype];
  return 0;
}
 
