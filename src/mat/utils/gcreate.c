
#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.41 1995/09/12 03:26:11 bsmith Exp curfman $";
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
$  -mat_seqaij   : AIJ type, uses MatCreateSeqAIJ
$  -mat_mpiaij   : AIJ type, uses MatCreateMPIAIJ
$  -mat_aij      : AIJ type, (Seq or MPI depending on comm) 
$  -mat_dense    : dense type, uses MatCreateSeqDense()
$  -mat_row      : row type, uses MatCreateSeqRow()
$  -mat_mpirow   : MatCreateMPIRow()
$  -mat_mpirowbs : rowbs type.
$                  uses MatCreateMPIRowbs()
$  -mat_bdiag    : block diagonal type, uses 
$                   MatCreateSeqBDiag()
$  -mat_mpibdiag : MatCreateMPIBDiag()

   Notes:
   The default matrix type is AIJ, using MatCreateSeqAIJ() and
   MatCreateMPIAIJ(). 

.keywords: matrix, create, initial

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqRow(), MatCreateMPIRow(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIRowbs(), MatConvert()
 @*/
int MatCreate(MPI_Comm comm,int m,int n,Mat *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (OptionsHasName(0,"-help")) {
    MPIU_printf(comm,"MatCreate() options: -mat_dense, -mat_aij, -mat_mpiaij, -mat_row\n");
    MPIU_printf(comm,"         -mat_mpirow, -mat_mpirowbs, -mat_bdiag, -mat_mpibdiag\n"); 
  }
  if (OptionsHasName(0,"-mat_dense")) {
    return MatCreateSeqDense(comm,m,n,V);
  }
  if (OptionsHasName(0,"-mat_bdiag") || OptionsHasName(0,"-mat_mpibdiag")) {
    int nb = 1, ndiag = 0, ndiag2 = 0, *d = 0, ierr;
    if (OptionsHasName(0,"-help")) {
      MPIU_printf(comm,"Options with -mat_bdiag: -mat_bdiag_bsize block_size\n");
      MPIU_printf(comm,"  -mat_bdiag_ndiag number_diags \n"); 
      MPIU_printf(comm,"  -mat_bdiag_dvals d1,d2,d3... (diagonal numbers)\n"); 
      MPIU_printf(comm,"   (for example) -mat_bdiag_dvals -5,-1,0,1,5\n"); 
    }
    OptionsGetInt(0,"-mat_bdiag_bsize",&nb);
    OptionsGetInt(0,"-mat_bdiag_ndiag",&ndiag);
    if (ndiag) {
      d = (int *)PETSCMALLOC( ndiag * sizeof(int) ); CHKPTRQ(d);
      ndiag2 = ndiag;
      OptionsGetIntArray(0,"-mat_bdiag_dvals",d,&ndiag2);
      if (ndiag2 != ndiag)
        SETERRQ(1,"MatCreate: Incompatible number of diags and diagonal vals");
    } else if (OptionsHasName(0,"-mat_bdiag_dvals"))
      SETERRQ(1,"MatCreate: Must specify number of diagonals with -mat_bdiag_ndiag");
    if (OptionsHasName(0,"-mpi_mpibdiag") || numtid>1) {
      ierr = MatCreateMPIBDiag(comm,PETSC_DECIDE,m,n,ndiag,nb,d,0,V); CHKERRQ(ierr);
    } else {
      ierr = MatCreateSeqBDiag(comm,m,n,ndiag,nb,d,0,V); CHKERRQ(ierr);
    }
    if (d) PETSCFREE(d);
    return ierr;
  }
  if (OptionsHasName(0,"-mat_mpirowbs")) {
    return MatCreateMPIRowbs(comm,PETSC_DECIDE,m,5,0,0,V);
  }
  if (OptionsHasName(0,"-mat_mpirow") || (OptionsHasName(0,"-mat_row") && numtid >1)) {
    return MatCreateMPIRow(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  if (OptionsHasName(0,"-mat_row")) {
    return MatCreateSeqRow(comm,m,n,10,0,V);
  }
  if (OptionsHasName(0,"-mat_mpiaij")  || (numtid >1)) { /* Default parallel format */
    return MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,5,0,0,0,V);
  }
  return MatCreateSeqAIJ(comm,m,n,10,0,V); /* default uniprocessor format */
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
  matname[0] = "MATSEQDENSE";
  matname[1] = "MATSEQAIJ";
  matname[2] = "MATMPIAIJ";
  matname[3] = "MATSHELL";
  matname[4] = "MATSEQROW";
  matname[5] = "MATMPIROW";
  matname[6] = "MATMPIROWBS";
  matname[7] = "MATSEQBDIAG";
  matname[8] = "MATMPIBDIAG";
  if (itype < 0 || itype > 8) *name = "unknown matrix type";
  else                        *name = matname[itype];
  return 0;
}
 
