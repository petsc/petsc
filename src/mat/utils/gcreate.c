#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcreate.c,v 1.107 1998/04/15 19:38:11 curfman Exp curfman $";
#endif

#include "sys.h"
#include "mat.h"       /*I "mat.h"  I*/

#undef __FUNC__  
#define __FUNC__ "MatCreate"
/*@C
   MatCreate - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() and MatCreateMPIAIJ(). 

   Input Parameters:
.  m - number of global rows
.  n - number of global columns
.  comm - MPI communicator
 
   Output Parameter:
.  A - the matrix

   Collective on MPI_Comm

   Basic Options Database Keys:
   These options use MatCreateSeqXXX or MatCreateMPIXXX,
   depending on the communicator, comm.
$    -mat_aij      : AIJ type
$    -mat_baij     : block AIJ type
$    -mat_dense    : dense type
$    -mat_bdiag    : block diagonal type

   More Options Database Keys:
$    -mat_seqaij   : AIJ type, uses MatCreateSeqAIJ()
$    -mat_mpiaij   : AIJ type, uses MatCreateMPIAIJ()
$    -mat_seqbdiag : block diagonal type, uses 
$                    MatCreateSeqBDiag()
$    -mat_mpibdiag : block diagonal type, uses 
$                    MatCreateMPIBDiag()
$    -mat_mpirowbs : rowbs type, uses MatCreateMPIRowbs()
$    -mat_seqdense : dense type, uses MatCreateSeqDense()
$    -mat_mpidense : dense type, uses MatCreateMPIDense()
$    -mat_seqbaij  : block AIJ type, uses MatCreateSeqBAIJ()
$    -mat_mpibaij  : block AIJ type, uses MatCreateMPIBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Notes:
   This routine calls MatGetTypeFromOptions() to determine the matrix type.

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatCreateSeqBAIJ,
          MatConvert(), MatGetTypeFromOptions()
 @*/
int MatCreate(MPI_Comm comm,int m,int n,Mat *A)
{
  MatType    type;
  PetscTruth set;
  int        ierr, bs=1, flg;

  PetscFunctionBegin;
  ierr = MatGetTypeFromOptions(comm,0,&type,&set); CHKERRQ(ierr);
  switch (type) {
  case MATSEQDENSE:
    ierr = MatCreateSeqDense(comm,m,n,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATMPIBDIAG:
    ierr = MatCreateMPIBDiag(comm,PETSC_DECIDE,m,n,PETSC_DEFAULT,PETSC_DEFAULT,
           PETSC_NULL,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATSEQBDIAG:
    ierr = MatCreateSeqBDiag(comm,m,n,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_NULL,
           PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATMPIROWBS:
    ierr = MatCreateMPIRowbs(comm,PETSC_DECIDE,m,PETSC_DEFAULT,PETSC_NULL,
           PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATMPIDENSE:
    ierr = MatCreateMPIDense(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATMPIAIJ:
    ierr = MatCreateMPIAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_DEFAULT,
           PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATSEQBAIJ:
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg); CHKERRQ(ierr);
    ierr = MatCreateSeqBAIJ(comm,bs,m,n,PETSC_DEFAULT,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  case MATMPIBAIJ:
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg); CHKERRQ(ierr);
    ierr = MatCreateMPIBAIJ(comm,bs,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  default:
    ierr = MatCreateSeqAIJ(comm,m,n,PETSC_DEFAULT,PETSC_NULL,A); CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

 



