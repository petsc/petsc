/*$Id: gcreate.c,v 1.121 2000/07/10 03:39:59 bsmith Exp balay $*/

#include "petscsys.h"
#include "petscmat.h"       /*I "petscmat.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreate"
/*@C
   MatCreate - Creates a matrix where the type is determined
   from the options database. Generates a parallel MPI matrix if the
   communicator has more than one processor.  The default matrix type is
   AIJ, using the routines MatCreateSeqAIJ() and MatCreateMPIAIJ(). 

   Collective on MPI_Comm

   Input Parameters:
+  m - number of local rows (or PETSC_DECIDE)
.  n - number of local columns (or PETSC_DECIDE)
.  M - number of global rows (or PETSC_DETERMINE)
.  N - number of global columns (or PETSC_DETERMINE)
-  comm - MPI communicator
 
   Output Parameter:
.  A - the matrix

   Basic Options Database Keys:
   These options use MatCreateSeqXXX or MatCreateMPIXXX,
   depending on the communicator, comm.
+    -mat_aij      - AIJ type
.    -mat_baij     - block AIJ type
.    -mat_dense    - dense type
-    -mat_bdiag    - block diagonal type

   More Options Database Keys:
+    -mat_seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_mpiaij   - AIJ type, uses MatCreateMPIAIJ()
.    -mat_seqbdiag - block diagonal type, uses MatCreateSeqBDiag()
.    -mat_mpibdiag - block diagonal type, uses MatCreateMPIBDiag()
.    -mat_mpirowbs - rowbs type, uses MatCreateMPIRowbs()
.    -mat_seqdense - dense type, uses MatCreateSeqDense()
.    -mat_mpidense - dense type, uses MatCreateMPIDense()
.    -mat_seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_mpibaij  - block AIJ type, uses MatCreateMPIBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Notes:
   If PETSC_DECIDE is not used for the arguments 'm' and 'n', then the
   user must ensure that they are chosen to be compatible with the
   vectors. To do this, one first considers the matrix-vector product 
   'y = A x'. The 'm' that is used in the above routine must match the 
   local size used in the vector creation routine VecCreateMPI() for 'y'.
   Likewise, the 'n' used must match that used as the local size in
   VecCreateMPI() for 'x'.

   This routine calls MatGetTypeFromOptions() to determine the matrix type.

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateMPIAIJ(), 
          MatCreateSeqBDiag(),MatCreateMPIBDiag(),
          MatCreateSeqDense(), MatCreateMPIDense(), 
          MatCreateMPIRowbs(), MatCreateSeqBAIJ(), MatCreateMPIBAIJ()
          MatConvert(), MatGetTypeFromOptions()
@*/
int MatCreate(MPI_Comm comm,int m,int n,int M,int N,Mat *A)
{
  MatType    type;
  PetscTruth set;
  int        ierr,bs = 1;

  PetscFunctionBegin;
  ierr = MatGetTypeFromOptions(comm,0,&type,&set);CHKERRQ(ierr);
  switch (type) {
  case MATSEQDENSE:
    m    = PetscMax(m,M);
    n    = PetscMax(n,N);
    ierr = MatCreateSeqDense(comm,m,n,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATMPIBDIAG:
    ierr = MatCreateMPIBDiag(comm,m,M,N,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_NULL,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATSEQBDIAG:
    m    = PetscMax(m,M);
    n    = PetscMax(n,N);
    ierr = MatCreateSeqBDiag(comm,m,n,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_NULL,PETSC_NULL,A);CHKERRQ(ierr);
    break;
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  case MATMPIROWBS:
    ierr = MatCreateMPIRowbs(comm,m,M,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
#endif
  case MATMPIDENSE:
    ierr = MatCreateMPIDense(comm,m,n,M,N,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATMPIAIJ:
    ierr = MatCreateMPIAIJ(comm,m,n,M,N,PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATSEQBAIJ:
    m    = PetscMax(m,M);
    n    = PetscMax(n,N);
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatCreateSeqBAIJ(comm,bs,m,n,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATMPIBAIJ:
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatCreateMPIBAIJ(comm,bs,m,n,M,N,PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATSEQSBAIJ:
    m    = PetscMax(m,M);
    n    = PetscMax(n,N);
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatCreateSeqSBAIJ(comm,bs,m,n,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  case MATMPISBAIJ:
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatCreateMPISBAIJ(comm,bs,m,n,M,N,PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  default:
    m    = PetscMax(m,M);
    n    = PetscMax(n,N);
    ierr = MatCreateSeqAIJ(comm,m,n,PETSC_DEFAULT,PETSC_NULL,A);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

 



