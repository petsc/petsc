
static char help[] = 
"Reads a matrix in SPAI format (ascii), a vector in SPAI format (ascii),\n\
 and write them in PETSc format\n";

/* 
  Include "mat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       is.h  - index sets
     viewer.h - viewers
*/
#include "mat.h"

/*
#include "src/contrib/spai/include/matrix.h"
#include "src/contrib/spai/include/vector.h"
#include "src/contrib/spai/include/read_mm_matrix.h"
*/

#include "matrix.h"
#include "vector.h"
#include "read_mm_matrix.h"

int main(int argc,char **args)
{
  char       file1[128];
  char       file2[128];
  char       file3[128];
  int        ierr, flg;
  matrix *A;
  vector *b;
  Viewer fd;
  Mat        A_PETSC;          /* matrix */
  Vec        b_PETSC;          /* RHS */
  

  PetscInitialize(&argc,&args,(char *)0,help);
  /* 
     Determine files from which we read the matrix 
  */
  ierr = OptionsGetString(PETSC_NULL,"-f0",file1,127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate ascii SPAI matrix file with the -f0 option");
  ierr = OptionsGetString(PETSC_NULL,"-f1",file2,127,&flg); CHKERRA(ierr);
  ierr = OptionsGetString(PETSC_NULL,"-f2",file3,127,&flg); CHKERRA(ierr);

  /* Read the "traspose" because PETSc uses compressed row storage.*/
  A = read_mm_matrix(file1,
		     1,
		     1,
		     1,
		     1, /* transpose */
		     0,
		     0,
		     PETSC_COMM_WORLD);

  /* read ascii SPAI rhs file */
  b = read_rhs_for_matrix(file2, A);

  ConvertMatrixToMat(A,&A_PETSC);
  ConvertVectorToVec(b,&b_PETSC);

  ierr = ViewerFileOpenBinary
    (PETSC_COMM_WORLD,file3,BINARY_CREATE,&fd); CHKERRA(ierr);
  ierr = MatView(A_PETSC,fd); CHKERRA(ierr);
  ierr = VecView(b_PETSC,fd); CHKERRA(ierr);
  
  PetscFinalize();

}

