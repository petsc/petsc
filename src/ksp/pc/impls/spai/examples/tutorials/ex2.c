
static char help[] = 
"Reads a PETSc matrix and vector from a file and saves in an ASCII file that\n\
  can be read by the SPAI test program.  Input parameters include\n\
  -f0 <input_file> : file to load\n\n";

/*T
   Routines: MatLoad(); VecLoad();
   Routines: ViewerBinaryOpen();
   Processors: 1
T*/

/* 
  Include "mat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       is.h  - index sets
     viewer.h - viewers
*/
#include "mat.h"
#include "src/contrib/spai/include/spai.h"

int main(int argc,char **args)
{
  Mat        A;                /* matrix */
  Vec        b;                /* RHS */
  Viewer     viewer;               /* viewer */
  char       file[128];        /* input file name */
  int        ierr, flg;
  PetscTruth set;
  MatType    mtype;
  FILE       *fd;

  PetscInitialize(&argc,&args,(char *)0,help);

#if defined(USE_PETSC_COMPLEX)
  SETERRQ(1,0,"This example does not work with complex numbers");
#else

  /* 
     Determine files from which we read the linear system
     (matrix and right-hand-side vector).
  */
  ierr = OptionsGetString(PETSC_NULL,"-f0",file,127,&flg); CHKERRA(ierr);
  if (!flg) SETERRQ(1,0,"Must indicate binary file with the -f0 option");


  /* 
       Open binary file.  Note that we use BINARY_RDONLY to indicate
       reading from this file.
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&viewer);CHKERRA(ierr);

  /* 
       Determine matrix format to be used (specified at runtime).
       See the manpage for MatLoad() for available formats.
  */
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);

  /*
       Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(viewer,mtype,&A); CHKERRA(ierr);
  ierr = VecLoad(viewer,PETSC_NULL,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);

  fd = fopen("example_matrix","w");
  ierr = MatDumpSPAI(A,fd); CHKERRQ(ierr);
  fclose(fd);
  fd = fopen("example_rhs","w");
  ierr = VecDumpSPAI(b,fd); CHKERRQ(ierr);
  fclose(fd);

  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);

  PetscFinalize();
#endif
  return 0;
}
