
static char help[] = "Reads a PETSc matrix and vector from a file and reorders it.\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n";

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat             A;                      /* matrix */
  PetscViewer     fd;                     /* viewer */
  char            file[PETSC_MAX_PATH_LEN];           /* input file name */
  IS              isrow,iscol;            /* row and column permutations */
  MatOrderingType rtype = MATORDERINGRCM;
  PetscBool       flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* -----------------------------------------------------------
                  Beginning of loop
     ----------------------------------------------------------- */
  /*
     Loop through the reordering 2 times.
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_view) can be done with the larger one (that actually
        is the system of interest).
  */

  /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                         Load system i
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /*
     Load the matrix; then destroy the viewer.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetOrdering(A,rtype,&isrow,&iscol));
  PetscCall(ISView(isrow,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

TEST*/
