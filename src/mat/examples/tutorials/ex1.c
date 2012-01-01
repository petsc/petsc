
static char help[] = "Reads a PETSc matrix and vector from a file and reorders it.\n\
  -f0 <input_file> : first file to load (small system)\n\
  -f1 <input_file> : second file to load (larger system)\n\n";

/*T
   Concepts: Mat^ordering a matrix - loading a binary matrix and vector;
   Concepts: Mat^loading a binary matrix and vector;
   Concepts: Vectors^loading a binary vector;
   Concepts: PetscLog^preloading executable
   Processors: 1
T*/

/* 
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers               
*/
#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A;                /* matrix */
  PetscViewer           fd;               /* viewer */
  char                  file[2][PETSC_MAX_PATH_LEN];     /* input file name */
  IS                    isrow,iscol;      /* row and column permutations */
  PetscErrorCode        ierr;
  const MatOrderingType rtype = MATORDERINGRCM;
  PetscBool             flg,PetscPreLoad = PETSC_FALSE;

  PetscInitialize(&argc,&args,(char *)0,help);


  /* 
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f0 option");
  ierr = PetscOptionsGetString(PETSC_NULL,"-f1",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) PetscPreLoad = PETSC_TRUE;

  /* -----------------------------------------------------------
                  Beginning of loop
     ----------------------------------------------------------- */
  /* 
     Loop through the reordering 2 times.  
      - The intention here is to preload and solve a small system;
        then load another (larger) system and solve it as well.
        This process preloads the instructions with the smaller
        system so that more accurate performance monitoring (via
        -log_summary) can be done with the larger one (that actually
        is the system of interest). 
  */
  PetscPreLoadBegin(PetscPreLoad,"Load");

    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
                           Load system i
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* 
       Open binary file.  Note that we use FILE_MODE_READ to indicate
       reading from this file.
    */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&fd);CHKERRQ(ierr);

    /*
       Load the matrix; then destroy the viewer.
    */
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatLoad(A,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);


    /* - - - - - - - - - - - New Stage - - - - - - - - - - - - -
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    PetscPreLoadStage("Reordering");
    ierr = MatGetOrdering(A,rtype,&isrow,&iscol);CHKERRQ(ierr);

    /* 
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = ISDestroy(&isrow);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  PetscPreLoadEnd();

  ierr = PetscFinalize();
  return 0;
}

