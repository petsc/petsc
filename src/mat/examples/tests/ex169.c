
static char help[] = "Test memory leak when duplicating a redundant matrix.\n\n";
                      

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,Ar,C;
  PetscViewer    fd;                        /* viewer */
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode ierr;
  PetscInt       ns=2,np;
  PetscSubcomm   subc;
  PetscBool      flg;
 
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  ierr = PetscOptionsGetString(NULL,NULL,"-f0",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f0 option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading matrix with %d processors\n",np);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  /* 
     Determines amount of subcomunicators 
  */
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsub",&ns,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Splitting in %d subcommunicators\n",ns);CHKERRQ(ierr);
  ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)A),&subc);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(subc,ns);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(subc,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
  ierr = PetscSubcommSetFromOptions(subc);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(A,0,PetscSubcommChild(subc),MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Copying matrix\n",ns);CHKERRQ(ierr);
  ierr = MatDuplicate(Ar,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&subc);CHKERRQ(ierr);
  
  /*
     Free memory
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
