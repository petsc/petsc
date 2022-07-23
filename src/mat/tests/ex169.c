
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
  PetscInt       ns=2;
  PetscMPIInt    size;
  PetscSubcomm   subc;
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f0",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f0 option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Reading matrix with %d processors\n",size));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  /*
     Determines amount of subcomunicators
  */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsub",&ns,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Splitting in %" PetscInt_FMT " subcommunicators\n",ns));
  PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)A),&subc));
  PetscCall(PetscSubcommSetNumber(subc,ns));
  PetscCall(PetscSubcommSetType(subc,PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommSetFromOptions(subc));
  PetscCall(MatCreateRedundantMatrix(A,0,PetscSubcommChild(subc),MAT_INITIAL_MATRIX,&Ar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Copying matrix\n"));
  PetscCall(MatDuplicate(Ar,MAT_COPY_VALUES,&C));
  PetscCall(MatAXPY(Ar,0.1,C,DIFFERENT_NONZERO_PATTERN));
  PetscCall(PetscSubcommDestroy(&subc));

  /*
     Free memory
  */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Ar));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump

TEST*/
