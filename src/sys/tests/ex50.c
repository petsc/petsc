
static char help[] = "Tests using PetscViewerGetSubViewer() recursively\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscViewer       viewer,subviewer,subsubviewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscSubcomm      psubcomm,psubsubcomm;
  MPI_Comm          comm,subcomm,subsubcomm;
  PetscMPIInt       size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCheck(size >= 4,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with at least 4 MPI processes");
  PetscCall(PetscOptionsGetViewer(comm,NULL,NULL,"-viewer",&viewer,&format,&flg));
  PetscCheck(viewer,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must use -viewer option");

  PetscCall(PetscViewerASCIIPrintf(viewer,"Print called on original full viewer %d\n",PetscGlobalRank));

  PetscCall(PetscSubcommCreate(comm,&psubcomm));
  PetscCall(PetscSubcommSetNumber(psubcomm,2));
  PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  PetscCall(PetscSubcommSetFromOptions(psubcomm));
  subcomm = PetscSubcommChild(psubcomm);

  PetscCall(PetscViewerGetSubViewer(viewer,subcomm,&subviewer));

  PetscCall(PetscViewerASCIIPrintf(subviewer,"  Print called on sub viewers %d\n",PetscGlobalRank));

  PetscCall(PetscSubcommCreate(subcomm,&psubsubcomm));
  PetscCall(PetscSubcommSetNumber(psubsubcomm,2));
  PetscCall(PetscSubcommSetType(psubsubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  PetscCall(PetscSubcommSetFromOptions(psubsubcomm));
  subsubcomm = PetscSubcommChild(psubsubcomm);

  PetscCall(PetscViewerGetSubViewer(subviewer,subsubcomm,&subsubviewer));

  PetscCall(PetscViewerASCIIPrintf(subsubviewer,"  Print called on sub sub viewers %d\n",PetscGlobalRank));

  PetscCall(PetscViewerRestoreSubViewer(subviewer,subsubcomm,&subsubviewer));
  PetscCall(PetscViewerRestoreSubViewer(viewer,subcomm,&subviewer));

  PetscCall(PetscSubcommDestroy(&psubsubcomm));
  PetscCall(PetscSubcommDestroy(&psubcomm));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args: -viewer

TEST*/
