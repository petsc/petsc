
static char help[] = "Tests using PetscViewerGetSubViewer() recursively\n\n";

/*T
   Concepts: viewers
   Processors: n
T*/
#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
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
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  PetscCheckFalse(size < 4,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must run with at least 4 MPI processes");
  CHKERRQ(PetscOptionsGetViewer(comm,NULL,NULL,"-viewer",&viewer,&format,&flg));
  PetscCheckFalse(!viewer,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must use -viewer option");

  CHKERRQ(PetscViewerASCIIPrintf(viewer,"Print called on original full viewer %d\n",PetscGlobalRank));

  CHKERRQ(PetscSubcommCreate(comm,&psubcomm));
  CHKERRQ(PetscSubcommSetNumber(psubcomm,2));
  CHKERRQ(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  CHKERRQ(PetscSubcommSetFromOptions(psubcomm));
  subcomm = PetscSubcommChild(psubcomm);

  CHKERRQ(PetscViewerGetSubViewer(viewer,subcomm,&subviewer));

  CHKERRQ(PetscViewerASCIIPrintf(subviewer,"  Print called on sub viewers %d\n",PetscGlobalRank));

  CHKERRQ(PetscSubcommCreate(subcomm,&psubsubcomm));
  CHKERRQ(PetscSubcommSetNumber(psubsubcomm,2));
  CHKERRQ(PetscSubcommSetType(psubsubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  CHKERRQ(PetscSubcommSetFromOptions(psubsubcomm));
  subsubcomm = PetscSubcommChild(psubsubcomm);

  CHKERRQ(PetscViewerGetSubViewer(subviewer,subsubcomm,&subsubviewer));

  CHKERRQ(PetscViewerASCIIPrintf(subsubviewer,"  Print called on sub sub viewers %d\n",PetscGlobalRank));

  CHKERRQ(PetscViewerRestoreSubViewer(subviewer,subsubcomm,&subsubviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,subcomm,&subviewer));

  CHKERRQ(PetscSubcommDestroy(&psubsubcomm));
  CHKERRQ(PetscSubcommDestroy(&psubcomm));
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      args: -viewer

TEST*/
