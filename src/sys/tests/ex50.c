
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
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size < 4,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must run with at least 4 MPI processes");
  ierr = PetscOptionsGetViewer(comm,NULL,NULL,"-viewer",&viewer,&format,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!viewer,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must use -viewer option");

  ierr = PetscViewerASCIIPrintf(viewer,"Print called on original full viewer %d\n",PetscGlobalRank);CHKERRQ(ierr);

  ierr = PetscSubcommCreate(comm,&psubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubcomm,2);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  ierr = PetscSubcommSetFromOptions(psubcomm);CHKERRQ(ierr);
  subcomm = PetscSubcommChild(psubcomm);

  ierr = PetscViewerGetSubViewer(viewer,subcomm,&subviewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(subviewer,"  Print called on sub viewers %d\n",PetscGlobalRank);CHKERRQ(ierr);

  ierr = PetscSubcommCreate(subcomm,&psubsubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubsubcomm,2);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(psubsubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
  /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
  ierr = PetscSubcommSetFromOptions(psubsubcomm);CHKERRQ(ierr);
  subsubcomm = PetscSubcommChild(psubsubcomm);

  ierr = PetscViewerGetSubViewer(subviewer,subsubcomm,&subsubviewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(subsubviewer,"  Print called on sub sub viewers %d\n",PetscGlobalRank);CHKERRQ(ierr);

  ierr = PetscViewerRestoreSubViewer(subviewer,subsubcomm,&subsubviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(viewer,subcomm,&subviewer);CHKERRQ(ierr);

  ierr = PetscSubcommDestroy(&psubsubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&psubcomm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      args: -viewer

TEST*/
