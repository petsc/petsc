#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.6 1997/04/13 18:17:43 curfman Exp $";
#endif

static char help[] = "This is an introductory PETSc example that illustrates printing.\n\n";

/*T
   Concepts: Introduction to PETSc;
   Routines: PetscInitialize(); PetscPrintf(); PetscFinalize();
   Processors: n
T*/
 
#include "petsc.h"
int main(int argc,char **argv)
{
  int ierr, rank,size;

  /*
    Every PETSc program should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints all the useful optioins that can be applied at 
                 the command line. If the user wishes to add to this list,
                 additional help messages, he can do it using this variable.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help); CHKERRA(ierr);

  /* 
     The following MPI calls give the number of processors
     being used and the rank of this process in the group.
   */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRA(ierr);

  /* 
     Here I would like to print only one message which represents
     all the processors in the group. I use PetscPrintf() with the 
     communicator PETSC_COMM_WORLD, so that a single message is
     printed representng PETSC_COMM_WORLD, i.e., all the processors.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"No of processors = %d rank = %d\n",size,rank);CHKERRA(ierr);

  /*
    Here a barrier is used to separate the two program states.
  */
  ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRA(ierr);

  /*
    Here I simply use PetscPrintf(). So this time the output 
    from different processors does not come in any particular
    order. Note the use of the communicator PETSC_COMM_SELF, which
    indicates that each processor independently prints to the
    screen.
  */
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Jumbled Hello World\n",rank); CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).  See PetscFinalize()
     manpage for more information.
  */
  ierr = PetscFinalize(); CHKERRA(ierr);
  return 0;
}
