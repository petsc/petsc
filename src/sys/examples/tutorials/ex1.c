#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.2 1996/12/16 16:58:22 balay Exp bsmith $";
#endif

static char help[] = "This is an introductory PETSc example \n\n";

/*T
   Concepts: Introduction to PETSc;
   Routines: PetscInitialize();PetscPrintf();PetscSynchronizedPrintf();
   Routines: PetscSynchronizedFlush();PetscFinalize();
   Processors: n
T*/
 
#include "petsc.h"
int main(int argc,char **argv)
{
  int ierr, rank,size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints all the useful optioins that can be applied at 
                 the command line. If the user wishes to add to this list,
                 additional help messages, he can do it using this variable.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help); CHKERRA(ierr);
  /* 
     Thw following MPI calls gives the number of processors
     being used, and the rank of this process in the group.
   */
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size); CHKERRA(ierr);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank); CHKERRA(ierr);
  /* 
     Here I would like to print only one message which represents
     all the processors in the group. I use PetscPrintf() on the 
     communicatior MPI_COMM_WORLD. This way only one message is
     printed represeintng MPI_COMM_WORLD, i.e all the processors.
  */
  ierr = PetscPrintf(MPI_COMM_WORLD,"No of processors = %d rank = %d\n",size,rank); CHKERRA(ierr);
  /* 
     Here I would like to print info from each processor such that,
     output from proc "n" appears after output from proc "n-1".
     To do this I use a combination PetscSynchronizedPrintf() and
     PetscSynchronizedFlush(). The communicator used is MPI_COMM_WORLD,
     All the processors print the message one after another. 
     PetscSynchronizedFlush() Indicates that the current proc in MPI_COMM
     has printed all it wanted to, and now, the next proc in MPI_COMM
     can start printing on the screen.
     */
  ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d] Synchronized Hello World.\n",rank); CHKERRA(ierr);
  ierr = PetscSynchronizedPrintf(MPI_COMM_WORLD,"[%d] Synchronized Hello World - Part II.\n",rank); CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(MPI_COMM_WORLD); CHKERRA(ierr);
  /*
    Here a barrier is used to separate the two states.
  */
  ierr = MPI_Barrier(MPI_COMM_WORLD); CHKERRA(ierr);
  /*
    Here I simply use PetscPrintf(). So this time the output 
    from different proc does not come in order. Note the use of
    MPI_COMM_SELF. This way only one proc from MPI_COMM_SELF,
    i.e each proc in MPI_COMM_WORLD prints to the screen.
  */
  ierr = PetscPrintf(MPI_COMM_SELF,"[%d] Jumbled Hello World\n",rank); CHKERRA(ierr);
  /*
    Every PETSc program has to end with a call to PETScFinalize().
    This routine calls MPI_Finalize()
 */
  ierr = PetscFinalize(); CHKERRA(ierr);
  return 0;
}
