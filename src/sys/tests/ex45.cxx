
static char help[] = "Demonstrates call PETSc first and then Trilinos in the same program.\n\n";

/*T
   Concepts: introduction to PETSc^Trilinos
   Processors: n

   Example obtained from: http://trilinos.org/docs/dev/packages/tpetra/doc/html/Tpetra_Lesson01.html
T*/

#include <petscsys.h>
#include <Teuchos_DefaultMpiComm.hpp> // wrapper for MPI_Comm
#include <Tpetra_Version.hpp> // Tpetra version string

// Do something with the given communicator.  In this case, we just
// print Tpetra's version to stdout on Process 0 in the given
// communicator.
void
exampleRoutine (const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  if (comm->getRank () == 0) {
    // On (MPI) Process 0, print out the Tpetra software version.
    std::cout << Tpetra::version () << std::endl << std::endl;
  }
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  // These "using" declarations make the code more concise, in that
  // you don't have to write the namespace along with the class or
  // object name.  This is especially helpful with commonly used
  // things like std::endl or Teuchos::RCP.
  using std::cout;
  using std::endl;
  using Teuchos::Comm;
  using Teuchos::MpiComm;
  using Teuchos::RCP;
  using Teuchos::rcp;

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
  RCP<const Comm<int> > comm (new MpiComm<int> (PETSC_COMM_WORLD));
  // Get my process' rank, and the total number of processes.
  // Equivalent to MPI_Comm_rank resp. MPI_Comm_size.
  const int myRank = comm->getRank ();
  const int size = comm->getSize ();
  if (myRank == 0) {
    cout << "Total number of processes: " << size << endl;
  }
  // Do something with the new communicator.
  exampleRoutine (comm);
  // This tells the Trilinos test framework that the test passed.
  if (myRank == 0) {
    cout << "End Result: TEST PASSED" << endl;
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: trilinos

   test:
      nsize: 3
      filter: grep -v "Tpetra in Trilinos"

TEST*/
