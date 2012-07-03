/*T
   Concepts: Sieve^'Hat' Sieve
   Concepts: Sieve^cone completion of a Sieve
   Concepts: Sieve^viewing a Sieve
   Processors: n
T*/

/*
  Construct a simple 'Hat' Sieve of the form
           (0 , 0)     (0 , 1)       ...       (0 , r)
           /  |  \     /  |  \                 /  |  \
     (-1,0)(-1, 1)(-1,2)(-1,3)(-1,4) ...(-1,b0)(-1,b1)(-1,b2)
         
  where r denotes rank, and bi = 2*r+i, i = 0,1,2.
  Each process contains a local Sieve with a single node (0,r) in the cap
  and its base, as indicated in the diagram.  Thus, the local base size is 3
  and the total base size is 2*s + 1, where 's' denotes the communicator size.

  Each process creates its local Sieve; then the processes collectively 
  complete the Sieve using cone completion.  Since each process shares some
  base nodes with other processes, the completion Stack is nonempty for each 
  process.  The footprint part of the Stack indicates the origin of each completed
  cone node.
  
*/

static char help[] = "Constructs, completes, and views a simple 'hat' Sieve.\n\n";

#include <petscmesh.h>
#include <petscviewer.h>
#include <stdlib.h>
#include <string.h>

typedef ALE::Three::Sieve<ALE::def::Point,int,int> sieve_type;

PetscErrorCode CreateSieve(MPI_Comm comm, int debug)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE::Obj<sieve_type> sieve = sieve_type(comm, debug);
  ierr = PetscPrintf(comm, "Creating a 'Hat' Sieve of global size %d: 3 base nodes and 1 cap node per process\n", 
                     3*sieve->commSize()+1);CHKERRQ(ierr);
  for(int i = 0; i < 3; i++) {
    sieve_type::point_type p(sieve->commRank(), 0), b(-1,2*sieve->commRank()+i);
    sieve->addArrow(p,b);
  }
  sieve->view(std::cout, "Hat");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscBool      flag;
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    ierr = CreateSieve(comm, verbosity);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

