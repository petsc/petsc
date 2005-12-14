/*T
   Concepts: PreSieve^'Hat' PreSieve
   Concepts: PreSieve^cone completion of a PreSieve
   Concepts: PreSieve^completion footprint
   Concepts: PreSieve^viewing a PreSieve
   Processors: n
T*/

/*
  Construct a simple 'Hat' PreSieve of the form
           (0 , 0)     (0 , 1)       ...       (0 , r)
           /  |  \     /  |  \                 /  |  \
     (-1,0)(-1, 1)(-1,2)(-1,3)(-1,4) ...(-1,b0)(-1,b1)(-1,b2)
         
  where r denotes rank, and bi = 2*r+i, i = 0,1,2.
  Each process contains a local PreSieve with a single node (0,r) in the cap
  and its base, as indicated in the diagram.  Thus, the local base size is 3
  and the total base size is 2*s + 1, where 's' denotes the communicator size.

  Each process creates its local PreSieve; then the processes collectively 
  complete the PreSieve using cone completion.  Since each process shares some
  base nodes with other processes, the completion Stack is nonempty for each 
  process.  The footprint part of the Stack indicates the origin of each completed
  cone node.
  
*/

static char help[] = "Constructs, completes, and views a simple 'hat' PreSieve.\n\n";

#include "petscda.h"
#include "petscviewer.h"
#include <stdlib.h>
#include <string.h>

#include <PreSieve.hh>
#include <Stack.hh>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     flag;
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;


  ALE::PreSieve *preSieve = new ALE::PreSieve(comm);
  ierr = PetscPrintf(comm, "Creating a 'Hat' PreSieve of global size %d: 3 base nodes and 1 cap node per process\n", 
                     3*preSieve->getCommSize()+1); CHKERRQ(ierr);
  for(int i = 0; i < 3; i++) {
    ALE::Point p(preSieve->getCommRank(), 0), b(-1,2*preSieve->getCommRank()+i);
    preSieve->addArrow(p,b);
  }
  preSieve->view("Hat");
  ALE::Stack *stack = preSieve->coneCompletion(ALE::PreSieve::completionTypePoint, ALE::PreSieve::footprintTypeCone);
  stack->view("'Hat's point cone completion with cone footprint'");

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

