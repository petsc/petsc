/*$Id: ex2.c,v 1.94 2001/08/07 21:30:54 bsmith Exp $*/

/* Program usage:  mpirun -np <procs> ex2 [-help] [all PETSc options] */ 

static char help[] = "Solves a linear system in parallel with ESI.\n\
Input parameters include:\n\
  -n <mesh_n>       : number of mesh points in x-direction\n\n";

/*T
   Concepts: ESI^basic parallel example;
   Concepts: ESI^Laplacian, 1d
   Concepts: Laplacian, 1d
   Processors: n
T*/

/* 
  Include "esi/petsc/solveriterative.h" so that we can use the PETSc ESI interface.

*/
#include "esi/petsc/solveriterative.h"
#include "esi/petsc/matrix.h"
#include "esi/petsc/vector.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  ::esi::IndexSpace<int>                  *indexspace;
  ::esi::Vector<double,int>               *x,*b;      
  ::esi::Operator<double,int>             *op;  
  ::esi::SolverIterative<double,int>      *solver;    
  ::esi::MatrixRowWriteAccess<double,int> *A;
  int                                     ierr,i,n = 3,Istart,Iend,c[3],N;
  double                                  v[3];
  ::esi::IndexSpaceFactory<int>           *ifactory;
  ::esi::VectorFactory<double,int>        *vfactory;
  ::esi::OperatorFactory<double,int>      *ofactory;
  ::esi::SolverIterative<double,int>      *osolver;     /* linear solver context */

  PetscInitialize(&argc,&args,(char *)0,help);
  /*
           Load up the factorys we will need to create our objects
  */
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::IndexSpace",reinterpret_cast<void *&>(ifactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::Matrix",reinterpret_cast<void *&>(ofactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::Vector",reinterpret_cast<void *&>(vfactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::SolverIterative",reinterpret_cast<void *&>(osolver));CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /*
        Define the layout of the vectors and matrices across the processors
  */
  ierr = ifactory->getIndexSpace("MPI",(void*)&PETSC_COMM_WORLD,n,indexspace);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good 
     performance.  Preallocation is not possible via the generic
     matrix creation routine
  */
  ierr = ofactory->getOperator(*indexspace,*indexspace,op);CHKERRQ(ierr);
  
  /* 
     ESI parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr  = indexspace->getLocalPartitionOffset(Istart);CHKERRQ(ierr);
  ierr  = indexspace->getLocalSize(Iend);CHKERRQ(ierr);
  ierr  = indexspace->getGlobalSize(N);CHKERRQ(ierr);
  Iend += Istart;

  /* 
     Set matrix elements for the 1-D, three-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.

   */
  ierr = op->getInterface("esi::MatrixRowWriteAccess",reinterpret_cast<void *&>(A));CHKERRQ(ierr);
  if (Istart == 0) {
    Istart++;
  }
  if (Iend == N) {
    Iend--;
  }
  v[0] = -1.0; v[1] = 2.0; v[2] = -1.0;
  for (i=Istart; i<Iend; i++) {
    c[0] = i-1; c[1] = i; c[2] = i+1;
    ierr = A->copyIntoRow(i,v,c,3);CHKERRQ(ierr);
  }
  ierr = A->loadComplete();CHKERRQ(ierr);

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */

  /* 
     Create parallel vectors.
      - We form 1 vector from scratch and then duplicate as needed.
      - When using VecCreate(), VecSetSizes and VecSetFromOptions()
        in this example, we specify only the
        vector's global dimension; the parallel partitioning is determined
        at runtime. 
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.  
      - The user can alternatively specify the local vector and matrix
        dimensions when more sophisticated partitioning is needed
        (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
        below).
  */

  /* 
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */



  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
