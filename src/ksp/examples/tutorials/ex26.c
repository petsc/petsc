
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

   Note the usual PETSc objects all work. Those related to vec, mat, pc, ksp
are prefixed with esi_

*/
#include "esi/ESI.h"

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  ::esi::IndexSpace<int>                      *indexspace;
  ::esi::Vector<double,int>                   *x,*b;      
  ::esi::Operator<double,int>                 *op;  
  ::esi::SolverIterative<double,int>          *solver;    
  ::esi::MatrixRowWriteAccess<double,int>     *A;
  int                                         ierr,i,n = 3,Istart,Iend,c[3],N;
  double                                      v[3],*barray;
  ::esi::IndexSpace<int>::Factory             *ifactory;
  ::esi::Vector<double,int>::Factory          *vfactory;
  ::esi::Operator<double,int>::Factory        *ofactory;
  ::esi::SolverIterative<double,int>::Factory *sfactory;     /* linear solver context */

  PetscInitialize(&argc,&args,(char *)0,help);
  /*
           Load up the factorys we will need to create our objects
  */
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::IndexSpace",reinterpret_cast<void *&>(ifactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::Matrix",reinterpret_cast<void *&>(ofactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::Vector",reinterpret_cast<void *&>(vfactory));CHKERRQ(ierr);
  ierr = ESILoadFactory("MPI",(void*)&PETSC_COMM_WORLD,"esi::petsc::SolverIterative",reinterpret_cast<void *&>(sfactory));CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /*
        Define the layout of the vectors and matrices across the processors
  */
  ierr = ifactory->create("MPI",(void*)&PETSC_COMM_WORLD,n,PETSC_DECIDE,PETSC_DECIDE,indexspace);CHKERRQ(ierr);

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
  ierr = ofactory->create(*indexspace,*indexspace,op);CHKERRQ(ierr);
  
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
    v[0] = 1.0;
    ierr = A->copyIntoRow(Istart,v,&Istart,1);CHKERRQ(ierr);
    Istart++;
  }
  if (Iend == N) {
    Iend--;
    v[0] = 1.0;
    ierr = A->copyIntoRow(Iend,v,&Iend,1);CHKERRQ(ierr);
  }
  v[0] = -1.0; v[1] = 2.0; v[2] = -1.0;
  for (i=Istart; i<Iend; i++) {
    c[0] = i-1; c[1] = i; c[2] = i+1;
    ierr = A->copyIntoRow(i,v,c,3);CHKERRQ(ierr);
  }

  /* 
     Assemble matrix, using the 2-step process:
  */
  ierr = A->loadComplete();CHKERRQ(ierr);


  /* 
     Create parallel vectors.i
      - We form 1 vector from scratch and then duplicate as needed.
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.  
      - The user can alternatively specify the local vector and matrix
        dimensions when more sophisticated partitioning is needed
        (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
        below).
  */
  ierr = vfactory->create(*indexspace,x);CHKERRQ(ierr);
  ierr = x->clone(b);CHKERRQ(ierr);

  ierr = b->getCoefPtrReadWriteLock(barray);CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    barray[i-Istart] = i;
  }
  ierr = b->releaseCoefPtrLock(barray);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = sfactory->create("MPI",(void*)&PETSC_COMM_WORLD,solver);CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = solver->setOperator(*op);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = solver->solve(*b,*x);CHKERRQ(ierr);


  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  indexspace->deleteReference();
  op->deleteReference();
  x->deleteReference();
  b->deleteReference();
  solver->deleteReference();
  delete ifactory;
  delete vfactory;
  delete ofactory;
  delete sfactory;
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
