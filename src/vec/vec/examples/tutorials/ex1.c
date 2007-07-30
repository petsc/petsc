//  mpi includes
#include <mpi.h>

//  petsc includes
#include <petsc.h>
#include <petscvec.h>

//  std includes

//  useful definitions
#define SIZE 36

int
main(int argc, char** argv) {

    //  Initialize the test code
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    //  Grab some useful MPI information
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //  Create test vectors
    Vec phin, eps;
    VecCreate(comm, &phin);
    VecSetSizes(phin, PETSC_DECIDE, SIZE);
    VecSetFromOptions(phin);
    VecDuplicate(phin, &eps);

    //  Put data into one of the vectors
    PetscInt ix[9];
    PetscScalar vecval[9] = {1.0};

    printf("%g %g %g %g %g %g %g %g %g\n",vecval[0],vecval[1],vecval[2],vecval[3],vecval[4],vecval[5],vecval[6],vecval[7],vecval[8]);
    if (rank == 0) {
        ix[0]=0;
        ix[1]=1;
        ix[2]=2;
        ix[3]=6;
        ix[4]=7;
        ix[5]=8;
        ix[6]=12;
        ix[7]=13;
        ix[8]=14;
        VecSetValues(phin, 9, ix, vecval, INSERT_VALUES);
    }
    if (rank == 1) {
        ix[0]=3;
        ix[1]=4;
        ix[2]=5;
        ix[3]=9;
        ix[4]=10;
        ix[5]=11;
        ix[6]=15;
        ix[7]=16;
        ix[8]=17;
        VecSetValues(phin, 9, ix, vecval, INSERT_VALUES);
    }
    if (rank == 2) {
        ix[0]=18;
        ix[1]=19;
        ix[2]=20;
        ix[3]=24;
        ix[4]=25;
        ix[5]=26;
        ix[6]=30;
        ix[7]=31;
        ix[8]=32;
        VecSetValues(phin, 9, ix, vecval, INSERT_VALUES);
    }
    if (rank == 3) {
        ix[0]=21;
        ix[1]=22;
        ix[2]=23;
        ix[3]=27;
        ix[4]=28;
        ix[5]=29;
        ix[6]=33;
        ix[7]=34;
        ix[8]=35;
        VecSetValues(phin, 9, ix, vecval, INSERT_VALUES);
    }

    //  Finalize the vector construction
    VecAssemblyBegin(phin);
    VecAssemblyEnd(phin);

    //  Free initialized variables
    VecDestroy(phin);

    //  Finalize the test code
    PetscFinalize();
    MPI_Finalize();

}
