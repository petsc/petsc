static char help[] = "Test memory scalability of MatMatMult() for AIJ and DENSE matrices. \n\
Modified from the code contributed by Ian Lin <iancclin@umich.edu> \n\n";

/*
Example:
  mpiexec -n <np> ./ex33 -mem_view -matmatmult_Bbn <Bbn>
*/

#include <petsc.h>

PetscErrorCode Print_memory(PetscLogDouble mem)
{
  double         max_mem,min_mem;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Reduce(&mem, &max_mem, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&mem, &min_mem, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  max_mem = max_mem / 1024.0 / 1024.0;
  min_mem = min_mem / 1024.0 / 1024.0;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, " max and min memory across all processors %.4f Mb, %.4f Mb.\n", (double)max_mem,(double)min_mem));
  PetscFunctionReturn(0);
}

/*
   Illustrate how to use MPI derived data types.
   It would save memory significantly. See MatMPIDenseScatter()
*/
PetscErrorCode TestMPIDerivedDataType()
{
  MPI_Datatype      type1, type2,rtype1,rtype2;
  PetscInt          i,j;
  PetscScalar       buffer[24]; /* An array of 4 rows, 6 cols */
  MPI_Status        status;
  PetscMPIInt       rank,size,disp[2];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCheck(size >= 2,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"Must use at least 2 processors");
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0) {
    /* proc[0] sends 2 rows to proc[1] */
    for (i=0; i<24; i++) buffer[i] = (PetscScalar)i;

    disp[0] = 0;  disp[1] = 2;
    PetscCallMPI(MPI_Type_create_indexed_block(2, 1, (const PetscMPIInt *)disp, MPIU_SCALAR, &type1));
    /* one column has 4 entries */
    PetscCallMPI(MPI_Type_create_resized(type1,0,4*sizeof(PetscScalar),&type2));
    PetscCallMPI(MPI_Type_commit(&type2));
    PetscCallMPI(MPI_Send(buffer, 6, type2, 1, 123, MPI_COMM_WORLD));

  } else if (rank == 1) {
    /* proc[1] receives 2 rows from proc[0], and put them into contiguous rows, starting at the row 1 (disp[0]) */
    PetscInt blen = 2;
    for (i=0; i<24; i++) buffer[i] = 0.0;

    disp[0] = 1;
    PetscCallMPI(MPI_Type_create_indexed_block(1, blen, (const PetscMPIInt *)disp, MPIU_SCALAR, &rtype1));
    PetscCallMPI(MPI_Type_create_resized(rtype1, 0, 4*sizeof(PetscScalar), &rtype2));

    PetscCallMPI(MPI_Type_commit(&rtype2));
    PetscCallMPI(MPI_Recv(buffer, 6, rtype2, 0, 123, MPI_COMM_WORLD, &status));
    for (i=0; i<4; i++) {
      for (j=0; j<6; j++) {
        PetscCall(PetscPrintf(MPI_COMM_SELF,"  %g", (double)PetscRealPart(buffer[i+j*4])));
      }
      PetscCall(PetscPrintf(MPI_COMM_SELF,"\n"));
    }
  }

  if (rank == 0) {
    PetscCallMPI(MPI_Type_free(&type1));
    PetscCallMPI(MPI_Type_free(&type2));
  } else if (rank == 1) {
    PetscCallMPI(MPI_Type_free(&rtype1));
    PetscCallMPI(MPI_Type_free(&rtype2));
  }
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscInt          mA = 2700,nX = 80,nz = 40;
  /* PetscInt        mA=6,nX=5,nz=2; //small test */
  PetscLogDouble    mem;
  Mat               A,X,Y;
  PetscBool         flg = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_mpiderivedtype",&flg,NULL));
  if (flg) {
    PetscCall(TestMPIDerivedDataType());
    PetscCall(PetscFinalize());
    return 0;
  }

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-mem_view",&flg,NULL));
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (flg) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Before start,"));
    PetscCall(Print_memory(mem));
  }

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,mA,mA,nz,PETSC_NULL,nz,PETSC_NULL,&A));
  PetscCall(MatSetRandom(A,PETSC_NULL));
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (flg) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "After creating A,"));
    PetscCall(Print_memory(mem));
  }

  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,mA,nX,PETSC_NULL,&X));
  PetscCall(MatSetRandom(X,PETSC_NULL));
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (flg) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "After creating X,"));
    PetscCall(Print_memory(mem));
  }

  PetscCall(MatMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Y));
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (flg) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "After MatMatMult,"));
    PetscCall(Print_memory(mem));
  }

  /* Test reuse */
  PetscCall(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y));
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (flg) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "After reuse MatMatMult,"));
    PetscCall(Print_memory(mem));
  }

  /* Check accuracy */
  PetscCall(MatMatMultEqual(A,X,Y,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult()");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 4
      output_file: output/ex33.out

   test:
      suffix: 2
      nsize: 8
      output_file: output/ex33.out

   test:
      suffix: 3
      nsize: 2
      args: -test_mpiderivedtype
      output_file: output/ex33_3.out

TEST*/
