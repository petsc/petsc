static char help[] = "Test memory scalability of MatMatMult() for AIJ and DENSE matrices. \n\
Modified from the code contributed by Ian Lin <iancclin@umich.edu> \n\n";

/*
Example:
  mpiexec -n <np> ./ex33 -mem_view -matmatmult_Bbn <Bbn>
*/

#include <petsc.h>

PetscErrorCode Print_memory(PetscLogDouble mem)
{
  PetscErrorCode ierr;
  double         max_mem,min_mem;

  PetscFunctionBeginUser;
  ierr = MPI_Reduce(&mem, &max_mem, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);CHKERRMPI(ierr);
  ierr = MPI_Reduce(&mem, &min_mem, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);CHKERRMPI(ierr);
  max_mem = max_mem / 1024.0 / 1024.0;
  min_mem = min_mem / 1024.0 / 1024.0;
  ierr = PetscPrintf(MPI_COMM_WORLD, " max and min memory across all processors %.4f Mb, %.4f Mb.\n", (double)max_mem,(double)min_mem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Illustrate how to use MPI derived data types.
   It would save memory significantly. See MatMPIDenseScatter()
*/
PetscErrorCode TestMPIDerivedDataType()
{
  PetscErrorCode    ierr;
  MPI_Datatype      type1, type2,rtype1,rtype2;
  PetscInt          i,j;
  PetscScalar       buffer[24]; /* An array of 4 rows, 6 cols */
  MPI_Status        status;
  PetscMPIInt       rank,size,disp[2];

  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);CHKERRMPI(ierr);
  PetscCheckFalse(size < 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use at least 2 processors");
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);CHKERRMPI(ierr);

  if (rank == 0) {
    /* proc[0] sends 2 rows to proc[1] */
    for (i=0; i<24; i++) buffer[i] = (PetscScalar)i;

    disp[0] = 0;  disp[1] = 2;
    ierr = MPI_Type_create_indexed_block(2, 1, (const PetscMPIInt *)disp, MPIU_SCALAR, &type1);CHKERRMPI(ierr);
    /* one column has 4 entries */
    ierr = MPI_Type_create_resized(type1,0,4*sizeof(PetscScalar),&type2);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&type2);CHKERRMPI(ierr);
    ierr = MPI_Send(buffer, 6, type2, 1, 123, MPI_COMM_WORLD);CHKERRMPI(ierr);

  } else if (rank == 1) {
    /* proc[1] receives 2 rows from proc[0], and put them into contiguous rows, starting at the row 1 (disp[0]) */
    PetscInt blen = 2;
    for (i=0; i<24; i++) buffer[i] = 0.0;

    disp[0] = 1;
    ierr = MPI_Type_create_indexed_block(1, blen, (const PetscMPIInt *)disp, MPIU_SCALAR, &rtype1);CHKERRMPI(ierr);
    ierr = MPI_Type_create_resized(rtype1, 0, 4*sizeof(PetscScalar), &rtype2);CHKERRMPI(ierr);

    ierr = MPI_Type_commit(&rtype2);CHKERRMPI(ierr);
    ierr = MPI_Recv(buffer, 6, rtype2, 0, 123, MPI_COMM_WORLD, &status);CHKERRMPI(ierr);
    for (i=0; i<4; i++) {
      for (j=0; j<6; j++) {
        ierr = PetscPrintf(MPI_COMM_SELF,"  %g", (double)PetscRealPart(buffer[i+j*4]));CHKERRQ(ierr);
      }
      ierr = PetscPrintf(MPI_COMM_SELF,"\n");CHKERRQ(ierr);
    }
  }

  if (rank == 0) {
    ierr = MPI_Type_free(&type1);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&type2);CHKERRMPI(ierr);
  } else if (rank == 1) {
    ierr = MPI_Type_free(&rtype1);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&rtype2);CHKERRMPI(ierr);
  }
  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscInt          mA = 2700,nX = 80,nz = 40;
  /* PetscInt        mA=6,nX=5,nz=2; //small test */
  PetscLogDouble    mem;
  Mat               A,X,Y;
  PetscErrorCode    ierr;
  PetscBool         flg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_mpiderivedtype",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = TestMPIDerivedDataType();CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-mem_view",&flg,NULL);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(MPI_COMM_WORLD, "Before start,");CHKERRQ(ierr);
    ierr = Print_memory(mem);CHKERRQ(ierr);
  }

  ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,mA,mA,nz,PETSC_NULL,nz,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(MPI_COMM_WORLD, "After creating A,");CHKERRQ(ierr);
    ierr = Print_memory(mem);CHKERRQ(ierr);
  }

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,mA,nX,PETSC_NULL,&X);CHKERRQ(ierr);
  ierr = MatSetRandom(X,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(MPI_COMM_WORLD, "After creating X,");CHKERRQ(ierr);
    ierr = Print_memory(mem);CHKERRQ(ierr);
  }

  ierr = MatMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(MPI_COMM_WORLD, "After MatMatMult,");CHKERRQ(ierr);
    ierr = Print_memory(mem);CHKERRQ(ierr);
  }

  /* Test reuse */
  ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(MPI_COMM_WORLD, "After reuse MatMatMult,");CHKERRQ(ierr);
    ierr = Print_memory(mem);CHKERRQ(ierr);
  }

  /* Check accuracy */
  ierr = MatMatMultEqual(A,X,Y,10,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult()");

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&Y);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
