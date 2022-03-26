
static char help[] = "Test MatGetMultiProcBlock() and MatCreateRedundantMatrix() \n\
Reads a PETSc matrix and vector from a file and solves a linear system.\n\n";
/*
  Example:
  mpiexec -n 4 ./ex37 -f <input_file> -nsubcomm 2 -psubcomm_view -subcomm_type <1 or 2>
*/

#include <petscksp.h>
#include <petscsys.h>

int main(int argc,char **args)
{
  KSP            subksp;
  Mat            A,subA;
  Vec            x,b,u,subb,subx,subu;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscInt       i,m,n,its;
  PetscReal      norm;
  PetscMPIInt    rank,size;
  MPI_Comm       comm,subcomm;
  PetscSubcomm   psubcomm;
  PetscInt       nsubcomm=1,id;
  PetscScalar    *barray,*xarray,*uarray,*array,one=1.0;
  PetscInt       type=1;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /* Load the matrix */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /* Load the matrix; then destroy the viewer.*/
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  /* Create rhs vector b */
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecSet(b,one));

  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecSet(x,0.0));

  /* Test MatGetMultiProcBlock() */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-subcomm_type",&type,NULL));

  PetscCall(PetscSubcommCreate(comm,&psubcomm));
  PetscCall(PetscSubcommSetNumber(psubcomm,nsubcomm));
  if (type == PETSC_SUBCOMM_GENERAL) { /* user provides color, subrank and duprank */
    PetscMPIInt color,subrank,duprank,subsize;
    duprank = size-1 - rank;
    subsize = size/nsubcomm;
    PetscCheckFalse(subsize*nsubcomm != size,comm,PETSC_ERR_SUP,"This example requires nsubcomm %D divides size %D",nsubcomm,size);
    color   = duprank/subsize;
    subrank = duprank - color*subsize;
    PetscCall(PetscSubcommSetTypeGeneral(psubcomm,color,subrank));
  } else if (type == PETSC_SUBCOMM_CONTIGUOUS) {
    PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  } else if (type == PETSC_SUBCOMM_INTERLACED) {
    PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED));
  } else SETERRQ(psubcomm->parent,PETSC_ERR_SUP,"PetscSubcommType %D is not supported yet",type);
  PetscCall(PetscSubcommSetFromOptions(psubcomm));
  subcomm = PetscSubcommChild(psubcomm);

  /* Test MatCreateRedundantMatrix() */
  if (size > 1) {

    PetscMPIInt subrank=-1,color=-1;
    MPI_Comm    dcomm;

    if (rank == 0) {
      color = 0; subrank = 0;
    } else if (rank == 1) {
      color = 0; subrank = 1;
    } else {
      color = 1; subrank = 0;
    }

    PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD,&dcomm,NULL));
    PetscCallMPI(MPI_Comm_split(dcomm,color,subrank,&subcomm));

    PetscCall(MatCreate(subcomm,&subA));
    PetscCall(MatSetSizes(subA,PETSC_DECIDE,PETSC_DECIDE,10,10));
    PetscCall(MatSetFromOptions(subA));
    PetscCall(MatSetUp(subA));
    PetscCall(MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY));
    PetscCall(MatZeroEntries(subA));

    /* Test MatMult() */
    PetscCall(MatCreateVecs(subA,&subx,&subb));
    PetscCall(VecSet(subx,1.0));
    PetscCall(MatMult(subA,subx,subb));

    PetscCall(VecDestroy(&subx));
    PetscCall(VecDestroy(&subb));
    PetscCall(MatDestroy(&subA));
    PetscCall(PetscCommDestroy(&dcomm));
  }

  /* Create subA */
  PetscCall(MatGetMultiProcBlock(A,subcomm,MAT_INITIAL_MATRIX,&subA));
  PetscCall(MatGetMultiProcBlock(A,subcomm,MAT_REUSE_MATRIX,&subA));

  /* Create sub vectors without arrays. Place b's and x's local arrays into subb and subx */
  PetscCall(MatGetLocalSize(subA,&m,&n));
  PetscCall(VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&subb));
  PetscCall(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subx));
  PetscCall(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subu));

  PetscCall(VecGetArray(b,&barray));
  PetscCall(VecGetArray(x,&xarray));
  PetscCall(VecGetArray(u,&uarray));
  PetscCall(VecPlaceArray(subb,barray));
  PetscCall(VecPlaceArray(subx,xarray));
  PetscCall(VecPlaceArray(subu,uarray));

  /* Create linear solvers associated with subA */
  PetscCall(KSPCreate(subcomm,&subksp));
  PetscCall(KSPSetOperators(subksp,subA,subA));
  PetscCall(KSPSetFromOptions(subksp));

  /* Solve sub systems */
  PetscCall(KSPSolve(subksp,subb,subx));
  PetscCall(KSPGetIterationNumber(subksp,&its));

  /* check residual */
  PetscCall(MatMult(subA,subx,subu));
  PetscCall(VecAXPY(subu,-1.0,subb));
  PetscCall(VecNorm(u,NORM_2,&norm));
  if (norm > 1.e-4 && rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%D]  Number of iterations = %3D\n",rank,its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: Residual norm of each block |subb - subA*subx |= %g\n",(double)norm));
  }
  PetscCall(VecResetArray(subb));
  PetscCall(VecResetArray(subx));
  PetscCall(VecResetArray(subu));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-subvec_view",&id,&flg));
  if (flg && rank == id) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%D] subb:\n", rank));
    PetscCall(VecGetArray(subb,&array));
    for (i=0; i<m; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i])));
    PetscCall(VecRestoreArray(subb,&array));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%D] subx:\n", rank));
    PetscCall(VecGetArray(subx,&array));
    for (i=0; i<m; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i])));
    PetscCall(VecRestoreArray(subx,&array));
  }

  PetscCall(VecRestoreArray(x,&xarray));
  PetscCall(VecRestoreArray(b,&barray));
  PetscCall(VecRestoreArray(u,&uarray));
  PetscCall(MatDestroy(&subA));
  PetscCall(VecDestroy(&subb));
  PetscCall(VecDestroy(&subx));
  PetscCall(VecDestroy(&subu));
  PetscCall(KSPDestroy(&subksp));
  PetscCall(PetscSubcommDestroy(&psubcomm));
  if (size > 1) {
    PetscCallMPI(MPI_Comm_free(&subcomm));
  }
  PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u)); PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex37.out

    test:
      suffix: 2
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      output_file: output/ex37.out

    test:
      suffix: mumps
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 2 -pc_factor_mat_solver_type mumps -pc_type lu
      requires: datafilespath  mumps !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      output_file: output/ex37.out

    test:
      suffix: 3
      nsize: 4
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 2 -subcomm_type 0
      requires: datafilespath  !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex37.out

    test:
      suffix: 4
      nsize: 4
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 2 -subcomm_type 1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex37.out

    test:
      suffix: 5
      nsize: 4
      args: -f ${DATAFILESPATH}/matrices/small -nsubcomm 2 -subcomm_type 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex37.out

TEST*/
