
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
  PetscErrorCode ierr;
  PetscInt       i,m,n,its;
  PetscReal      norm;
  PetscMPIInt    rank,size;
  MPI_Comm       comm,subcomm;
  PetscSubcomm   psubcomm;
  PetscInt       nsubcomm=1,id;
  PetscScalar    *barray,*xarray,*uarray,*array,one=1.0;
  PetscInt       type=1;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Load the matrix */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /* Load the matrix; then destroy the viewer.*/
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  /* Create rhs vector b */
  CHKERRQ(MatGetLocalSize(A,&m,NULL));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecSetSizes(b,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecSet(b,one));

  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));
  CHKERRQ(VecSet(x,0.0));

  /* Test MatGetMultiProcBlock() */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-subcomm_type",&type,NULL));

  CHKERRQ(PetscSubcommCreate(comm,&psubcomm));
  CHKERRQ(PetscSubcommSetNumber(psubcomm,nsubcomm));
  if (type == PETSC_SUBCOMM_GENERAL) { /* user provides color, subrank and duprank */
    PetscMPIInt color,subrank,duprank,subsize;
    duprank = size-1 - rank;
    subsize = size/nsubcomm;
    PetscCheckFalse(subsize*nsubcomm != size,comm,PETSC_ERR_SUP,"This example requires nsubcomm %D divides size %D",nsubcomm,size);
    color   = duprank/subsize;
    subrank = duprank - color*subsize;
    CHKERRQ(PetscSubcommSetTypeGeneral(psubcomm,color,subrank));
  } else if (type == PETSC_SUBCOMM_CONTIGUOUS) {
    CHKERRQ(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
  } else if (type == PETSC_SUBCOMM_INTERLACED) {
    CHKERRQ(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED));
  } else SETERRQ(psubcomm->parent,PETSC_ERR_SUP,"PetscSubcommType %D is not supported yet",type);
  CHKERRQ(PetscSubcommSetFromOptions(psubcomm));
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

    CHKERRQ(PetscCommDuplicate(PETSC_COMM_WORLD,&dcomm,NULL));
    CHKERRMPI(MPI_Comm_split(dcomm,color,subrank,&subcomm));

    CHKERRQ(MatCreate(subcomm,&subA));
    CHKERRQ(MatSetSizes(subA,PETSC_DECIDE,PETSC_DECIDE,10,10));
    CHKERRQ(MatSetFromOptions(subA));
    CHKERRQ(MatSetUp(subA));
    CHKERRQ(MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatZeroEntries(subA));

    /* Test MatMult() */
    CHKERRQ(MatCreateVecs(subA,&subx,&subb));
    CHKERRQ(VecSet(subx,1.0));
    CHKERRQ(MatMult(subA,subx,subb));

    CHKERRQ(VecDestroy(&subx));
    CHKERRQ(VecDestroy(&subb));
    CHKERRQ(MatDestroy(&subA));
    CHKERRQ(PetscCommDestroy(&dcomm));
  }

  /* Create subA */
  CHKERRQ(MatGetMultiProcBlock(A,subcomm,MAT_INITIAL_MATRIX,&subA));
  CHKERRQ(MatGetMultiProcBlock(A,subcomm,MAT_REUSE_MATRIX,&subA));

  /* Create sub vectors without arrays. Place b's and x's local arrays into subb and subx */
  CHKERRQ(MatGetLocalSize(subA,&m,&n));
  CHKERRQ(VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&subb));
  CHKERRQ(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subx));
  CHKERRQ(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subu));

  CHKERRQ(VecGetArray(b,&barray));
  CHKERRQ(VecGetArray(x,&xarray));
  CHKERRQ(VecGetArray(u,&uarray));
  CHKERRQ(VecPlaceArray(subb,barray));
  CHKERRQ(VecPlaceArray(subx,xarray));
  CHKERRQ(VecPlaceArray(subu,uarray));

  /* Create linear solvers associated with subA */
  CHKERRQ(KSPCreate(subcomm,&subksp));
  CHKERRQ(KSPSetOperators(subksp,subA,subA));
  CHKERRQ(KSPSetFromOptions(subksp));

  /* Solve sub systems */
  CHKERRQ(KSPSolve(subksp,subb,subx));
  CHKERRQ(KSPGetIterationNumber(subksp,&its));

  /* check residual */
  CHKERRQ(MatMult(subA,subx,subu));
  CHKERRQ(VecAXPY(subu,-1.0,subb));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  if (norm > 1.e-4 && rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%D]  Number of iterations = %3D\n",rank,its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Residual norm of each block |subb - subA*subx |= %g\n",(double)norm));
  }
  CHKERRQ(VecResetArray(subb));
  CHKERRQ(VecResetArray(subx));
  CHKERRQ(VecResetArray(subu));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-subvec_view",&id,&flg));
  if (flg && rank == id) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%D] subb:\n", rank));
    CHKERRQ(VecGetArray(subb,&array));
    for (i=0; i<m; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i])));
    CHKERRQ(VecRestoreArray(subb,&array));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%D] subx:\n", rank));
    CHKERRQ(VecGetArray(subx,&array));
    for (i=0; i<m; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i])));
    CHKERRQ(VecRestoreArray(subx,&array));
  }

  CHKERRQ(VecRestoreArray(x,&xarray));
  CHKERRQ(VecRestoreArray(b,&barray));
  CHKERRQ(VecRestoreArray(u,&uarray));
  CHKERRQ(MatDestroy(&subA));
  CHKERRQ(VecDestroy(&subb));
  CHKERRQ(VecDestroy(&subx));
  CHKERRQ(VecDestroy(&subu));
  CHKERRQ(KSPDestroy(&subksp));
  CHKERRQ(PetscSubcommDestroy(&psubcomm));
  if (size > 1) {
    CHKERRMPI(MPI_Comm_free(&subcomm));
  }
  CHKERRQ(MatDestroy(&A)); CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&x));

  ierr = PetscFinalize();
  return ierr;
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
