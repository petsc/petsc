
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
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* Load the matrix; then destroy the viewer.*/
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* Create rhs vector b */
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecSet(b,one);CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* Test MatGetMultiProcBlock() */
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsubcomm",&nsubcomm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-subcomm_type",&type,NULL);CHKERRQ(ierr);

  ierr = PetscSubcommCreate(comm,&psubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubcomm,nsubcomm);CHKERRQ(ierr);
  if (type == PETSC_SUBCOMM_GENERAL) { /* user provides color, subrank and duprank */
    PetscMPIInt color,subrank,duprank,subsize;
    duprank = size-1 - rank;
    subsize = size/nsubcomm;
    if (subsize*nsubcomm != size) SETERRQ2(comm,PETSC_ERR_SUP,"This example requires nsubcomm %D divides size %D",nsubcomm,size);
    color   = duprank/subsize;
    subrank = duprank - color*subsize;
    ierr    = PetscSubcommSetTypeGeneral(psubcomm,color,subrank);CHKERRQ(ierr);
  } else if (type == PETSC_SUBCOMM_CONTIGUOUS) {
    ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
  } else if (type == PETSC_SUBCOMM_INTERLACED) {
    ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
  } else SETERRQ1(psubcomm->parent,PETSC_ERR_SUP,"PetscSubcommType %D is not supported yet",type);
  ierr = PetscSubcommSetFromOptions(psubcomm);CHKERRQ(ierr);
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

    ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&dcomm,NULL);CHKERRQ(ierr);
    ierr = MPI_Comm_split(dcomm,color,subrank,&subcomm);CHKERRMPI(ierr);

    ierr = MatCreate(subcomm,&subA);CHKERRQ(ierr);
    ierr = MatSetSizes(subA,PETSC_DECIDE,PETSC_DECIDE,10,10);CHKERRQ(ierr);
    ierr = MatSetFromOptions(subA);CHKERRQ(ierr);
    ierr = MatSetUp(subA);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatZeroEntries(subA);CHKERRQ(ierr);

    /* Test MatMult() */
    ierr = MatCreateVecs(subA,&subx,&subb);CHKERRQ(ierr);
    ierr = VecSet(subx,1.0);CHKERRQ(ierr);
    ierr = MatMult(subA,subx,subb);CHKERRQ(ierr);

    ierr = VecDestroy(&subx);CHKERRQ(ierr);
    ierr = VecDestroy(&subb);CHKERRQ(ierr);
    ierr = MatDestroy(&subA);CHKERRQ(ierr);
    ierr = PetscCommDestroy(&dcomm);CHKERRQ(ierr);
  }

  /* Create subA */
  ierr = MatGetMultiProcBlock(A,subcomm,MAT_INITIAL_MATRIX,&subA);CHKERRQ(ierr);
  ierr = MatGetMultiProcBlock(A,subcomm,MAT_REUSE_MATRIX,&subA);CHKERRQ(ierr);

  /* Create sub vectors without arrays. Place b's and x's local arrays into subb and subx */
  ierr = MatGetLocalSize(subA,&m,&n);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&subb);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subx);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&subu);CHKERRQ(ierr);

  ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(u,&uarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subb,barray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subx,xarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(subu,uarray);CHKERRQ(ierr);

  /* Create linear solvers associated with subA */
  ierr = KSPCreate(subcomm,&subksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(subksp,subA,subA);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(subksp);CHKERRQ(ierr);

  /* Solve sub systems */
  ierr = KSPSolve(subksp,subb,subx);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(subksp,&its);CHKERRQ(ierr);

  /* check residual */
  ierr = MatMult(subA,subx,subu);CHKERRQ(ierr);
  ierr = VecAXPY(subu,-1.0,subb);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-4 && !rank) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%D]  Number of iterations = %3D\n",rank,its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Residual norm of each block |subb - subA*subx |= %g\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = VecResetArray(subb);CHKERRQ(ierr);
  ierr = VecResetArray(subx);CHKERRQ(ierr);
  ierr = VecResetArray(subu);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-subvec_view",&id,&flg);CHKERRQ(ierr);
  if (flg && rank == id) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] subb:\n", rank);CHKERRQ(ierr);
    ierr = VecGetArray(subb,&array);CHKERRQ(ierr);
    for (i=0; i<m; i++) {ierr = PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i]));CHKERRQ(ierr);}
    ierr = VecRestoreArray(subb,&array);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] subx:\n", rank);CHKERRQ(ierr);
    ierr = VecGetArray(subx,&array);CHKERRQ(ierr);
    for (i=0; i<m; i++) {ierr = PetscPrintf(PETSC_COMM_SELF,"%g\n",(double)PetscRealPart(array[i]));CHKERRQ(ierr);}
    ierr = VecRestoreArray(subx,&array);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&uarray);CHKERRQ(ierr);
  ierr = MatDestroy(&subA);CHKERRQ(ierr);
  ierr = VecDestroy(&subb);CHKERRQ(ierr);
  ierr = VecDestroy(&subx);CHKERRQ(ierr);
  ierr = VecDestroy(&subu);CHKERRQ(ierr);
  ierr = KSPDestroy(&subksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&psubcomm);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MPI_Comm_free(&subcomm);CHKERRMPI(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);

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
