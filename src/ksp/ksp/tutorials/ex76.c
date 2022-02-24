#include <petscksp.h>

static char help[] = "Solves a linear system using PCHPDDM.\n\n";

int main(int argc,char **args)
{
  Vec                x,b;        /* computed solution and RHS */
  Mat                A,aux,X,B;  /* linear system matrix */
  KSP                ksp;        /* linear solver context */
  PC                 pc;
  IS                 is,sizes;
  const PetscInt     *idx;
  PetscMPIInt        rank,size;
  PetscInt           m,N = 1;
  const char         *deft = MATAIJ;
  PetscViewer        viewer;
  char               dir[PETSC_MAX_PATH_LEN],name[PETSC_MAX_PATH_LEN],type[256];
  PetscBool          flg;
#if defined(PETSC_USE_LOG)
  PetscLogEvent      event;
#endif
  PetscEventPerfInfo info1,info2;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 4,PETSC_COMM_WORLD,PETSC_ERR_USER,"This example requires 4 processes");
  ierr = PetscOptionsGetInt(NULL,NULL,"-rhs",&N,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&aux);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&is);CHKERRQ(ierr);
  ierr = PetscStrcpy(dir,".");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL);CHKERRQ(ierr);
  /* loading matrices */
  ierr = PetscSNPrintf(name,sizeof(name),"%s/sizes_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&sizes);CHKERRQ(ierr);
  ierr = ISLoad(sizes,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(sizes,&idx);CHKERRQ(ierr);
  ierr = MatSetSizes(A,idx[0],idx[1],idx[2],idx[3]);CHKERRQ(ierr);
  ierr = ISRestoreIndices(sizes,&idx);CHKERRQ(ierr);
  ierr = ISDestroy(&sizes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/A.dat",dir);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/is_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = ISLoad(is,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/Neumann_%d_%d.dat",dir,rank,size);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(aux,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-pc_hpddm_levels_1_st_share_sub_ksp",&flg,NULL);CHKERRQ(ierr);
  if (flg) { /* PETSc LU/Cholesky is struggling numerically for bs > 1          */
             /* only set the proper bs for the geneo_share_* tests, 1 otherwise */
    ierr = MatSetBlockSizesFromMats(aux,A,A);CHKERRQ(ierr);
  }
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(aux,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  /* ready for testing */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = MatConvert(A,type,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatConvert(aux,type,MAT_INPLACE_MATRIX,&aux);CHKERRQ(ierr);
  }
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCHPDDM);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-pc_hpddm_block_splitting",&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    ierr = PCHPDDMSetAuxiliaryMat(pc,is,aux,NULL,NULL);CHKERRQ(ierr);
    ierr = PCHPDDMHasNeumannMat(pc,PETSC_FALSE);CHKERRQ(ierr); /* PETSC_TRUE is fine as well, just testing */
  }
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-set_rhs",&flg,NULL);CHKERRQ(ierr);
  if (flg) { /* user-provided RHS for concurrent generalized eigenvalue problems                                   */
    Mat      a,c,P; /* usually assembled automatically in PCHPDDM, this is solely for testing PCHPDDMSetRHSMat() */
    PetscInt rstart,rend,location;
    ierr = MatDuplicate(aux,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr); /* duplicate so that MatStructure is SAME_NONZERO_PATTERN */
    ierr = MatGetDiagonalBlock(A,&a);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&m);CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,rend-rstart,m,1,NULL,&P);CHKERRQ(ierr);
    for (m = rstart; m < rend; ++m) {
      ierr = ISLocate(is,m,&location);CHKERRQ(ierr);
      PetscCheck(location >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "IS of the auxiliary Mat does not include all local rows of A");
      ierr = MatSetValue(P,m-rstart,location,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)a,MATSEQAIJ,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatPtAP(a,P,MAT_INITIAL_MATRIX,1.0,&X);CHKERRQ(ierr); /* MatPtAP() is used to extend diagonal blocks with zeros on the overlap */
    } else { /* workaround for MatPtAP() limitations with some types */
      ierr = MatConvert(a,MATSEQAIJ,MAT_INITIAL_MATRIX,&c);CHKERRQ(ierr);
      ierr = MatPtAP(c,P,MAT_INITIAL_MATRIX,1.0,&X);CHKERRQ(ierr);
      ierr = MatDestroy(&c);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = MatAXPY(B,1.0,X,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCHPDDMSetRHSMat(pc,B);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
#endif
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&aux);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  if (N > 1) {
    KSPType type;
    ierr = PetscOptionsClearValue(NULL,"-ksp_converged_reason");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&B);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&X);CHKERRQ(ierr);
    ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
    /* this is algorithmically optimal in the sense that blocks of vectors are coarsened or interpolated using matrix--matrix operations */
    /* PCHPDDM however heavily relies on MPI[S]BAIJ format for which there is no efficient MatProduct implementation */
    ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
    ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
    ierr = PetscStrcmp(type,KSPHPDDM,&flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM)
    if (flg) {
      PetscReal    norm;
      KSPHPDDMType type;
      ierr = KSPHPDDMGetType(ksp,&type);CHKERRQ(ierr);
      if (type == KSP_HPDDM_TYPE_PREONLY || type == KSP_HPDDM_TYPE_CG || type == KSP_HPDDM_TYPE_GMRES || type == KSP_HPDDM_TYPE_GCRODR) {
        Mat C;
        ierr = MatDuplicate(X,MAT_DO_NOT_COPY_VALUES,&C);CHKERRQ(ierr);
        ierr = KSPSetMatSolveBatchSize(ksp,1);CHKERRQ(ierr);
        ierr = KSPMatSolve(ksp,B,C);CHKERRQ(ierr);
        ierr = MatAYPX(C,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRQ(ierr);
        ierr = MatDestroy(&C);CHKERRQ(ierr);
        PetscCheckFalse(norm > 100*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"KSPMatSolve() and KSPSolve() difference has nonzero norm %g with pseudo-block KSPHPDDMType %s",(double)norm,KSPHPDDMTypes[type]);
      }
    }
#endif
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCHPDDM,&flg);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  if (flg) {
    ierr = PCHPDDMGetSTShareSubKSP(pc,&flg);CHKERRQ(ierr);
  }
#endif
  if (flg && PetscDefined(USE_LOG)) {
    ierr = PetscLogEventRegister("MatLUFactorSym",PC_CLASSID,&event);CHKERRQ(ierr);
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info1);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("MatLUFactorNum",PC_CLASSID,&event);CHKERRQ(ierr);
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info2);CHKERRQ(ierr);
    if (info1.count || info2.count) {
      PetscCheckFalse(info2.count <= info1.count,PETSC_COMM_SELF,PETSC_ERR_PLIB,"LU numerical factorization (%d) not called more times than LU symbolic factorization (%d), broken -pc_hpddm_levels_1_st_share_sub_ksp",info2.count,info1.count);
    } else {
      ierr = PetscLogEventRegister("MatCholFctrSym",PC_CLASSID,&event);CHKERRQ(ierr);
      ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info1);CHKERRQ(ierr);
      ierr = PetscLogEventRegister("MatCholFctrNum",PC_CLASSID,&event);CHKERRQ(ierr);
      ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info2);CHKERRQ(ierr);
      PetscCheckFalse(info2.count <= info1.count,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cholesky numerical factorization (%d) not called more times than Cholesky symbolic factorization (%d), broken -pc_hpddm_levels_1_st_share_sub_ksp",info2.count,info1.count);
    }
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      args: -ksp_rtol 1e-3 -ksp_converged_reason -pc_type {{bjacobi hpddm}shared output} -pc_hpddm_coarse_sub_pc_type lu -sub_pc_type lu -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

   testset:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_coarse_pc_type redundant -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
      test:
        suffix: geneo
        args: -pc_hpddm_coarse_p {{1 2}shared output} -pc_hpddm_levels_1_st_pc_type cholesky -pc_hpddm_levels_1_eps_nev {{5 15}separate output} -mat_type {{aij baij sbaij}shared output}
      test:
        suffix: geneo_block_splitting
        output_file: output/ex76_geneo_pc_hpddm_levels_1_eps_nev-15.out
        filter: sed -e "s/Linear solve converged due to CONVERGED_RTOL iterations 1[7-8]/Linear solve converged due to CONVERGED_RTOL iterations 11/g"
        args: -pc_hpddm_coarse_p 2 -pc_hpddm_levels_1_eps_nev 15 -pc_hpddm_block_splitting -pc_hpddm_levels_1_st_pc_type lu -pc_hpddm_levels_1_eps_gen_non_hermitian -mat_type {{aij baij}shared output}
      test:
        suffix: geneo_share
        output_file: output/ex76_geneo_pc_hpddm_levels_1_eps_nev-5.out
        args: -pc_hpddm_levels_1_st_pc_type cholesky -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_st_share_sub_ksp

   testset:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 4
      args: -ksp_converged_reason -ksp_max_it 150 -pc_type hpddm -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_coarse_p 1 -pc_hpddm_coarse_pc_type redundant -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -pc_hpddm_define_subdomains
      test:
        suffix: geneo_share_cholesky
        output_file: output/ex76_geneo_share.out
        # extra -pc_hpddm_levels_1_eps_gen_non_hermitian needed to avoid failures with PETSc Cholesky
        args: -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_st_pc_type cholesky -mat_type {{aij baij sbaij}shared output} -pc_hpddm_levels_1_eps_gen_non_hermitian -pc_hpddm_has_neumann -pc_hpddm_levels_1_st_share_sub_ksp {{false true}shared output}
      test:
        suffix: geneo_share_cholesky_matstructure
        output_file: output/ex76_geneo_share.out
        # extra -pc_hpddm_levels_1_eps_gen_non_hermitian needed to avoid failures with PETSc Cholesky
        args: -pc_hpddm_levels_1_sub_pc_type cholesky -mat_type {{baij sbaij}shared output} -pc_hpddm_levels_1_eps_gen_non_hermitian -pc_hpddm_levels_1_st_share_sub_ksp -pc_hpddm_levels_1_st_matstructure same -set_rhs {{false true} shared output}
      test:
        requires: mumps
        suffix: geneo_share_lu
        output_file: output/ex76_geneo_share.out
        # extra -pc_factor_mat_solver_type mumps needed to avoid failures with PETSc LU
        args: -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_st_pc_type lu -mat_type baij -pc_hpddm_levels_1_st_pc_factor_mat_solver_type mumps -pc_hpddm_levels_1_sub_pc_factor_mat_solver_type mumps -pc_hpddm_has_neumann -pc_hpddm_levels_1_st_share_sub_ksp {{false true}shared output}
      test:
        requires: mumps
        suffix: geneo_share_lu_matstructure
        output_file: output/ex76_geneo_share.out
        # extra -pc_factor_mat_solver_type mumps needed to avoid failures with PETSc LU
        args: -pc_hpddm_levels_1_sub_pc_type lu -mat_type baij -pc_hpddm_levels_1_sub_pc_factor_mat_solver_type mumps -pc_hpddm_levels_1_st_share_sub_ksp -pc_hpddm_levels_1_st_matstructure {{same different}shared output} -pc_hpddm_levels_1_st_pc_type lu -pc_hpddm_levels_1_st_pc_factor_mat_solver_type mumps

   test:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      suffix: fgmres_geneo_20_p_2
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -pc_hpddm_log_separate {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO

   testset:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      output_file: output/ex76_fgmres_geneo_20_p_2.out
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_2_eps_nev {{5 20}shared output} -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_type gmres -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
      test:
        suffix: fgmres_geneo_20_p_2_geneo
        args: -mat_type {{aij sbaij}shared output}
      test:
        suffix: fgmres_geneo_20_p_2_geneo_algebraic
        args: -pc_hpddm_levels_2_st_pc_type mat
   # PCHPDDM + KSPHPDDM test to exercise multilevel + multiple RHS in one go
   test:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      suffix: fgmres_geneo_20_p_2_geneo_rhs
      output_file: output/ex76_fgmres_geneo_20_p_2.out
      # for -pc_hpddm_coarse_correction additive
      filter: sed -e "s/Linear solve converged due to CONVERGED_RTOL iterations 37/Linear solve converged due to CONVERGED_RTOL iterations 25/g"
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type baij -pc_hpddm_levels_2_eps_nev 5 -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_max_it 10 -pc_hpddm_levels_2_ksp_type hpddm -pc_hpddm_levels_2_ksp_hpddm_type gmres -ksp_type hpddm -ksp_hpddm_variant flexible -pc_hpddm_coarse_mat_type baij -mat_type aij -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -rhs 4 -pc_hpddm_coarse_correction {{additive deflated balanced}shared output}

   testset:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES) mumps defined(PETSC_HAVE_OPENMP_SUPPORT)
      filter: egrep -e "Linear solve" -e "      executing" | sed -e "s/MPI =      1/MPI =      2/g" -e "s/OMP =      1/OMP =      2/g"
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 15 -pc_hpddm_levels_1_st_pc_type cholesky -pc_hpddm_coarse_p {{1 2}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -pc_hpddm_coarse_pc_factor_mat_solver_type mumps -pc_hpddm_coarse_mat_mumps_icntl_4 2 -pc_hpddm_coarse_mat_mumps_use_omp_threads {{1 2}shared output}
      test:
        suffix: geneo_mumps_use_omp_threads_1
        output_file: output/ex76_geneo_mumps_use_omp_threads.out
        args: -pc_hpddm_coarse_mat_type {{baij sbaij}shared output}
      test:
        suffix: geneo_mumps_use_omp_threads_2
        output_file: output/ex76_geneo_mumps_use_omp_threads.out
        args: -pc_hpddm_coarse_mat_type aij -pc_hpddm_levels_1_eps_threshold 0.3 -pc_hpddm_coarse_pc_type cholesky

   test:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      suffix: reuse_symbolic
      output_file: output/ex77_preonly.out
      nsize: 4
      args: -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -rhs 4 -pc_hpddm_coarse_correction {{additive deflated balanced}shared output} -ksp_pc_side {{left right}shared output} -ksp_max_it 20 -ksp_type hpddm -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -pc_hpddm_define_subdomains -ksp_error_if_not_converged

TEST*/
