!
!   Description: Solves a linear system with a block of right-hand sides using KSPHPDDM.
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
      Mat                            X,B
      Mat                            A
      KSP                            ksp
      PC                             pc
      Mat                            F
      PetscScalar                    alpha
      PetscReal                      norm
      PetscInt                       m,K
      PetscViewer                    viewer
      character*(PETSC_MAX_PATH_LEN) name
      PetscBool                      flg
      PetscErrorCode                 ierr

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',name,flg,ierr))
      if (flg .eqv. PETSC_FALSE) then
        SETERRA(PETSC_COMM_WORLD,PETSC_ERR_SUP,'Must provide a binary file for the matrix')
      endif
      K = 5
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',K,flg,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr))
      PetscCallA(MatLoad(A,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(MatGetLocalSize(A,m,PETSC_NULL_INTEGER,ierr))
      PetscCallA(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,K,PETSC_NULL_SCALAR,B,ierr))
      PetscCallA(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,K,PETSC_NULL_SCALAR,X,ierr))
      PetscCallA(MatSetRandom(B,PETSC_NULL_RANDOM,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSetUp(ksp,ierr))
      PetscCallA(KSPMatSolve(ksp,B,X,ierr))
      PetscCallA(KSPGetMatSolveBatchSize(ksp,M,ierr))
      if (M .ne. PETSC_DECIDE) then
        PetscCallA(KSPSetMatSolveBatchSize(ksp,PETSC_DECIDE,ierr))
        PetscCallA(MatZeroEntries(X,ierr))
        PetscCallA(KSPMatSolve(ksp,B,X,ierr))
      endif
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PetscObjectTypeCompare(pc,PCLU,flg,ierr))
      if (flg) then
        PetscCallA(PCFactorGetMatrix(pc,F,ierr))
        PetscCallA(MatMatSolve(F,B,B,ierr))
        alpha = -1.0
        PetscCallA(MatAYPX(B,alpha,X,SAME_NONZERO_PATTERN,ierr))
        PetscCallA(MatNorm(B,NORM_INFINITY,norm,ierr))
        if (norm > 100*PETSC_MACHINE_EPSILON) then
          SETERRA(PETSC_COMM_WORLD,PETSC_ERR_PLIB,'KSPMatSolve() and MatMatSolve() difference has nonzero norm')
        endif
      endif
      PetscCallA(MatDestroy(X,ierr))
      PetscCallA(MatDestroy(B,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   testset:
!      nsize: 2
!      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      args: -ksp_converged_reason -ksp_max_it 1000 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat
!      test:
!         suffix: 1
!         output_file: output/ex77_1.out
!         args:
!      test:
!         suffix: 2a
!         requires: hpddm
!         output_file: output/ex77_2_ksp_hpddm_type-gmres.out
!         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type gmres
!      test:
!         suffix: 2b
!         requires: hpddm
!         output_file: output/ex77_2_ksp_hpddm_type-bgmres.out
!         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type bgmres
!      test:
!         suffix: 3a
!         requires: hpddm
!         output_file: output/ex77_3_ksp_hpddm_type-gcrodr.out
!         args: -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type gcrodr
!      test:
!         suffix: 3b
!         requires: hpddm
!         output_file: output/ex77_3_ksp_hpddm_type-bgcrodr.out
!         args: -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr
!      test:
!         nsize: 4
!         suffix: 4
!         requires: hpddm
!         output_file: output/ex77_4.out
!         args: -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5
!   test:
!      nsize: 1
!      suffix: preonly
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      output_file: output/ex77_preonly.out
!      args: -N 6 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -pc_type lu -ksp_type hpddm -ksp_hpddm_type preonly
!   test:
!      nsize: 4
!      suffix: 4_slepc
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      output_file: output/ex77_4.out
!      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
!      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type dense -ksp_hpddm_recycle_eps_converged_reason -ksp_hpddm_recycle_st_pc_type redundant
!
!TEST*/
