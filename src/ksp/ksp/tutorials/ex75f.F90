!
!   Description: Solves a series of linear systems using KSPHPDDM.
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
      Vec                            x,b
      Mat                            A
#if defined(PETSC_HAVE_HPDDM)
      Mat                            U
#endif
      KSP                            ksp
      PetscInt                       i,j,nmat
      PetscViewer                    viewer
      character*(PETSC_MAX_PATH_LEN) dir,name
      character*(8)                  fmt
      character(3)                   cmat
      PetscBool                      flg,reset
      PetscErrorCode                 ierr

      PetscCallA(PetscInitialize(ierr))
      dir = '.'
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-load_dir',dir,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nmat',nmat,flg,ierr))
      reset = PETSC_FALSE
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-reset',reset,flg,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      do 50 i=0,nmat-1
        j = i+400
        fmt = '(I3)'
        write (cmat,fmt) j
        write (name,'(a)')trim(dir)//'/A_'//cmat//'.dat'
        PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr))
        PetscCallA(MatLoad(A,viewer,ierr))
        PetscCallA(PetscViewerDestroy(viewer,ierr))
        if (i .eq. 0) then
          PetscCallA(MatCreateVecs(A,x,b,ierr))
        endif
        write (name,'(a)')trim(dir)//'/rhs_'//cmat//'.dat'
        PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr))
        PetscCallA(VecLoad(b,viewer,ierr))
        PetscCallA(PetscViewerDestroy(viewer,ierr))
        PetscCallA(KSPSetFromOptions(ksp,ierr))
        PetscCallA(KSPSolve(ksp,b,x,ierr))
        PetscCallA(PetscObjectTypeCompare(ksp,KSPHPDDM,flg,ierr))
#if defined(PETSC_HAVE_HPDDM)
        if (flg .and. reset) then
          PetscCallA(KSPHPDDMGetDeflationMat(ksp,U,ierr))
          PetscCallA(KSPReset(ksp,ierr))
          PetscCallA(KSPSetOperators(ksp,A,A,ierr))
          PetscCallA(KSPSetFromOptions(ksp,ierr))
          PetscCallA(KSPSetUp(ksp,ierr))
          if (U .ne. PETSC_NULL_MAT) then
            PetscCallA(KSPHPDDMSetDeflationMat(ksp,U,ierr))
            PetscCallA(MatDestroy(U,ierr))
          endif
        endif
#endif
  50  continue
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      suffix: 1
!      output_file: output/ex75_1.out
!      nsize: 1
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      args: -nmat 1 -pc_type none -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!
!   test:
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      suffix: 1_icc
!      output_file: output/ex75_1_icc.out
!      nsize: 1
!      args: -nmat 1 -pc_type icc -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!
!   testset:
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      args: -nmat 3 -pc_type none -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 20 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!      test:
!        nsize: 1
!        suffix: 2_seq
!        output_file: output/ex75_2.out
!      test:
!        nsize: 2
!        suffix: 2_par
!        output_file: output/ex75_2.out
!
!   testset:
!      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!      output_file: output/ex75_2_icc.out
!      nsize: 1
!      args: -nmat 3 -pc_type icc -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!      test:
!        suffix: 2_icc
!        args:
!      test:
!        suffix: 2_icc_atol
!        output_file: output/ex75_2_icc_atol.out
!        args: -ksp_atol 1e-12
!
!TEST*/
