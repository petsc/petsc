!
!   Description: Solves a series of linear systems using KSPHPDDM.
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
      Vec             x,b
      Mat             A
#if defined(PETSC_HAVE_HPDDM)
      Mat             U
#endif
      KSP             ksp
      PetscInt        i,j,nmat
      PetscViewer     viewer
      character*(128) name
      character*(128) dir
      character*(8)   fmt
      character(3)    cmat
      PetscBool       flg,reset
      PetscErrorCode  ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print *,'Unable to initialize PETSc'
        stop
      endif
      dir = '.'
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, &
     &     '-load_dir',dir,flg,ierr);CHKERRA(ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,    &
     &     '-nmat',nmat,flg,ierr);CHKERRA(ierr)
      reset = PETSC_FALSE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,   &
     &     '-reset',reset,flg,ierr);CHKERRA(ierr)
      call MatCreate(PETSC_COMM_WORLD,A,ierr);                            &
     &    CHKERRA(ierr)
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);                          &
     &    CHKERRA(ierr)
      call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
      do 50 i=0,nmat-1
        j = i+400
        fmt = '(I3)'
        write (cmat,fmt) j
        write (name,'(a)')trim(dir)//'/A_'//cmat//'.dat'
        call PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ, &
     &       viewer,ierr);CHKERRA(ierr)
        call MatLoad(A,viewer,ierr);CHKERRA(ierr)
        call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
        if (i .eq. 0) then
          call MatCreateVecs(A,x,b,ierr);CHKERRA(ierr)
        endif
        write (name,'(a)')trim(dir)//'/rhs_'//cmat//'.dat'
        call PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ, &
     &       viewer,ierr);CHKERRA(ierr)
        call VecLoad(b,viewer,ierr);CHKERRA(ierr)
        call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
        call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
        call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
        call KSPSolve(ksp,b,x,ierr);CHKERRA(ierr)
#if defined(PETSC_HAVE_HPDDM)
        call PetscObjectTypeCompare(ksp,KSPHPDDM,flg,ierr);              &
             CHKERRA(ierr)
        if (flg .and. reset) then
          call KSPHPDDMGetDeflationSpace(ksp,U,ierr);                    &
               CHKERRA(ierr)
          call KSPReset(ksp,ierr);CHKERRA(ierr)
          call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
          call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
          call KSPSetUp(ksp,ierr);CHKERRA(ierr)
          if (U .ne. PETSC_NULL_MAT) then
            call KSPHPDDMSetDeflationSpace(ksp,U,ierr);                  &
                 CHKERRA(ierr)
            call MatDestroy(U,ierr);CHKERRA(ierr)
          endif
        endif
#endif
  50  continue
      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(b,ierr);CHKERRA(ierr)
      call MatDestroy(A,ierr);CHKERRA(ierr)
      call KSPDestroy(ksp,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      suffix: 1
!      output_file: output/ex75_1.out
!      nsize: 1
!      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
!      args: -nmat 1 -pc_type none -ksp_converged_reason -ksp_type {{gmres hpddm}shared ouput} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!
!   test:
!      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
!      suffix: 1_icc
!      output_file: output/ex75_1_icc.out
!      nsize: 1
!      args: -nmat 1 -pc_type icc -ksp_converged_reason -ksp_type {{gmres hpddm}shared ouput} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!
!   testset:
!      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
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
!   test:
!      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
!      suffix: 2_icc
!      output_file: output/ex75_2_icc.out
!      nsize: 1
!      args: -nmat 3 -pc_type icc -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!
!TEST*/
