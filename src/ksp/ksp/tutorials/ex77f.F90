!
!   Description: Solves a linear system with a block of right-hand sides using KSPHPDDM.
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
      Mat             X,B
      Vec             cx,cb
      Mat             A
      KSP             ksp
      PetscScalar, pointer :: x_v(:),b_v(:)
      PetscInt        m,n,L,K
      PetscViewer     viewer
      character*(128) dir,name
      PetscBool       flg
      PetscErrorCode  ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print *,'Unable to initialize PETSc'
        stop
      endif
      dir = '.'
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-load_dir',dir,flg,ierr);CHKERRA(ierr)
      K = 5
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',K,flg,ierr);CHKERRA(ierr)
      call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);CHKERRA(ierr)
      call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
      write (name,'(a)')trim(dir)//'/A_400.dat'
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
      call MatLoad(A,viewer,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      call MatGetLocalSize(A,m,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,K,PETSC_NULL_SCALAR,B,ierr);CHKERRA(ierr)
      call MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,K,PETSC_NULL_SCALAR,X,ierr);CHKERRA(ierr)
      call MatSetRandom(B,PETSC_NULL_RANDOM,ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
      call KSPSetUp(ksp,ierr);CHKERRA(ierr)
      call PetscObjectTypeCompare(ksp,KSPHPDDM,flg,ierr);CHKERRA(ierr)
#if defined(PETSC_HAVE_HPDDM)
      if (flg) then
        call KSPHPDDMMatSolve(ksp,B,X,ierr);CHKERRA(ierr)
      else
#endif
        call MatGetSize(A,L,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
        do 50 n=0,K-1
          call MatDenseGetColumnF90(B,n,b_v,ierr);CHKERRA(ierr)
          call MatDenseGetColumnF90(X,n,x_v,ierr);CHKERRA(ierr)
          call VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,L,b_v,cb,ierr);CHKERRA(ierr)
          call VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,L,x_v,cx,ierr);CHKERRA(ierr)
          call KSPSolve(ksp,cb,cx,ierr);CHKERRA(ierr)
          call VecDestroy(cx,ierr);CHKERRA(ierr)
          call VecDestroy(cb,ierr);CHKERRA(ierr)
          call MatDenseRestoreColumnF90(X,x_v,ierr);CHKERRA(ierr)
          call MatDenseRestoreColumnF90(B,b_v,ierr);CHKERRA(ierr)
  50    continue
#if defined(PETSC_HAVE_HPDDM)
      endif
#endif
      call MatDestroy(X,ierr);CHKERRA(ierr)
      call MatDestroy(B,ierr);CHKERRA(ierr)
      call MatDestroy(A,ierr);CHKERRA(ierr)
      call KSPDestroy(ksp,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   testset:
!      nsize: 2
!      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
!      args: -ksp_converged_reason -ksp_max_it 1000 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
!      test:
!         suffix: 1
!         output_file: output/ex77_1.out
!         args:
!      test:
!         suffix: 2a
!         output_file: output/ex77_2_ksp_hpddm_type-gmres.out
!         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type gmres
!      test:
!         suffix: 2b
!         output_file: output/ex77_2_ksp_hpddm_type-bgmres.out
!         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type bgmres
!      test:
!         suffix: 3a
!         output_file: output/ex77_3_ksp_hpddm_type-gcrodr.out
!         args: -ksp_type hpddm -ksp_hpddm_recycle 10 -ksp_hpddm_type gcrodr
!      test:
!         suffix: 3b
!         output_file: output/ex77_3_ksp_hpddm_type-bgcrodr.out
!         args: -ksp_type hpddm -ksp_hpddm_recycle 10 -ksp_hpddm_type bgcrodr
!
!TEST*/
