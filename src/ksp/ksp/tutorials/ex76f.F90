!
!   Description: Solves a linear systems using PCHPDDM.
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      use petscisdef
      implicit none
      Vec                            x,b
      Mat                            A,aux,Y,C
      KSP                            ksp
      PC                             pc
      IS                             is,sizes
      PetscScalar                    one
      PetscInt, pointer ::           idx(:)
      PetscMPIInt                    rank,size
      PetscInt                       m,N
      PetscViewer                    viewer
      character*(PETSC_MAX_PATH_LEN) dir,name
      character*(8)                  fmt
      character(1)                   crank,csize
      PetscBool                      flg
      PetscErrorCode                 ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print *,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 4) then
        print *,'This example requires 4 processes'
        stop
      endif
      N = 1
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-rhs',N,flg,ierr);CHKERRA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
      call MatCreate(PETSC_COMM_SELF,aux,ierr);CHKERRA(ierr)
      call ISCreate(PETSC_COMM_SELF,is,ierr);CHKERRA(ierr)
      dir = '.'
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-load_dir',dir,flg,ierr);CHKERRA(ierr)
      fmt = '(I1)'
      write (crank,fmt) rank
      write (csize,fmt) size
      write (name,'(a)')trim(dir)//'/sizes_'//crank//'_'//csize//'.dat'
      call PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ, viewer,ierr);CHKERRA(ierr)
      call ISCreate(PETSC_COMM_SELF,sizes,ierr);CHKERRA(ierr)
      call ISLoad(sizes,viewer,ierr);CHKERRA(ierr)
      call ISGetIndicesF90(sizes,idx,ierr);CHKERRA(ierr)
      call MatSetSizes(A,idx(1),idx(2),idx(3),idx(4),ierr);CHKERRA(ierr)
      call ISRestoreIndicesF90(sizes,idx,ierr);CHKERRA(ierr)
      call ISDestroy(sizes,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      call MatSetUp(A,ierr);CHKERRA(ierr)
      write (name,'(a)')trim(dir)//'/A.dat'
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
      call MatLoad(A,viewer,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      write (name,'(a)')trim(dir)//'/is_'//crank//'_'//csize//'.dat'
      call PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
      call ISLoad(is,viewer,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      write (name,'(a)')trim(dir)//'/Neumann_'//crank//'_'//csize//'.dat'
      call PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
      call MatSetBlockSizesFromMats(aux,A,A,ierr);CHKERRA(ierr)
      call MatLoad(aux,viewer,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      call MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE,ierr);CHKERRA(ierr)
      call MatSetOption(aux,MAT_SYMMETRIC,PETSC_TRUE,ierr);CHKERRA(ierr)
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);CHKERRA(ierr)
      call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
      call KSPGetPC(ksp,pc,ierr);CHKERRA(ierr)
      call PCSetType(pc,PCHPDDM,ierr);CHKERRA(ierr)
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
      call PCHPDDMSetAuxiliaryMat(pc,is,aux,PETSC_NULL_FUNCTION,PETSC_NULL_INTEGER,ierr);CHKERRA(ierr)
      call PCHPDDMHasNeumannMat(pc,PETSC_FALSE,ierr);CHKERRA(ierr)
#endif
      call ISDestroy(is,ierr);CHKERRA(ierr)
      call MatDestroy(aux,ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
      call MatCreateVecs(A,x,b,ierr);CHKERRA(ierr)
      one = 1.0
      call VecSet(b,one,ierr);CHKERRA(ierr)
      call KSPSolve(ksp,b,x,ierr);CHKERRA(ierr)
      call VecGetLocalSize(x,m,ierr);CHKERRA(ierr)
      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(b,ierr);CHKERRA(ierr)
      if (N .gt. 1) then
        call PetscOptionsClearValue(PETSC_NULL_OPTIONS,'-ksp_converged_reason',ierr);CHKERRA(ierr)
        call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
        call MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,PETSC_NULL_SCALAR,C,ierr);CHKERRA(ierr)
        call MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,PETSC_NULL_SCALAR,Y,ierr);CHKERRA(ierr)
        call MatSetRandom(C,PETSC_NULL_RANDOM,ierr);CHKERRA(ierr)
        call KSPMatSolve(ksp,C,Y,ierr);CHKERRA(ierr)
        call MatDestroy(Y,ierr);CHKERRA(ierr)
        call MatDestroy(C,ierr);CHKERRA(ierr)
      endif
      call KSPDestroy(ksp,ierr);CHKERRA(ierr)
      call MatDestroy(A,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_DYNAMIC_LIBRARIES) define(PETSC_USE_SHARED_LIBRARIES)
!      output_file: output/ex76_1.out
!      nsize: 4
!      args: -ksp_rtol 1e-3 -ksp_converged_reason -pc_type {{bjacobi hpddm}shared output} -pc_hpddm_coarse_sub_pc_type lu -sub_pc_type lu -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_DYNAMIC_LIBRARIES) define(PETSC_USE_SHARED_LIBRARIES)
!      suffix: geneo
!      output_file: output/ex76_geneo_pc_hpddm_levels_1_eps_nev-5.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_st_pc_type cholesky -pc_hpddm_coarse_p {{1 2}shared output} -pc_hpddm_coarse_pc_type redundant -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_DYNAMIC_LIBRARIES) define(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_DYNAMIC_LIBRARIES) define(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2_geneo
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_2_eps_nev {{5 20}shared output} -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_type gmres -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!   # PCHPDDM + KSPHPDDM test to exercise multilevel + multiple RHS in one go
!   test:
!      requires: hpddm slepc datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) define(PETSC_HAVE_DYNAMIC_LIBRARIES) define(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2_geneo_rhs
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type baij -pc_hpddm_levels_2_eps_nev 5 -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_max_it 10 -pc_hpddm_levels_2_ksp_type hpddm -pc_hpddm_levels_2_ksp_hpddm_type gmres -ksp_type hpddm -ksp_hpddm_variant flexible -pc_hpddm_coarse_mat_type baij -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -rhs 4
!
!TEST*/
