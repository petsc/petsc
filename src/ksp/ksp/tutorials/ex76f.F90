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

      PetscCallA(PetscInitialize(ierr))

      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      N = 1
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-rhs',N,flg,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatCreate(PETSC_COMM_SELF,aux,ierr))
      PetscCallA(ISCreate(PETSC_COMM_SELF,is,ierr))
      dir = '.'
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-load_dir',dir,flg,ierr))
      fmt = '(I1)'
      write (crank,fmt) rank
      write (csize,fmt) size
      write (name,'(a)')trim(dir)//'/sizes_'//crank//'_'//csize//'.dat'
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ, viewer,ierr))
      PetscCallA(ISCreate(PETSC_COMM_SELF,sizes,ierr))
      PetscCallA(ISLoad(sizes,viewer,ierr))
      PetscCallA(ISGetIndicesF90(sizes,idx,ierr))
      PetscCallA(MatSetSizes(A,idx(1),idx(2),idx(3),idx(4),ierr))
      PetscCallA(ISRestoreIndicesF90(sizes,idx,ierr))
      PetscCallA(ISDestroy(sizes,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      write (name,'(a)')trim(dir)//'/A.dat'
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,viewer,ierr))
      PetscCallA(MatLoad(A,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      write (name,'(a)')trim(dir)//'/is_'//crank//'_'//csize//'.dat'
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,viewer,ierr))
      PetscCallA(ISLoad(is,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      write (name,'(a)')trim(dir)//'/Neumann_'//crank//'_'//csize//'.dat'
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,viewer,ierr))
      PetscCallA(MatSetBlockSizesFromMats(aux,A,A,ierr))
      PetscCallA(MatLoad(aux,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE,ierr))
      PetscCallA(MatSetOption(aux,MAT_SYMMETRIC,PETSC_TRUE,ierr))
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCHPDDM,ierr))
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
      PetscCallA(PCHPDDMSetAuxiliaryMat(pc,is,aux,PETSC_NULL_FUNCTION,PETSC_NULL_INTEGER,ierr))
      PetscCallA(PCHPDDMHasNeumannMat(pc,PETSC_FALSE,ierr))
#endif
      PetscCallA(ISDestroy(is,ierr))
      PetscCallA(MatDestroy(aux,ierr))
      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(MatCreateVecs(A,x,b,ierr))
      one = 1.0
      PetscCallA(VecSet(b,one,ierr))
      PetscCallA(KSPSolve(ksp,b,x,ierr))
      PetscCallA(VecGetLocalSize(x,m,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      if (N .gt. 1) then
        PetscCallA(PetscOptionsClearValue(PETSC_NULL_OPTIONS,'-ksp_converged_reason',ierr))
        PetscCallA(KSPSetFromOptions(ksp,ierr))
        PetscCallA(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,PETSC_NULL_SCALAR,C,ierr))
        PetscCallA(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,PETSC_NULL_SCALAR,Y,ierr))
        PetscCallA(MatSetRandom(C,PETSC_NULL_RANDOM,ierr))
        PetscCallA(KSPMatSolve(ksp,C,Y,ierr))
        PetscCallA(MatDestroy(Y,ierr))
        PetscCallA(MatDestroy(C,ierr))
      endif
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      output_file: output/ex76_1.out
!      nsize: 4
!      args: -ksp_rtol 1e-3 -ksp_converged_reason -pc_type {{bjacobi hpddm}shared output} -pc_hpddm_coarse_sub_pc_type lu -sub_pc_type lu -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      suffix: geneo
!      output_file: output/ex76_geneo_pc_hpddm_levels_1_eps_nev-5.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_st_pc_type cholesky -pc_hpddm_coarse_p {{1 2}shared output} -pc_hpddm_coarse_pc_type redundant -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!
!   test:
!      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2_geneo
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_2_eps_nev {{5 20}shared output} -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_type gmres -ksp_type fgmres -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
!   # PCHPDDM + KSPHPDDM test to exercise multilevel + multiple RHS in one go
!   test:
!      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
!      suffix: fgmres_geneo_20_p_2_geneo_rhs
!      output_file: output/ex76_fgmres_geneo_20_p_2.out
!      nsize: 4
!      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_2_p 2 -pc_hpddm_levels_2_mat_type baij -pc_hpddm_levels_2_eps_nev 5 -pc_hpddm_levels_2_sub_pc_type cholesky -pc_hpddm_levels_2_ksp_max_it 10 -pc_hpddm_levels_2_ksp_type hpddm -pc_hpddm_levels_2_ksp_hpddm_type gmres -ksp_type hpddm -ksp_hpddm_variant flexible -pc_hpddm_coarse_mat_type baij -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO -rhs 4
!
!TEST*/
