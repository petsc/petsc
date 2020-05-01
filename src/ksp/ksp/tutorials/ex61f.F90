!
!        Demonstrates having each OpenMP thread manage its own PETSc objects and solves
!           - each thread is ONLY allowed to access objects that IT created
!                  that is, threads cannot shared objects
!
!        Run with "export OMP_NUM_THREADS=16 ./ex61f"
!           to use 16 independent threads
!
!        ./configure --with-threadsafety --with-log=0 --with-openmp
!
         module omp_module
         implicit none
         contains
         subroutine split_indices(total,num_pieces,ibeg,iend)
           implicit none

           integer :: total
           integer :: num_pieces
           integer :: ibeg(num_pieces), iend(num_pieces)
           integer :: itmp1, itmp2, ioffset, i

           itmp1 = total/num_pieces
           itmp2 = mod(total,num_pieces)
           ioffset = 0
           do i=1,itmp2
              ibeg(i) = ioffset + 1
              iend(i) = ioffset + (itmp1+1)
              ioffset = iend(i)
           enddo
           do i=itmp2+1,num_pieces
              ibeg(i) = ioffset + 1
              if (ibeg(i) > total) then
                 iend(i) = ibeg(i) - 1
              else
                 iend(i) = ioffset + itmp1
                 ioffset = iend(i)
              endif
           enddo

         end subroutine split_indices
       end module omp_module

      module assert_mod
      implicit none
      contains
      subroutine assert(lcond,msg,icase)
      logical,intent(in) :: lcond
      character(len=*), intent(in) :: msg
      integer, intent(in) :: icase

      if (.not.lcond) then
         write(*,*) msg, icase
         stop 'assertion error '
      endif
      return
      end subroutine assert
      end module assert_mod

      program tpetsc

#include <petsc/finclude/petsc.h>
      use assert_mod
      use omp_module
      use petsc
      implicit none
!     ----------------------------
!     test concurrent petsc solver
!     ----------------------------

      integer, parameter :: NCASES=100
      integer, parameter :: MAXTHREADS=64
      real(8), parameter :: tol = 1.0d-6

      integer, dimension(MAXTHREADS) :: ibeg,iend

!$   integer, external :: omp_get_num_threads

      Mat, target :: Mcol_f_mat(MAXTHREADS)
      Vec, target :: Mcol_f_vecb(MAXTHREADS)
      Vec, target :: Mcol_f_vecx(MAXTHREADS)
      KSP, target :: Mcol_f_ksp(MAXTHREADS)
      PC,  target :: MPC(MAXTHREADS)

      Mat, pointer :: col_f_mat
      Vec, pointer :: col_f_vecb
      Vec, pointer :: col_f_vecx
      KSP, pointer :: col_f_ksp
      PC, pointer :: pc

      PetscInt :: ith, nthreads 
      PetscErrorCode ierr

      integer, parameter :: nz_per_row = 9
      integer, parameter :: n =100 
      integer :: i,j,ij,ij2,ii,jj,nz,ip, dx,dy,icase
      integer, allocatable :: ilist(:),jlist(:)
      PetscScalar :: aij
      PetscScalar, allocatable :: alist(:)
      logical :: isvalid_ii, isvalid_jj, is_diag

      PetscInt nrow
      PetscInt ncol
      PetscScalar, allocatable :: x(:), b(:)
      real(8) :: err(NCASES)

      integer :: t1,t2,count_rate
      real :: ttime

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      allocate(ilist(n*n*nz_per_row),jlist(n*n*nz_per_row),alist(n*n*nz_per_row))
      allocate(x(0:(n*n-1)),b(0:(n*n-1)))
      nrow = n*n
      ncol = nrow

      nthreads = 1
!$omp parallel
!$omp master
!$      nthreads = omp_get_num_threads()
!$omp end master
!$omp end parallel
      print*,'nthreads = ', nthreads,' NCASES = ', NCASES

      call split_indices(NCASES,nthreads,ibeg,iend)


!$omp parallel do                                                        &
!$omp private(ith,ierr)                                                  &
!$omp private(col_f_mat,col_f_vecb,col_f_vecx,col_f_ksp)
      do ith=1,nthreads
         col_f_mat => Mcol_f_mat(ith)
         col_f_vecb => Mcol_f_vecb(ith)
         col_f_vecx => Mcol_f_vecx(ith)
         col_f_ksp => Mcol_f_ksp(ith)


         call MatCreateSeqAIJ( PETSC_COMM_SELF, nrow,ncol, nz_per_row,PETSC_NULL_INTEGER, col_f_mat, ierr)
         call assert(ierr.eq.0,'matcreateseqaij return ',ierr)

         call VecCreateSeqWithArray(PETSC_COMM_SELF,1,nrow,PETSC_NULL_SCALAR, col_f_vecb, ierr)
         call assert(ierr.eq.0,'veccreateseqwitharray return ierr',ierr)

         call VecDuplicate(col_f_vecb, col_f_vecx,ierr)
         call assert(ierr.eq.0,'vecduplicate return ierr',ierr)

         call KSPCreate(PETSC_COMM_SELF, col_f_ksp,ierr)
         call assert(ierr.eq.0,'kspcreate return ierr',ierr)

       enddo

!      -----------------------
!      setup sparsity pattern
!      -----------------------
       nz = 0
       do j=1,n
       do i=1,n
        ij = i + (j-1)*n
        do dx=-1,1
        do dy=-1,1
          ii = i + dx
          jj = j + dy

          ij2 = ii + (jj-1)*n
          isvalid_ii = (1 <= ii).and.(ii <= n)
          isvalid_jj = (1 <= jj).and.(jj <= n)
          if (isvalid_ii.and.isvalid_jj) then
             is_diag = (ij .eq. ij2)
             nz = nz + 1
             ilist(nz) = ij
             jlist(nz) = ij2
             if (is_diag) then
               aij = 4.0
             else
               aij = -1.0
             endif
             alist(nz) = aij
           endif
          enddo
          enddo
         enddo
         enddo

       print*,'nz = ', nz

!      ---------------------------------
!      convert from fortan to c indexing
!      ---------------------------------
       ilist(1:nz) = ilist(1:nz) - 1
       jlist(1:nz) = jlist(1:nz) - 1


!      --------------
!      setup matrices
!      --------------
       call system_clock(t1,count_rate)

!$omp  parallel do                                                      &
!$omp& private(ith,icase,ip,i,j,ii,jj,aij,ierr,x,b)                      &
!$omp& private(col_f_mat,col_f_vecb,col_f_vecx,col_f_ksp,pc)
       do ith=1,nthreads
         col_f_mat => Mcol_f_mat(ith)
         col_f_vecb => Mcol_f_vecb(ith)
         col_f_vecx => Mcol_f_vecx(ith)
         col_f_ksp => Mcol_f_ksp(ith)
         pc => MPC(ith)

        do icase=ibeg(ith),iend(ith)

         do ip=1,nz
           ii = ilist(ip)
           jj = jlist(ip)
           aij = alist(ip)
           call MatSetValue(col_f_mat,ii,jj,aij,INSERT_VALUES,ierr)
           call assert(ierr.eq.0,'matsetvalue return ierr',ierr)
         enddo
         call MatAssemblyBegin(col_f_mat,MAT_FINAL_ASSEMBLY,ierr)
         call assert(ierr.eq.0,'matassemblybegin return ierr',ierr)

         call MatAssemblyEnd(col_f_mat,MAT_FINAL_ASSEMBLY,ierr)
         call assert(ierr.eq.0,'matassemblyend return ierr',ierr)

         call KSPSetOperators(col_f_ksp,col_f_mat,col_f_mat,ierr)
         call assert(ierr.eq.0,'KSPSetOperators return ierr',ierr)

         ! set linear solver
         call KSPGetPC(col_f_ksp,pc,ierr)
         call assert(ierr.eq.0,'KSPGetPC return ierr ', ierr)

         call PCSetType(pc,PCLU,ierr)
         call assert(ierr.eq.0,'PCSetType return ierr ',ierr)

         ! associate petsc vector with allocated array
         x(:) = 1
!$omp    critical
         call VecPlaceArray(col_f_vecx,x,ierr)
!$omp    end critical
         call assert(ierr.eq.0,'VecPlaceArray col_f_vecx return ',ierr)

         b(:) = 0
         do ip=1,nz
           i = ilist(ip)
           j = jlist(ip)
           aij = alist(ip)
           b(i) = b(i) + aij*x(j)
         enddo
         x = 0
!$omp    critical
         call VecPlaceArray(col_f_vecb,b,ierr)
!$omp    end critical
         call assert(ierr.eq.0,'VecPlaceArray col_f_vecb return ',ierr)



!  -----------------------------------------------------------
!  key test, need to solve multiple linear systems in parallel
!  -----------------------------------------------------------
         call KSPSetFromOptions(col_f_ksp,ierr)
         call assert(ierr.eq.0,'KSPSetFromOptions return ierr ',ierr)

         call KSPSetUp(col_f_ksp,ierr)
         call assert(ierr.eq.0,'KSPSetup return ierr ',ierr)


         call KSPSolve(col_f_ksp,col_f_vecb,col_f_vecx,ierr)
         call assert(ierr.eq.0,'KSPSolve return ierr ',ierr)


!        ------------
!        check answer
!        ------------
         err(icase) = maxval(abs(x(:)-1))

!$omp    critical
         call VecResetArray(col_f_vecx,ierr)
!$omp    end critical
         call assert(ierr.eq.0,'VecResetArray return ierr ',ierr)

!$omp    critical
         call VecResetArray(col_f_vecb,ierr)
!$omp    end critical
         call assert(ierr.eq.0,'VecResetArray return ierr ',ierr)

       enddo
       enddo

!$omp parallel do                                                        &
!$omp private(ith,ierr)                                                  &
!$omp private(col_f_mat,col_f_vecb,col_f_vecx,col_f_ksp)
      do ith=1,nthreads
         col_f_mat => Mcol_f_mat(ith)
         col_f_vecb => Mcol_f_vecb(ith)
         col_f_vecx => Mcol_f_vecx(ith)
         col_f_ksp => Mcol_f_ksp(ith)


         call MatDestroy(col_f_mat, ierr)
         call assert(ierr.eq.0,'matdestroy return ',ierr)

         call VecDestroy(col_f_vecb, ierr)
         call assert(ierr.eq.0,'vecdestroy return ierr',ierr)

         call VecDestroy(col_f_vecx,ierr)
         call assert(ierr.eq.0,'vecdestroy return ierr',ierr)

         call KSPDestroy(col_f_ksp,ierr)
         call assert(ierr.eq.0,'kspdestroy return ierr',ierr)

       enddo

       call system_clock(t2,count_rate)
       ttime = real(t2-t1)/real(count_rate)
       write(*,*) 'total time is ',ttime

       write(*,*) 'maxval(err) ', maxval(err)
       do icase=1,NCASES
        if (err(icase) > tol) then
         write(*,*) 'icase,err(icase) ',icase,err(icase)
        endif
       enddo

       deallocate(ilist,jlist,alist)
       deallocate(x,b)
       call PetscFinalize(ierr)
       call assert(ierr.eq.0,'petscfinalize return ierr',ierr)

       end program tpetsc

!/*TEST
!
!   build:
!      requires: double !complex !define(PETSC_USE_64BIT_INDICES)
!
!   test:
!      output_file: output/ex61f_1.out
!      TODO: Need to determine how to test OpenMP code
!
!TEST*/
