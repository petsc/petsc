      subroutine samggetdimmat(k, nnu, nna)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get number of unknowns and nonzeros of coarse grid   ! 
! matrix on level k                                               !
! input:  k:        coarse grid level                             !
! output: nnu, nna: number of unknowns and nonzeros               !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
! look up the module where you find coarse grid ia array (and others) 
      use u_wkspace
! coarse grids on  a and ja array   (in amg_mod.f)
      use a_wkspace  
      implicit none 
      integer              imin(25),imax(25),iarr(25)
      integer              levels,m,isym,irow0
      common /samg_minx/   imin(25),imax(25),iarr(25)
      common /samg_data/   levels,m,isym,irow0
      integer              ierr, k, ilo, ihi, n1, n2, nnu, nna

!     check level input parameter 
      if (k.lt.2.or.k.gt.levels) then 
          write(*,*) 'ERROR in samggetdimmat: k out of range'
          write(*,*) 'Specified value for k on input = ',k
          write(*,*) 'The current number of levels = ',levels
          stop 'k should satisfy: 2 <= k <= levels!'
      endif  

!     determine number of unknowns 
      ilo = imin(k)       
      ihi = imax(k)   
      nnu = ihi-ilo+1
!     determine number of nonzeros 
      n1  = ia(ilo)       
      n2  = ia(ihi+1)-1
      nna = n2-n1+1 

      write(*,*) 'The current level k = ', k 
      write(*,*) 'The number of levels = ', levels 
      write(*,*) 'De dimensie van de matrix = ', nnu
      write(*,*) 'Het aantal nullen van de matrix = ', nna 
      
      return
      end subroutine samggetdimmat

      subroutine samggetmat(k, aout, iaout, jaout)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get coarse grid matrix on level k                    !
! input:        k:              coarse grid level                 !
! input/output: aut, ia, jaout: coarse grid matrix in skyline     ! 
!                               format                            !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! WARNING: This routine assumes memory allocation for aout, iaout, 
! and jaout OUTSIDE routine

! look up the module where you find coarse grid ia array (and others) 
      use u_wkspace
! coarse grids on  a and ja array   (in amg_mod.f)
      use a_wkspace  
      implicit none 
      integer               imin(25),imax(25),iarr(25)
      integer               levels,m,isym,irow0
      common /samg_minx/    imin(25),imax(25),iarr(25)
      common /samg_data/    levels,m,isym,irow0
      double precision      aout(*)
      integer               iaout(*), jaout(*) 
      integer               k, ilo, ihi, n1, n2, nnu, nna, i 

!     check level input parameter 
      if (k.lt.2.or.k.gt.levels) then 
          write(*,*) 'ERROR in samggetmat: k out of range'
          write(*,*) 'Specified value for k on input = ',k
          write(*,*) 'The current number of levels = ',levels
          stop 'k should satisfy: 2 <= k <= levels!'
      endif  

!     determine number of unknowns 
      ilo = imin(k)       
      ihi = imax(k)   
      nnu = ihi-ilo+1
!     determine number of nonzeros 
      n1  = ia(ilo)       
      n2  = ia(ihi+1)-1
      nna = n2-n1+1 

!     get matrix values 
      iaout(1:nnu+1)   = ia(ilo:ihi+1)
      aout(1:nna)      = a(n1:n2)
      jaout(1:nna)     = ja(n1:n2)  

      return 
      end subroutine samggetmat

      subroutine samggetdimint(k, nnu, nna)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine to get size and number of nonzeros of interpolation operator  ! 
! from grid k+1 (coarse grid) to k (finer grid).                        !
! input:  k:           fine grid level                                  ! 
! output: nnu and nna: size and number of nonzeros of interpolation     ! 
!                      operator                                         ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      use u_wkspace   ! look up the module where you find coarse grid 
                      ! iw array (and others) 
      use w_wkspace   ! w and jw array   (in amg_mod.f) 
                      ! (interpolation weights and couplings)
      implicit none 
      integer              imin(25),imax(25),iarr(25)
      integer              levels,m,isym,irow0
      common /samg_minx/   imin(25),imax(25),iarr(25)
      common /samg_data/   levels,m,isym,irow0
      integer              k, nnu, nna, ilo, ihi, n1, n2 

!     check level input parameter 
      if (k.lt.1.or.k.gt.levels-1) then 
          write(*,*) 'ERROR in samggetdimint: k out of range'
          write(*,*) 'Specified value for k on input = ',k
          write(*,*) 'The current number of levels = ',levels
          stop 'k should satisfy: 1 <= k <= levels-1!'
      endif  

!     determine size  
      ilo = imin(k)       
      ihi = imax(k)   
      nnu = ihi-ilo+1
!     determine number of nonzeros 
      n1  = iw(ilo)       
      n2  = iw(ihi+1)-1
      nna = n2-n1+1 

      write(*,*) 'The current level k = ', k 
      write(*,*) 'The number of levels = ', levels 
      write(*,*) 'De dimensie van de interp. op. k -> k+1= ', nnu
      write(*,*) 'Het aantal nullen van de interp. op = ', nna 
      
      return
      end subroutine samggetdimint

      subroutine samggetint(k, wout, iwout, jwout)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get interpolation operator from grid k+1 (coarse grid)     !
! to k (finer grid).                                                    !
! input:  k:                  fine grid level                           ! 
! output: wout, iwout, jwout: interpolation in skyline format           ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! WARNING: This routine assumes memory allocation for wout, iwout, 
! and jwout OUTSIDE routine

      use u_wkspace   ! look up the module where you find coarse grid 
                      ! iw array (and others) 
      use w_wkspace   ! w and jw array   (in amg_mod.f) 
                      ! (interpolation weights and couplings)
      implicit none 
      integer              imin(25),imax(25),iarr(25)
      integer              levels,m,isym,irow0
      common /samg_minx/   imin(25),imax(25),iarr(25)
      common /samg_data/   levels,m,isym,irow0
      double precision     wout(*)
      integer              iwout(*), jwout(*) 
      integer              k, ilo, ihi, n1, n2, nnu, nna 

!     determine number of unknowns 
      ilo = imin(k)       
      ihi = imax(k)   
      nnu = ihi-ilo+1
!     determine number of nonzeros 
      n1  = iw(ilo)       
      n2  = iw(ihi+1)-1
      nna = n2-n1+1 

!     get interpolation values 
      iwout(1:nnu+1)   = iw(ilo:ihi+1)
      wout(1:nna)      = w(n1:n2)
      jwout(1:nna)     = jw(n1:n2)  

      end subroutine samggetint
