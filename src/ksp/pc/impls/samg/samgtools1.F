      subroutine samgpetsc_apply_shift(ia, nnu, ia_shift, ja, nna,       &
     &                                 ja_shift)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This routine applies a ia_shift to all elements in the ia array,! 
! and a ja_shift to all elements in the ja array.                 ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none 
      integer ia(*), nnu, ia_shift, ja(*), nna, ja_shift
      integer i 

      do i=1,nnu+1 
         ia(i) = ia(i) - ia_shift 
      enddo

      do i=1,nna
         ja(i) = ja(i) - ja_shift          
      enddo 

      end subroutine samgpetsc_apply_shift

      subroutine samgpetsc_get_levels(levelscp)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine to get the numbers of levels created by SAMG            ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     
      implicit none 
    
      integer levelscp, levels 

      common /samg_mlev/ levels

      levelscp = levels 

      return 
      end subroutine samgpetsc_get_levels

      subroutine samgpetsc_get_dim_operator(k, nnu, nna)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get number of unknowns and nonzeros of coarse grid   ! 
! matrix on level k                                               !
! input:  k:        coarse grid level                             !
! output: nnu, nna: number of unknowns and nonzeros               !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! The number of unknowns is determined from the imin and imax arrays. 
! These arrays are stored in common blocks 
! The number of nonzeros is determined from the ia array. 
! This array is stored in the u_wkspace work array. 
! For information on this latter array, see module samg_wkspace_status
! in amg_mod.f 
  
      use u_wkspace

      implicit none 
      integer              imin(25),imax(25),ipmin(25),ipmax(25)
      integer              levels
      common /samg_minx/   imin(25),imax(25),ipmin(25),ipmax(25)
      common /samg_mlev/   levels
      integer              k, ilo, ihi, n1, n2, nnu, nna

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
      end subroutine samgpetsc_get_dim_operator 

      subroutine samgpetsc_get_operator(k, aout, iaout, jaout)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get coarse grid matrix on level k                    !
! input:        k:              coarse grid level                 !
! input/output: aut, ia, jaout: coarse grid matrix in skyline     ! 
!                               format                            !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! WARNING: This routine assumes memory allocation for aout, iaout, 
! and jaout OUTSIDE routine

! The coarse grid matrices are stored in (a, ia, ja). The array ia is 
! stored in the u_wkspace work array, while the arrays a and ja are stored 
! in the a_wkspace work array. See module samg_wkspace_status
! in amg_mod.f for details. 

      use u_wkspace
      use a_wkspace  

      implicit none 
      integer               imin(25),imax(25),ipmin(25),ipmax(25)
      integer               levels
      common /samg_minx/    imin(25),imax(25),ipmin(25),ipmax(25)
      common /samg_mlev/    levels
      double precision      aout(*)
      integer               iaout(*), jaout(*) 
      integer               k, ilo, ihi, n1, n2, nnu, nna 

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
      end subroutine samgpetsc_get_operator 

      subroutine samgpetsc_get_dim_interpol(k, nnu, nna)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine to get size and number of nonzeros of interpolation operator  ! 
! from grid k+1 (coarse grid) to k (finer grid).                        !
! input:  k:           fine grid level                                  ! 
! output: nnu and nna: size and number of nonzeros of interpolation     ! 
!                      operator                                         ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! The the size is determined from the imin and imax arrays. 
! These arrays are stored in common blocks 
! The number of nonzeros is determined from the iw array. 
! This array is stored in the u_wkspace work array. 
! For information on this latter array, see module samg_wkspace_status
! in amg_mod.f 

      use u_wkspace   

      implicit none 
      integer              imin(25),imax(25),ipmin(25),ipmax(25)
      integer              levels
      common /samg_minx/   imin(25),imax(25),ipmin(25),ipmax(25)
      common /samg_mlev/   levels
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
      end subroutine samgpetsc_get_dim_interpol

      subroutine samgpetsc_get_interpol(k, wout, iwout, jwout)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Routine do get interpolation operator from grid k+1 (coarse grid)     !
! to k (finer grid).                                                    !
! input:  k:                  fine grid level                           ! 
! output: wout, iwout, jwout: interpolation in skyline format           ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! WARNING: This routine assumes memory allocation for wout, iwout, 
! and jwout OUTSIDE routine

! The interpolation opreators are stored in (w, iw, jw). The array iw is 
! stored in the u_wkspace work array, while the arrays w and jw are stored 
! in the w_wkspace work array. See module samg_wkspace_status
! in amg_mod.f for details. 

      use u_wkspace   
      use w_wkspace   

      implicit none 
      integer              imin(25),imax(25),ipmin(25),ipmax(25)
      integer              levels
      common /samg_minx/   imin(25),imax(25),ipmin(25),ipmax(25)
      common /samg_mlev/   levels
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

      end subroutine samgpetsc_get_interpol
