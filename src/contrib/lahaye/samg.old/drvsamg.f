      subroutine drvsamg(nnu, a, ia, ja, u, f, matrix, nsolve, ifirst, 
     *                   eps, ncyc, iswtch, a_cmplx, g_cmplx, p_cmplx, 
     *                   w_avrge, chktol, idump, iout, nrd, 
     *                   nrc, nru, levelx, nptmn, ncg, nwt, ntr, ecg, 
     *                   ewt, etr, levelscp, debug)

      use samg_input_class23      
      implicit none
      double precision, dimension(*) :: a,u,f
      integer,          dimension(*) :: ia,ja
      integer,          dimension(:), allocatable :: iu,ip,iscale
!     ..Class 0 Integer High level control parameters.. 
!     ..ia, ja, iu, ip, iscale declared in drv_mod.f ..
      integer           nnu, nsys, npnts, 
     *                  matrix, ncyc_done, ierr
      parameter    (nsys = 1)
!     ..Class 0 Real High level control parameters.. 
       double precision res_in, res_out
!     ..Class 1 AMG Integer Parameters .. 
      integer           nsolve, ifirst, ncyc, iswtch
!     ..Class 1 AMG Real Parameters.. 
       double precision eps
!     ..Class 1 Dimensioning Parameters.. 
       double precision a_cmplx, g_cmplx, p_cmplx, w_avrge 
!     ..Class 1 Integer Parameters controlling Input/Output 
      integer           idump, iout 
!     ..Class 1 Real    Parameters controlling Input/Output 
       double precision chktol 
!     ..Class 2 Integer Parameters.. 
!     ..Solution Phase of samg..
      integer           nrd, nrc, nru
!     ..Class 3 Integer Parameters.. 
!     ..General parameters for the preparation of SAMG.. 
      integer           levelx, nptmn
!     ..Class 3 Integer and Real Parameters.. 
!     ..Special parameters for the preparation of SAMG.. 
      integer           ncg, nwt, ntr
      double precision  ecg, ewt, etr

!     ..Allow additional information on AMG to flow to calling program.. 
      integer           levelscp 
      integer           imin(25),imax(25),iarr(25)
      integer           levels,m,isym,irow0
      common /samg_minx/ imin(25),imax(25),iarr(25)
      common /samg_data/ levels,m,isym,irow0
      integer, dimension(:), pointer :: iaout,jaout
      double precision, dimension(:), pointer :: aout

!     ..Print intermediate results if specified..  
      integer           debug 

!     define class2/3 parameters through interface module "samg_input_class23"

      nrd_samg   = nrd
      nrc_samg   = nrc
      nru_samg   = nru
      levelx_samg= levelx
      nptmn_samg = nptmn
      ncg_samg   = ncg
      nwt_samg   = nwt
      ntr_samg   = ntr
      ecg_samg   = ecg
      ewt_samg   = ewt
      etr_samg   = etr

!     Allocate iu, ip, iscale 
      allocate(iu(1:nnu),stat=ierr)
      allocate(ip(1:nnu),stat=ierr)
      allocate(iscale(1:nsys),stat=ierr)

         call samg(nnu,nsys,npnts,ia,ja,a,f,u,iu,ip,matrix,iscale, 
     *             res_in,res_out,ncyc_done,ierr,nsolve,ifirst,eps, 
     *             ncyc,iswtch,a_cmplx,g_cmplx,p_cmplx,w_avrge,chktol, 
     *             idump,iout)

!      write(*,*) 'Writing AMG solution'
!      open(12, file = 'tmp_amg.txt')
!      do i=1,nnu/2
!         write(12,*) i, u(2*i-1), u(2*i)
!      enddo 
!      close(12)

!         call fetch(2, aout, iaout, jaout)         

!     Allow number of levels to flow to C++ calling program 
      levelscp = levels
!     Deallocate iu, ip, iscale 
      deallocate(iu, ip, iscale)

      return
      end

      subroutine fetch(k, aout, iaout, jaout)  
! look up the module where you find coarse grid ia array (and others) 
      use u_wkspace
! coarse grids on  a and ja array   (in amg_mod.f)
      use a_wkspace  
      implicit none 
      integer imin(25),imax(25),iarr(25)
      integer levels,m,isym,irow0
      common /samg_minx/   imin(25),imax(25),iarr(25)
      common /samg_data/   levels,m,isym,irow0
      integer ierr, k, ilo, ihi, n1, n2, nnu, nna, i 
      integer, dimension(:), pointer :: iaout,jaout
      double precision, dimension(:), pointer :: aout

      if (k.lt.2.or.k.gt.levels) then 
          write(*,*) 'k=',k
          write(*,*) 'levels=',levels
          stop 'here we fetch only amg-level 1..levels !'
      endif  

! unknowns on grid k from ilo .. ihi
! use lbound ubound functions to find out about ilo,ihi
! in the calling subroutines if you dont pass them back
      ilo = imin(k)       
      ihi = imax(k)   
      nnu = ihi-ilo+1
! dimensionings for storing  matrix coefficients   
      n1  = ia(ilo)       
      n2  = ia(ihi+1)-1
      nna = n2-n1+1 

      allocate(iaout(1:nnu+1),stat=ierr) ; 
            if (ierr.gt.0) stop 'alloc failed'
      allocate(aout(1:nna),stat=ierr) ; 
            if (ierr.gt.0) stop 'alloc failed'
      allocate(jaout(1:nna),stat=ierr) ; 
            if (ierr.gt.0) stop  'alloc failed'


      iaout(1:nnu+1)   = ia(ilo:ihi+1)
      aout(1:nna)      = a(n1:n2)
      jaout(1:nna)     = ja(n1:n2)  

      write(*,*) 'De dimensie van de matrix = ', nnu
      write(*,*) 'Het aantal nullen van de matrix = ', nna 
      
      do i=1,nna
         write(*,*) i, jaout(i), aout(i) 
      end do 
      write(*,*)
      do i=1,nnu+1
         write(*,*) i, iaout(i) 
      end do 

      deallocate(iaout, jaout, aout)

      return
      end subroutine fetch
