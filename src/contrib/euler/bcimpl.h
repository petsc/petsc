!
!  Work arrays for Jacobian contributions from implicit boundary conditions.
!  Space is allocated in UserCreateEuler().  
!
!  Note:  These extra arrays are a convenient means of handling the
!         parallel vector scatter data.  We could conserve a bit of
!         space by handling this differently. 
!
      double precision b1bc(ndof,ndof,4,ysf1:yefp1,zsf1:zefp1)
      double precision b2bc(ndof,ndof,xsf1:xefp1,4,zsf1:zefp1)
      double precision b2bc_tmp(ndof,ndof,xsf1:xefp1,4,zsf1:zefp1)
      double precision b3bc(ndof,ndof,xsf1:xefp1,ysf1:yefp1,4)
!
!      double precision b1bc(ndof,ndof,4,nj,nk), b2bc(ndof,ndof,ni,4,nk)
!      double precision b3bc(ndof,ndof,ni,nj,4), b2bc_tmp(ndof,ndof,ni,4,nk)
!
!      COMMON /BCINFO/ b1bc(ndof,ndof,4,nj,nk), b2bc(ndof,ndof,ni,4,nk),
!     &                b3bc(ndof,ndof,ni,nj,4), b2bc_tmp(ndof,ndof,ni,4,nk)

