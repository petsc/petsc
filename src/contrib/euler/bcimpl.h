c
c  Work arrays for Jacobian contributions from implicit boundary conditions.
c  Space is allocated in UserCreateEuler().  
c
c  Note:  These extra arrays are a convenient means of handling the
c         parallel vector scatter data.  We could conserve a bit of
c         space by handling this differently. 
c
      double precision b1bc(ndof,ndof,4,ysf1:yefp1,zsf1:zefp1)
      double precision b2bc(ndof,ndof,xsf1:xefp1,4,zsf1:zefp1)
      double precision b2bc_tmp(ndof,ndof,xsf1:xefp1,4,zsf1:zefp1)
      double precision b3bc(ndof,ndof,xsf1:xefp1,ysf1:yefp1,4)
c
c      double precision b1bc(ndof,ndof,4,nj,nk), b2bc(ndof,ndof,ni,4,nk)
c      double precision b3bc(ndof,ndof,ni,nj,4), b2bc_tmp(ndof,ndof,ni,4,nk)
c
c      COMMON /BCINFO/ b1bc(ndof,ndof,4,nj,nk), b2bc(ndof,ndof,ni,4,nk),
c     &                b3bc(ndof,ndof,ni,nj,4), b2bc_tmp(ndof,ndof,ni,4,nk)

