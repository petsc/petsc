c
c  "$Id: rbuild_fp.F,v 1.6 1997/10/19 21:23:51 curfman Exp curfman $";
c
c ----------------------------------------------------------------------
c
      integer function rbuild_direct_fp( f, sctype, dt, dxx )
c
c  rbuild_direct - Assembles residual vector directly (without using
c                  VecSetValues().  See the routine rbuild() for
c                  a version using VecSetValues().  
c
c                  This routine assembles only internal nodes, since
c                  boundary nodes are directly placed in the appropriate 
c                  location of the vector.
c
c  Input Parameters:
c    f      - local array for PETSc vector storage
c    sctype - type of scaling
c    dt     - pseudo-transient continuation parameter
c    dr, dru, drv, drw, de - residual components
c
c  Output Parameter:
c    Fvec      - fully assembled residual vector
c
c  Notes:
c
c   - All calls to VecSetValues() use 0-based indexing.  I.e., for
c     a vector of dimension N, the indices are 0 through N-1,
c     rather than the usual Fortran convention of 1 through N.
c
c   - Due to grid point reordering with DAs, we must always work
c     with the local grid points, then transform them to the new
c     global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
c     We cannot work directly with the global numbers for the original
c     uniprocessor grid!

#include "param.h"
#include "wing.h"
#include "dt.h"
#include "dvarbls.h"

      double precision f(0:*)
      integer i,j,k,jk,sctype,ijkx,ndof_fe

      rbuild_direct_fp = 0
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild_direct_fp: invalid value of bctype flag!'
        rbuild_direct_fp = -1
        return
      endif
      if (sctype .ne. DT_MULT .and. sctype .ne. DT_DIV) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild_direct_fp: invalid value of sctype flag!'
        rbuild_direct_fp = -1
        return
      endif
      if (model .eq. MMFP_INT) then
        ndof_fe = 0
      else if (model .eq. MMHYBRID_F_INT .or.
     &         model .eq. MMHYBRID_EF1_INT) then
        ndof_fe = 5
      else
        if (rank .eq. 0) write(6,*) 'Unsupported model type'
        rbuild_direct_fp = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c      if (bctype .eq. IMPLICIT) then
c
c      if (zsf1 .eq. 1) then
c        k=1
c        do 100 j=ysf2,yef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 100 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx) = fbcfpk1(i,j)
c 100    continue
c      endif
c
c      if (ysf1 .eq. 1) then
c         j=1
c         do 200 k=zsf2,zef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c         do 200 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcfpj1(i,k)
c 200    continue
c      endif
c
c      if (xsf1 .eq. 1) then
c        i=1
c        do 300 k=zsf2,zef01
c           ik = i-xsf + (k-zsf)*xm*ym
c        do 300 j=ysf2,yef01
c           ijkx = ndof * (ik + (j-ysf)*xm)
c           f(ijkx)   = fbcfpi1(j,k)
c 300     continue
c      endif
c
c      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (sctype .eq. DT_MULT) then
         do 400 k=zsf2,zef01
         do 400 j=ysf2,yef01
             jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 400 i=xsf2,xef01
             ijkx            = ndof * (i-xsf + jk)
             f(ijkx+ndof_fe) = dfp(i,j,k)*dt(i,j,k)
 400     continue
      else if (sctype .eq. DT_DIV) then
         do 450 k=zsf2,zef01
         do 450 j=ysf2,yef01
             jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 450 i=xsf2,xef01
             ijkx            = ndof * (i-xsf + jk)
             f(ijkx+ndof_fe) = dfp(i,j,k)
 450     continue
      else
        if (rank .eq. 0) write(6,*) 'rbuild_direct_fp: invalid sctype!'
        rbuild_direct_fp = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c      if (bctype .eq. IMPLICIT) then
c
c      if (xef01 .eq. ni) then
c        i=ni1
c        do 500 k=zsf2,zef01
c           ik = i-xsf + (k-zsf)*xm*ym
c        do 500 j=ysf2,yef01
c           ijkx = ndof * (ik + (j-ysf)*xm)
c           f(ijkx)   = fbcfpi2(j,k)
c 500    continue
c      endif
c
c      if (yef01 .eq. nj) then
c        j=nj1
c        do 600 k=zsf2,zef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 600 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcfpj2(i,k)
c 600    continue
c      endif
c
c      if (zef01 .eq. nk) then
c        k=nk1
c        do 700 j=ysf2,yef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 700 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcfpk2(i,j)
c 700    continue
c      endif
c      endif
c
      return
      end
