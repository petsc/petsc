c
c  This file contains routines that are now obselete.  But since
c  these were crucial during development stages of the parallel,
c  fully implicit version of code, we retain them for future
c  reference, although hopefully they won't be needed anymore.
c
c ***************************************************************
c routines taken from mat.F

      integer function buildmat( mat, sctype, is1, iter,
     &                     b1, b2, b3, b4, b5, b6, d, dt, ltog, nloc,
     &                     b1bc, b2bc, b3bc, b2bc_tmp, ao )
c
c  buildmat - Assembles matrix in generic PETSc format.  
c
c  Input Parameters:
c    mat      - matrix data structure
c    bctype   - type of boundary condition formulation
c    sctype   - type of scaling
c    iter     - iteration number
c    b*, d    - matrix elements corresponding to stencil points
c               for the standard 3D, 7-point stencil
c      b1, b4 - west, east
c      b2, b5 - south, north
c      b3, b6 - down, up
c      d      - center
c    dt       - pseudo-transient continuation parameter
c    ltog     - local-to-global mapping for distributed arrays
c    nloc     - number of elements in ltog mapping
c    is1      - index set with PETSc ordering numbers for certain
c               boundary nodes
c
c  Output Parameter:
c    mat      - fully assembled matrix
c
c  Notes:
c
c   - All calls to MatSetValues() use 0-based indexing.  I.e., for
c     a matrix of dimension N, the matrix rows are 0 through N-1,
c     rather than the usual Fortran convention of 1 through N.
c
c   - Due to grid point reordering with DAs, we must always work
c     with the local grid points, then transform them to the new
c     global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
c     We cannot work directly with the global numbers for the original
c     uniprocessor grid!
c
#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "wing.h"
#include "bcimpl.h"

      double precision  val,scale,negone
      integer row,km,jm,im,is1(0:*),id,ijkid,iter
      integer b1c,b2c,b3c,b4c,b5c,b6c,IV,i,j,k,l,m
      integer col,ierr,sctype,ijkn,ik,ict,ictmark
      integer nloc,ltog(0:nloc-1),rstart,rend,jk,ijk
      Mat     mat
      AO      ao

      buildmat = 0
c     IV corresponds to INSERT_VALUES mode for MatSetValues()
      IV = INSERT_VALUES
c
      if (bctype .ne. EXPLICIT .and. bctype .ne. IMPLICIT) then
        if (rank .eq. 0) 
     &    write(6,*) 'buildmat: invalid value of bctype flag!'
        buildmat = -1
        return
      endif
      if (sctype .ne. DT_MULT .and. sctype .ne. DT_DIV) then
        if (rank .eq. 0) 
     &    write(6,*) 'buildmat: invalid value of sctype flag!'
        buildmat = -1
        return
      endif

      negone = -one
      row    = -1

      if (bctype .eq. EXPLICIT) then

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Code for interior grid points i=2,ni; j=2,nj; k=2,nk
c  Only this section of code is called for bctype = EXPLICIT
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
c      do 410 k=2,nk
c      do 410 j=2,nj
c      do 410 i=2,ni
c
      do 310 k=zsf2,zef01
         km = k-1
      do 310 j=ysf2,yef01
         jm = j-1
      do 310 i=xsf2,xef01
         im = i-1
         ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         b2c = ijk - nc*gxm
         b3c = ijk - nc*gxm*gym
         b1c = ijk - nc
         b4c = ijk + nc
         b5c = ijk + nc*gxm
         b6c = ijk + nc*gxm*gym

         if (sctype .eq. DT_MULT) then
            scale = dt(i,j,k)
         else
            scale = one
         endif
         do 301 l=1,nc
            row = ltog(ijk + l)
c           uniprocessor: row = ijk + l
         do 301 m=1,nc
c     Use this assembly order for sorted columns.  Would be better to
c     to insert by blocks, but need to implement column-oriented
c     input for all PETSc matrix formats.
            if (k .gt. 2) then
               val = -b3(l,m,i,j,km)*scale
c              uniprocessor: col = b3c+m
               col = ltog(b3c+m)
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
            if (j .gt. 2)  then
               val = -b2(l,m,i,jm,k)*scale
               col = ltog(b2c+m)
c              uniprocessor: col = b2c+m
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
            if (i .gt. 2)  then
               val = -b1(l,m,im,j,k)*scale
c              uniprocessor: col = b1c+m
               col = ltog(b1c+m)
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
c Code to compute diagonal terms.  Needed if scaling for compatibility with
c implicit bc code of Driss.
            val = (b1(l,m,i,j,k) +
     &                          b2(l,m,i,j,k) +
     &                          b3(l,m,i,j,k) -
     &                          b4(l,m,im,j,k) -
     &                          b5(l,m,i,jm,k) -
     &                          b6(l,m,i,j,km) )*scale
            if (l .eq. m) then
               if (sctype .eq. DT_MULT) then
                  val = one + val
               else
                  val = one/dt(i,j,k) + val
               endif
            endif
c Instead, for sctype == DR_DIV, can directly use array d(), as computed in jform().
c            val = d(l,m,i,j,k)
            col = ltog(ijk+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            if (i .lt. ni) then
               val = b4(l,m,i,j,k)*scale
               col = ltog(b4c+m)
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
            if (j .lt. nj) then
               val = b5(l,m,i,j,k)*scale
               col = ltog(b5c+m)
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
            if (k .lt. nk) then
               val = b6(l,m,i,j,k)*scale
               col = ltog(b6c+m)
               call MatSetValues(mat,1,row,1,col,val,IV,ierr)
            endif
 301     continue
 310  continue

c ----------------------------------------------------------------------

      else if (bctype .eq. IMPLICIT) then

c Temporarily initialize diagonal to one
      if (iter .eq. 1) then
         call MatGetOwnershipRange(mat,rstart,rend,ierr)
         do 10 i=rstart,rend-1
            call MatSetValues( mat, 1, i, 1, i, one, IV, ierr )
 10       continue
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Code for implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (zsf1 .eq. 1) then
         k=1
         do 100 j=ysf2,yef01
            jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 100 i=xsf2,xef01
            ijk  = nc * (i-gxsf + jk) - 1
            ijkn = ijk + nc*gxm*gym
c           ijkn = upper neighbor (z=2)
            do 102 l=1,nc
               row = ltog(ijk + l)
            do 102 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col, 
     &              b3bc(l,m,i,j,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b3bc(l,m,i,j,k+1), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

c              call MatSetValues( mat, b3bc(l,m,i,j,k+1), row, 
c    &              ijk + nc*ni1*nj1 + m, ierr )
 102        continue
 100     continue
      endif

      if (ysf1 .eq. 1) then
         j=1
         ictmark = 0
         do 200 k=zsf2,zef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 200 i=xsf2,xef01
            ijk = nc * (i-gxsf + jk) - 1
            ijkn = ijk + nc*gxm
c           ijkn = northern neighbor (j=2)
            IF (K.GT.KTIP) GO TO 123
            IF (I.LE.ITL.OR.I.GT.ITU) GO TO 123
            do 202 l=1,nc
               row = ltog(ijk + l)
            do 202 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc(l,m,i,1,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc(l,m,i,2,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
 202        continue
         goto 200
 123        continue
c
            ID=NI+1-I
c            ID=NI+2-I
c Note: This j value corresponds to j=2 in Fortran storage!!
            ijkid = nc * ( id-1 + j*ni1 + (k-1)*ni1*nj1 ) - 1
            do 125 l=1,nc
               row = ltog(ijk + l)
               ict = ictmark
            do 124 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc(l,m,i,1,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
c
c Note: We cannot call AOApplicationToPetsc() here, since this
c       parallel database routine MUST be called by all processors.
c       Luckily, we have already determined these column numbers
c       in the index set is1 when setting up the vector scatters,
c       so we can just use them here.
c              uniprocessor: col = ijkid + m
               col = ijkid + m
               call AOApplicationToPetsc(ao,1,col,ierr)
c               if (col .ne. is1(ict))
c     &             write(6,*) 'MAT LAST: k,j,i,l,m,ict,col,ictc = ',
c     &                    k,j,i,l,m,ict,col,is1(ict)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc_tmp(l,m,i,2,k), IV, ierr )
c               call MatSetValues( mat, 1, row, 1, is1(ict),
c     &              b2bc_tmp(l,m,i,2,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,is1(ict)
                  buildmat = -1
                  return
               endif
               ict = ict + 1
 124        continue
 125        continue
            ictmark = ictmark + nc
 200     continue
      endif

      if (xsf1 .eq. 1) then
         i=1
         do 300 k=zsf2,zef01
            ik = i-gxsf + (k-gzsf)*gxm*gym
         do 300 j=ysf2,yef01
            ijk  = nc * (ik + (j-gysf)*gxm) - 1
            ijkn = ijk + nc
c           ijkn = eastern neighbor (i=2)
            do 302 l=1,nc
               row = ltog(ijk + l)
            do 302 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b1bc(l,m,1,j,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b1bc(l,m,2,j,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
 302        continue
 300     continue
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Code for interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


      do 410 k=zsf2,zef01
         km = k-1
      do 410 j=ysf2,yef01
         jm = j-1
      do 410 i=xsf2,xef01
         im = i-1
         ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         b3c = ijk - nc*gxm*gym
         b2c = ijk - nc*gxm
         b1c = ijk - nc
         b4c = ijk + nc
         b5c = ijk + nc*gxm
         b6c = ijk + nc*gxm*gym

         if (sctype .eq. DT_MULT) then
            scale = dt(i,j,k)
         else
            scale = one
         endif
         do 402 l=1,nc
            row = ltog(ijk + l)
         do 402 m=1,nc
c     Use this assembly order for sorted columns.  Would be better to
c     to insert by blocks, but need to implement column-oriented
c     input for all PETSc matrix formats.
            val = -b3(l,m,i,j,km)*scale
            col = ltog(b3c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

            val = -b2(l,m,i,jm,k)*scale
            col = ltog(b2c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

            val = -b1(l,m,im,j,k)*scale
            col = ltog(b1c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

c Code to compute diagonal terms.  Needed if scaling for compatibility with
c implicit bc code of Driss.
            val = (b1(l,m,i,j,k) +
     &                          b2(l,m,i,j,k) +
     &                          b3(l,m,i,j,k) -
     &                          b4(l,m,im,j,k) -
     &                          b5(l,m,i,jm,k) -
     &                          b6(l,m,i,j,km) )*scale
            if (l .eq. m) then
               if (sctype .eq. DT_MULT) then
                  val = one + val
               else
                  val = one/dt(i,j,k) + val
               endif
            endif
c Instead, for sctype == DR_DIV, can directly use array d(), as computed in jform().
c            val = d(l,m,i,j,k)
            col = ltog(ijk+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

            val = b4(l,m,i,j,k)*scale
            col = ltog(b4c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

            val = b5(l,m,i,j,k)*scale
            col = ltog(b5c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

            val = b6(l,m,i,j,k)*scale
            col = ltog(b6c+m)
            call MatSetValues(mat,1,row,1,col,val,IV,ierr)
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

 402     continue
 410  continue

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Code for implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (xef01 .eq. ni) then
         i=ni1
         do 500 k=zsf2,zef01
            ik = i-gxsf + (k-gzsf)*gxm*gym
         do 500 j=ysf2,yef01
            ijk  = nc * (ik + (j-gysf)*gxm) - 1
            ijkn = ijk - nc
c           ijkn = western neighbor (i=ni)
            do 502 l=1,nc
               row = ltog(ijk + l)
            do 502 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b1bc(l,m,4,j,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b1bc(l,m,3,j,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
 502        continue
 500     continue
      endif

      if (yef01 .eq. nj) then
         j=nj1
         do 600 k=zsf2,zef01
            jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 600 i=xsf2,xef01
            ijk  = nc * (i-gxsf + jk) - 1
            ijkn = ijk - nc*gxm
c           ijkn = southern neighbor (k=nj)
            do 602 l=1,nc
               row = ltog(ijk + l)
            do 602 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc(l,m,i,4,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b2bc(l,m,i,3,k), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
 602        continue
 600     continue
      endif

      if (zef01 .eq. nk) then
         k=nk1
         do 700 j=ysf2,yef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 700 i=xsf2,xef01
            ijk = nc * (i-gxsf + jk) - 1
            ijkn = ijk - nc*gxm*gym
c           ijkn = lower neighbor (k=nk)
            do 702 l=1,nc
               row = ltog(ijk + l)
            do 702 m=1,nc
               col = ltog(ijk + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b3bc(l,m,i,j,4), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif

               col = ltog(ijkn + m)
               call MatSetValues( mat, 1, row, 1, col,
     &              b3bc(l,m,i,j,3), IV, ierr )
               if (ierr .ne. 0) then
                  write(6,*) 'buildmat:rank,k,j,i,row,col = ',
     &                        rank,k,j,i,row,col
                  buildmat = -1
                  return
               endif
 702        continue
 700     continue

      endif
      endif

      return
      end

c ----------------------------------------------------------------------
c
      integer function rbuild( Fvec, sctype, dt, dxx, fff,
     &                ltog, nloc )
c
c  rbuild - Assembles residual vector.
c
c  Input Parameters:
c    Fvec   - PETSc vector data structure
c    sctype - type of scaling
c    dt     - pseudo-transient continuation parameter
c    dr, dru, drv, drw, de - residual components
c    ltog     - local-to-global mapping for distributed arrays
c    nloc     - number of elements in ltog mapping
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
#include "ibcwork.h"

      double precision  val(5)
      integer i,j,k,jk,sctype,ijkx,ierr,ik
      integer nloc,ltog(0:nloc-1),pos(5)
      Vec     Fvec

      rbuild = 0
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild: invalid value of bctype flag!'
        rbuild = -1
        return
      endif
      if (sctype .ne. DT_MULT .and. sctype .ne. DT_DIV) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild: invalid value of sctype flag!'
        rbuild = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. IMPLICIT) then

      if (zsf1 .eq. 1) then
        k=1
        do 100 j=ysf2,yef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 100 i=xsf2,xef01
           ijkx = nc * (i-gxsf + jk)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcrk1(i,j)
           val(2) = fbcruk1(i,j)
           val(3) = fbcrvk1(i,j)
           val(4) = fbcrwk1(i,j)
           val(5) = fbcek1(i,j)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 100    continue
      endif

      if (ysf1 .eq. 1) then
         j=1
         do 200 k=zsf2,zef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 200 i=xsf2,xef01
           ijkx = nc * (i-gxsf + jk)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcrj1(i,k)
           val(2) = fbcruj1(i,k)
           val(3) = fbcrvj1(i,k)
           val(4) = fbcrwj1(i,k)
           val(5) = fbcej1(i,k)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 200    continue
      endif

      if (xsf1 .eq. 1) then
        i=1
        do 300 k=zsf2,zef01
           ik = i-gxsf + (k-gzsf)*gxm*gym
        do 300 j=ysf2,yef01
           ijkx = nc * (ik + (j-gysf)*gxm)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcri1(j,k)
           val(2) = fbcrui1(j,k)
           val(3) = fbcrvi1(j,k)
           val(4) = fbcrwi1(j,k)
           val(5) = fbcei1(j,k)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 300     continue
      endif

      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (sctype .eq. DT_MULT) then
         do 400 k=zsf2,zef01
         do 400 j=ysf2,yef01
             jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 400 i=xsf2,xef01
             ijkx   = nc * (i-gxsf + jk)
             pos(1) = ltog(ijkx)
             pos(2) = ltog(ijkx+1)
             pos(3) = ltog(ijkx+2)
             pos(4) = ltog(ijkx+3)
             pos(5) = ltog(ijkx+4)
             val(1) = dr(i,j,k)*dt(i,j,k)
             val(2) = dru(i,j,k)*dt(i,j,k)
             val(3) = drv(i,j,k)*dt(i,j,k)
             val(4) = drw(i,j,k)*dt(i,j,k)
             val(5) = de(i,j,k)*dt(i,j,k)
             call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
 400     continue
      else if (sctype .eq. DT_DIV) then
         do 450 k=zsf2,zef01
         do 450 j=ysf2,yef01
             jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
         do 450 i=xsf2,xef01
             ijkx   = nc * (i-gxsf + jk)
             pos(1) = ltog(ijkx)
             pos(2) = ltog(ijkx+1)
             pos(3) = ltog(ijkx+2)
             pos(4) = ltog(ijkx+3)
             pos(5) = ltog(ijkx+4)
             val(1) = dr(i,j,k)
             val(2) = dru(i,j,k)
             val(3) = drv(i,j,k)
             val(4) = drw(i,j,k)
             val(5) = de(i,j,k)
             call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
 450     continue
      else
        if (rank .eq. 0) write(6,*) 'rbuild: invalid sctype!'
        rbuild = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. IMPLICIT) then

      if (xef01 .eq. ni) then
        i=ni1
        do 500 k=zsf2,zef01
           ik = i-gxsf + (k-gzsf)*gxm*gym
        do 500 j=ysf2,yef01
           ijkx = nc * (ik + (j-gysf)*gxm)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcri2(j,k)
           val(2) = fbcrui2(j,k)
           val(3) = fbcrvi2(j,k)
           val(4) = fbcrwi2(j,k)
           val(5) = fbcei2(j,k)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 500    continue
      endif

      if (yef01 .eq. nj) then
        j=nj1
        do 600 k=zsf2,zef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 600 i=xsf2,xef01
           ijkx = nc * (i-gxsf + jk)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcrj2(i,k)
           val(2) = fbcruj2(i,k)
           val(3) = fbcrvj2(i,k)
           val(4) = fbcrwj2(i,k)
           val(5) = fbcej2(i,k)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 600    continue
      endif

      if (zef01 .eq. nk) then
        k=nk1
        do 700 j=ysf2,yef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 700 i=xsf2,xef01
           ijkx = nc * (i-gxsf + jk)
           pos(1) = ltog(ijkx)
           pos(2) = ltog(ijkx+1)
           pos(3) = ltog(ijkx+2)
           pos(4) = ltog(ijkx+3)
           pos(5) = ltog(ijkx+4)
           val(1) = fbcrk2(i,j)
           val(2) = fbcruk2(i,j)
           val(3) = fbcrvk2(i,j)
           val(4) = fbcrwk2(i,j)
           val(5) = fbcek2(i,j)
           call VecSetValues(Fvec,5,pos,val,INSERT_VALUES,ierr)
           if (ierr .ne. 0) then
              write(6,*) 'rbuild:rank,k,j,i,ijkx = ',rank,k,j,i,ijkx
              rbuild = -1
              return
           endif
 700    continue
      endif
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Vector assembley
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c  For the Euler code, the vector assembly is done completely locally,
c  so no message-pasing is performed during these phases. 

      call VecAssemblyBegin(Fvec,ierr)
      call VecAssemblyEnd(Fvec,ierr)

      return
      end

c ***************************************************************
c routines taken from xmod.F
c
      integer function jform(eps,epsi,
     &                       b1,b2,b3,b4,b5,b6,d,dt,
     &                       xx,p,
     &                       br,bl,be,sadai,sadaj,sadak,
     &                       aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                       f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c jform - Computes the Jacobian matrix, J(x) and stores it in the 
c         (intermediate) Eagle format for later conversion to a PETSc
c         matrix.  This version uses explicit boundary conditions.
c
c         This approach of using an intermediate storage space is
c         inefficient in terms of both memory and time, but is a good
c         place to get started with an existing code.  We retain this
c         routine only to show an example of converting from an
c         existing code that uses the Eagle format.  See jform2() 
c         for the more efficient approach of directly assembling a
c         PETSc matrix.
c
c         This routine is called by FormJacobian().
c
#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "es.h"
#include "sada.h"
#include "avctrs.h"
#include "fv.h"
#include "sigma.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /FREQ/ IFREQ,ISGS
      double precision  qer(5),qel(5),eps,epsi,dql,dqr
      integer i,j,k,l,m

      jform = 0
      do 10 m=1,5
         qel(m)=zero
         qer(m)=zero
 10   continue

      do 30 m=1,5
      qel(m)=eps
c Form Jacobian elements in left part of matrix
      call nd (m,epsi,qer,qel,b1,b2,b3,xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)

      qel(m)=zero
   30 continue
      do 31 m=1,5
      qer(m)=eps
c Form Jacobian elements in right part of matrix
      call nd (m,epsi,qer,qel,b4,b5,b6,
     &         xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qer(m)=zero
   31 continue
c Find eigendecomposition of diagonal block of Jacobian; Form DT matrix.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,xx,p,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      DO 21 M=1,5
      DO 21 L=1,5
      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
      do 21 i=xsf2,xef01
c      DO 21 K=2,NK
c      DO 21 J=2,NJ
c      DO 21 I=2,NI
      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
      D(L,M,I,J,K)=dql-dqr
      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+one/DT(I,J,K)
   21 CONTINUE

      return
      end


c -------------------------------------------------------------------

      integer function jform2(eps,epsi,ltog,nloc,mat,d,dt,
     &                       xx,p,
     &                       br,bl,be,sadai,sadaj,sadak,
     &                       aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                       f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c jform2 - Forms the Jacobian matrix, J(x), with direct assembly 
c          of the matrix into the PETSc formats via calls to
c          MatSetValues().  This version uses explicit boundary
c          conditions.  This approach of direct matrix assembly 
c          is recommended -- rather than the use of the intermediate
c          Eagle format as in the routine jform().
c
c          This routine is called by FormJacobian().
c
#include "param.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "es.h"
#include "sada.h"
#include "avctrs.h"
#include "diag.h"
#include "fv.h"
#include "sigma.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /FREQ/ IFREQ,ISGS
      double precision  qer(5),qel(5),eps,epsi,dti
      integer nloc,ltog(0:nloc-1)
      integer m,ierr,nd2,ijk,jk,rc(5),i,j,k
      Mat     mat

      jform2 = 0
      do 10 m=1,5
         qel(m)=zero
         qer(m)=zero
 10   continue

      do 30 m=1,5
      qel(m)=eps
c Form Jacobian elements in left part of matrix
      ierr = nd2 (m,epsi,qer,qel,1,mat,ltog,nloc,d,
     &         xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      if (ierr .ne. 0) then
         jform2 = -1
         return 
      endif
      qel(m)=zero
   30 continue

      do 31 m=1,5
      qer(m)=eps
c Form Jacobian elements in right part of matrix
      ierr = nd2 (m,epsi,qer,qel,2,mat,ltog,nloc,d,
     &         xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)

      if (ierr .ne. 0) then
         jform2 = -1
         return 
      endif
      qer(m)=zero
   31 continue

c Find eigendecomposition of diagonal block of Jacobian; Form DT matrix.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,xx,p,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c We must specify that we are inserting matrix blocks by columns, since
c the PETSc default is row-oriented input.
      call MatSetOption(mat,MAT_COLUMN_ORIENTED,ierr)

      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
       jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
      do 21 i=xsf2,xef01
c      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
c      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
c      D(L,M,I,J,K)=dql-dqr
c      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+one/DT(I,J,K)
       ijk = ltog(nc * (i-gxsf + jk))
       rc(1) = ijk
       rc(2) = ijk + 1
       rc(3) = ijk + 2
       rc(4) = ijk + 3
       rc(5) = ijk + 4
       dti = one/DT(I,J,K)
       D(1,1,I,J,K)=D(1,1,I,J,K) + dti
       D(2,2,I,J,K)=D(2,2,I,J,K) + dti
       D(3,3,I,J,K)=D(3,3,I,J,K) + dti
       D(4,4,I,J,K)=D(4,4,I,J,K) + dti
       D(5,5,I,J,K)=D(5,5,I,J,K) + dti
       call MatSetValues(mat,nc,rc,nc,rc,d(1,1,i,j,k),
     &                   INSERT_VALUES,ierr)
   21 CONTINUE

C  The rest of the input is row-oriented.
      call MatSetOption(mat,MAT_ROW_ORIENTED,ierr)

      return
      end

c ***************************************************************
c routines taken from: implbc.F
c

      integer function jformdt(eps,epsi,b1,b2,b3,b4,b5,b6,
     &                 b1bc,b2bc,b3bc,b2bc_tmp,
     &                 d,dt,xx,p,xx_bc,p_bc,
     &                 br,bl,be,sadai,sadaj,sadak,
     &                 aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                 f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2,fff,ao)
c
c jformdt - Computes the Jacobian matrix, J(x) and stores it in the 
c           (intermediate) Eagle format for later conversion to a PETSc
c           matrix.  This version uses implicit boundary conditions.
c
c           This approach of using an intermediate storage space is
c           inefficient in terms of both memory and time, but is a good
c           place to get started with an existing code.  We retain this
c           routine only to show an example of converting from an
c           existing code that uses the Eagle format.  See jformdt2() 
c           for the more efficient approach of directly assembling a
c           PETSc matrix.
c
c           This routine is called by FormJacobian().
c
#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "ang.h"
#include "es.h"
#include "bc.h"
#include "bcimpl.h"
#include "fv.h"
#include "sigma.h"
#include "ibcwork.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)
c      COMMON /M/ B1(5,5,NI,NJ,NK),B2(5,5,NI,NJ,NK),B3(5,5,NI,NJ,NK)
c      COMMON /M/ B4(5,5,NI,NJ,NK),B5(5,5,NI,NJ,NK),B6(5,5,NI,NJ,NK)
c      COMMON /DIAG/ D(5,5,NI,NJ,NK)
c      COMMON /TSTEP/ DT(NI,NJ,NK)
c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /FREQ/ IFREQ,ISGS
c      COMMON /BCINFO/  b1bc(5,5,4,nj,nk), b2bc(5,5,ni,4,nk),
c     &                 b3bc(5,5,ni,nj,4), b2bc_tmp(5,5,ni,4,nk)
      double precision  qer(5),qel(5),eps,epsi,dql,dqr
      integer i,j,k,l,m,dummy,ierr,ndm,ndem
      AO      ao

      jformdt = 0
      do 10 m=1,5
         qel(m)=zero
         qer(m)=zero
 10   continue
      dummy = -1

      do 30 m=1,5
      qel(m)=eps
      ierr = ndm (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &          dummy,dummy,dummy,dummy,
     &          xx,p,xx_bc,p_bc,
     &          sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,ao)

      if (ierr .ne. 0) then
         jformdt = -1
         return 
      endif
      ierr = ndem (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &          dummy,dummy,dummy,
     &          xx,p,
     &          sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      if (ierr .ne. 0) then
         jformdt = -1
         return 
      endif
      call nd (m,epsi,qer,qel,b1,b2,b3,
     &          xx,p,
     &          br,bl,be,sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &          f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qel(m)=zero
   30 continue

      do 31 m=1,5
      qer(m)=eps
      call nd (m,epsi,qer,qel,b4,b5,b6,
     &          xx,p,
     &          br,bl,be,sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &          f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qer(m)=zero
   31 continue
c Find eigendecomposition of diagonal block of Jacobian; Form DT matrix.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,xx,p,
     &            sadai,sadaj,sadak,
     &            aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      DO 21 M=1,5
      DO 21 L=1,5
      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
      do 21 i=xsf2,xef01
c      DO 21 K=2,NK
c      DO 21 J=2,NJ
c      DO 21 I=2,NI
      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
      D(L,M,I,J,K)=dql-dqr
      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+one/DT(I,J,K)
   21 CONTINUE

      return
      end

c -------------------------------------------------------------------
c
c   rscale - Scales input vector by dt.
c
      integer function rscale( dt, dxx )

#include "param.h"
#include "dt.h"
#include "dvarbls.h"

      integer i,j,k

      rscale = 0
      do 100 k=zsf2,zef01
      do 100 j=ysf2,yef01
      do 100 i=xsf2,xef01
c      do 100 k=2,nk
c      do 100 j=2,nj
c      do 100 i=2,ni
         dr(i,j,k)  = dt(i,j,k)*dr(i,j,k)
         dru(i,j,k) = dt(i,j,k)*dru(i,j,k)
         drv(i,j,k) = dt(i,j,k)*drv(i,j,k)
         drw(i,j,k) = dt(i,j,k)*drw(i,j,k)
         de(i,j,k)  = dt(i,j,k)*de(i,j,k)
 100  continue

      return
      end

c ***************************************************************
c routines taken from xmod2.F

      SUBROUTINE ND (m,epsi,qer,qel,b1,b2,b3,
     &               xx,p,
     &               br,bl,be,sadai,sadaj,sadak,
     &               aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &               f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c nd - Uses finite differencing to compute a numerical Jacobian
c      estimation for the interior grid points only (no boundary
c      points).  This version stores the Jacobian in intermediate
c      work arrays in the Eagle format, for later translation to
c      a PETSc matrix.  This approach is less efficient than
c      directly assembling the PETSc matrix as in the routien nd2().
c
c      This routine is called by jform() (explicit boundary conditions)
c      and jformdt() (implicit boundary conditions).
c
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "es.h"
#include "condi.h"
#include "sigma.h"
#include "fv.h"

C   NDL COMPUTES THE LEFT NUMERICAL JACOBIAN.
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /ES/ BR(5,5,NI,NJ,NK),BL(5,5,NI,NJ,NK),BE(2,5,NI,NJ,NK)
c      common /fv/ f1(5,ni,nj,nk),g1(5,ni,nj,nk),h1(5,ni,nj,nk)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /SIGMA/ SP(5,NI),SM(5,NI)
c      COMMON /SIGMA/ SP1(5,NI),SP2(5,NI),SM1(5,NI),SM2(5,NI)
c      COMMON /CONDI/ ORDER,PSI,LIMIT
c      COMMON /FREQ/ IFREQ,ISGS
c      double precision b1(5,5,ni1,nj1,nk1),b2(5,5,ni1,nj1,nk1)
c      double precision b3(5,5,ni1,nj1,nk1)
c
c Note: These dimensions MUST agree with those in matrixb.h !!
      double precision  B1(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision  B2(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision  B3(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision  dq(5,d_ni),f(5,d_ni)
      double precision  ql(5,d_ni),qer(5),qel(5)
      double precision  epsi,tl,pl
      integer i,j,k,l,m,ib,jb,kb

C   NUMERICAL JACOBIAN IN THE I DIRECTION
      ib = 1
      jb = 2
      kb = 2
      CALL RLVECS(QER,QEL,1,0,0,ib,jb,kb,AIX,AIY,AIZ,SADAI,
     &                  xx,p,br,bl,be)

      do 2 k=gzsf2w,zef01
      do 2 j=gysf2w,yef01
      do 21 i=gxsf1,xef01
c      DO 2 K=2,NK
c      DO 2 J=2,NJ
c      DO 21 I=1,NI
      ql(1,I)=R(I,J,K)+qel(1)
      ql(2,I)=RU(I,J,K)+qel(2)
      ql(3,I)=RV(I,J,K)+qel(3)
      ql(4,I)=RW(I,J,K)+qel(4)
      ql(5,I)=E(I,J,K)+qel(5)
      DQ(1,I)=R(I+1,J,K)+qer(1)-ql(1,i)
      DQ(2,I)=RU(I+1,J,K)+qer(2)-ql(2,i)
      DQ(3,I)=RV(I+1,J,K)+qer(3)-ql(3,i)
      DQ(4,I)=RW(I+1,J,K)+qer(4)-ql(4,i)
      DQ(5,I)=E(I+1,J,K)+qer(5)-ql(5,i)
      DO 21 L=1,5
   21 SP(L,I)=BL(L,1,I,J,K)*DQ(1,I)+BL(L,2,I,J,K)*DQ(2,I)+
     *        BL(L,3,I,J,K)*DQ(3,I)+BL(L,4,I,J,K)*DQ(4,I)+
     *        BL(L,5,I,J,K)*DQ(5,I)

      DO 22 I=gxsf1w,xef01
c      DO 22 I=1,NI
      DO 22 L=1,5
      SM2(L,I)=BE(2,L,I,J,K)*SP(L,I)
   22 SP2(L,I)=BE(1,L,I,J,K)*SP(L,I)
      DO 23 I=gxsf1w,xef01
c      DO 23 I=1,NI
      DO 23 L=1,5
   23 F(L,I)=BR(L,1,I,J,K)*SM2(1,I)+BR(L,2,I,J,K)*SM2(2,I)+
     *       BR(L,3,I,J,K)*SM2(3,I)+BR(L,4,I,J,K)*SM2(4,I)+
     *       BR(L,5,I,J,K)*SM2(5,I)
      DO 2 I=gxsf1w,xef01
c      DO 2 I=1,NI
      TL=(ql(2,i)*AIX(I,J,K)+ql(3,i)*AIY(I,J,K)+
     *    ql(4,i)*AIZ(I,J,K))/ql(1,i)
      pl=gm1*(ql(5,i)-p5*(ql(2,i)**2+ql(3,i)**2+ql(4,i)**2)
     * /ql(1,i))
      F(1,i)=ql(1,i)*TL+F(1,i)
      F(2,i)=ql(2,i)*TL+AIX(I,J,K)*pl+F(2,i)
      F(3,i)=ql(3,i)*TL+AIY(I,J,K)*pl+F(3,i)
      F(4,i)=ql(4,i)*TL+AIZ(I,J,K)*pl+F(4,i)
      F(5,i)=(ql(5,i)+pl)*TL+F(5,i)
      b1(1,m,i,j,k)=(f(1,i)-f1(1,i,j,k))*epsi
      b1(2,m,i,j,k)=(f(2,i)-f1(2,i,j,k))*epsi
      b1(3,m,i,j,k)=(f(3,i)-f1(3,i,j,k))*epsi
      b1(4,m,i,j,k)=(f(4,i)-f1(4,i,j,k))*epsi
      b1(5,m,i,j,k)=(f(5,i)-f1(5,i,j,k))*epsi
    2 continue

C   NUMERICAL JACOBIAN IN THE J DIRECTION
      ib = 2
      jb = 1
      kb = 2
      CALL RLVECS(QER,QEL,0,1,0,ib,jb,kb,AJX,AJY,AJZ,SADAJ,
     &                   xx,p,br,bl,be)

      do 3 k=gzsf2w,zef01
      do 3 i=gxsf2w,xef01
      do 31 j=gysf1,yef01
c      DO 3 K=2,NK
c      DO 3 I=2,NI
c      DO 31 J=1,NJ
      ql(1,J)=R(I,J,K)+qel(1)
      ql(2,J)=RU(I,J,K)+qel(2)
      ql(3,J)=RV(I,J,K)+qel(3)
      ql(4,J)=RW(I,J,K)+qel(4)
      ql(5,J)=E(I,J,K)+qel(5)
      DQ(1,J)=R(I,J+1,K)+qer(1)-ql(1,j)
      DQ(2,J)=RU(I,J+1,K)+qer(2)-ql(2,j)
      DQ(3,J)=RV(I,J+1,K)+qer(3)-ql(3,j)
      DQ(4,J)=RW(I,J+1,K)+qer(4)-ql(4,j)
      DQ(5,J)=E(I,J+1,K)+qer(5)-ql(5,j)
      DO 31 L=1,5
   31 SP(L,J)=BL(L,1,I,J,K)*DQ(1,J)+BL(L,2,I,J,K)*DQ(2,J)+
     *        BL(L,3,I,J,K)*DQ(3,J)+BL(L,4,I,J,K)*DQ(4,J)+
     *        BL(L,5,I,J,K)*DQ(5,J)
      DO 32 J=gysf1w,yef01
c      DO 32 J=1,NJ
      DO 32 L=1,5
      SM2(L,J)=BE(2,L,I,J,K)*SP(L,J)
   32 SP2(L,J)=BE(1,L,I,J,K)*SP(L,J)
      DO 33 J=gysf1w,yef01
c      DO 33 J=1,NJ
      DO 33 L=1,5
   33 F(L,J)=BR(L,1,I,J,K)*SM2(1,J)+BR(L,2,I,J,K)*SM2(2,J)+
     *       BR(L,3,I,J,K)*SM2(3,J)+BR(L,4,I,J,K)*SM2(4,J)+
     *       BR(L,5,I,J,K)*SM2(5,J)
      DO 3 J=gysf1w,yef01
c      DO 3 J=1,NJ
      TL=(ql(2,j)*AJX(I,J,K)+ql(3,j)*AJY(I,J,K)+
     *    ql(4,j)*AJZ(I,J,K))/ql(1,j)
      pl=gm1*(ql(5,j)-p5*(ql(2,j)**2+ql(3,j)**2+ql(4,j)**2)
     * /ql(1,j))
      F(1,j)=ql(1,j)*TL+F(1,j)
      F(2,j)=ql(2,j)*TL+AJX(I,J,K)*pl+F(2,j)
      F(3,j)=ql(3,j)*TL+AJY(I,J,K)*pl+F(3,j)
      F(4,j)=ql(4,j)*TL+AJZ(I,J,K)*pl+F(4,j)
      F(5,j)=(ql(5,j)+pl)*TL+F(5,j)
      b2(1,m,i,j,k)=(f(1,j)-g1(1,i,j,k))*epsi
      b2(2,m,i,j,k)=(f(2,j)-g1(2,i,j,k))*epsi
      b2(3,m,i,j,k)=(f(3,j)-g1(3,i,j,k))*epsi
      b2(4,m,i,j,k)=(f(4,j)-g1(4,i,j,k))*epsi
      b2(5,m,i,j,k)=(f(5,j)-g1(5,i,j,k))*epsi
    3 continue
C   NUMERICAL JACOBIAN IN THE K DIRECTION
      ib = 2
      jb = 2
      kb = 1
      CALL RLVECS(QER,QEL,0,0,1,ib,jb,kb,AKX,AKY,AKZ,SADAK,
     &                   xx,p,br,bl,be)

      do 4 j=gysf2w,yef01
      do 4 i=gxsf2w,xef01
      do 41 k=gzsf1,zef01
c      DO 4 J=2,NJ
c      DO 4 I=2,NI
c      DO 41 K=1,NK
      ql(1,K)=R(I,J,K)+qel(1)
      ql(2,K)=RU(I,J,K)+qel(2)
      ql(3,K)=RV(I,J,K)+qel(3)
      ql(4,K)=RW(I,J,K)+qel(4)
      ql(5,K)=E(I,J,K)+qel(5)
      DQ(1,K)=R(I,J,K+1)+qer(1)-ql(1,k)
      DQ(2,K)=RU(I,J,K+1)+qer(2)-ql(2,k)
      DQ(3,K)=RV(I,J,K+1)+qer(3)-ql(3,k)
      DQ(4,K)=RW(I,J,K+1)+qer(4)-ql(4,k)
      DQ(5,K)=E(I,J,K+1)+qer(5)-ql(5,k)
      DO 41 L=1,5
   41 SP(L,K)=BL(L,1,I,J,K)*DQ(1,K)+BL(L,2,I,J,K)*DQ(2,K)+
     *        BL(L,3,I,J,K)*DQ(3,K)+BL(L,4,I,J,K)*DQ(4,K)+
     *        BL(L,5,I,J,K)*DQ(5,K)
      do 42 k=gzsf1w,zef01
c      DO 42 K=1,NK
      DO 42 L=1,5
      SM2(L,K)=BE(2,L,I,J,K)*SP(L,K)
   42 SP2(L,K)=BE(1,L,I,J,K)*SP(L,K)
      do 43 k=gzsf1w,zef01
c      DO 43 K=1,NK
      DO 43 L=1,5
   43 F(L,K)=BR(L,1,I,J,K)*SM2(1,K)+BR(L,2,I,J,K)*SM2(2,K)+
     *       BR(L,3,I,J,K)*SM2(3,K)+BR(L,4,I,J,K)*SM2(4,K)+
     *       BR(L,5,I,J,K)*SM2(5,K)
      do 4 k=gzsf1w,zef01
c      DO 4 K=1,NK
      TL=(ql(2,k)*AKX(I,J,K)+ql(3,k)*AKY(I,J,K)+
     *    ql(4,k)*AKZ(I,J,K))/ql(1,k)
      pl=gm1*(ql(5,k)-p5*(ql(2,k)**2+ql(3,k)**2+ql(4,k)**2)
     * /ql(1,k))
      F(1,k)=ql(1,k)*TL+F(1,k)
      F(2,k)=ql(2,k)*TL+AKX(I,J,K)*pl+F(2,k)
      F(3,k)=ql(3,k)*TL+AKY(I,J,K)*pl+F(3,k)
      F(4,k)=ql(4,k)*TL+AKZ(I,J,K)*pl+F(4,k)
      F(5,k)=(ql(5,k)+pl)*TL+F(5,k)
      b3(1,m,i,j,k)=(f(1,k)-h1(1,i,j,k))*epsi
      b3(2,m,i,j,k)=(f(2,k)-h1(2,i,j,k))*epsi
      b3(3,m,i,j,k)=(f(3,k)-h1(3,i,j,k))*epsi
      b3(4,m,i,j,k)=(f(4,k)-h1(4,i,j,k))*epsi
      b3(5,m,i,j,k)=(f(5,k)-h1(5,i,j,k))*epsi
    4 continue
      return
      end
