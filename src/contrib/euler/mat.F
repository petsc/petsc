      integer function buildmat( mat, sctype, is1, iter,
     &                     b1, b2, b3, b4, b5, b6, d, dt, ltog, nloc,
     &                     b1bc, b2bc, b3bc, b2bc_tmp )
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

      Double  val,scale,negone
      integer row,km,jm,im,is1(0:*),id,ijkid,iter
      integer b1c,b2c,b3c,b4c,b5c,b6c,IV,i,j,k,l,m
      integer col,ierr,sctype,ijkn,ik,ict,ictmark
      integer nloc,ltog(0:nloc-1),rstart,rend,jk,ijk
      Mat     mat

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
c               col = ijkid + m
c               call AOApplicationToPetsc(ao,1,col,ierr)
c               call MatSetValues( mat, 1, row, 1, col,
c     &              b2bc_tmp(l,m,i,2,k), IV, ierr )
               call MatSetValues( mat, 1, row, 1, is1(ict),
     &              b2bc_tmp(l,m,i,2,k), IV, ierr )
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
c      integer function rbuild( Fvec, sctype, dt, dr, dru, 
c     &                drv, drw, de, ltog, nloc,
      integer function rbuild( Fvec, sctype, dt, dxx,
     &                ltog, nloc,
     &                fbcri1, fbcrui1, fbcrvi1, fbcrwi1, fbcei1,
     &                fbcri2, fbcrui2, fbcrvi2, fbcrwi2, fbcei2,
     &                fbcrj1, fbcruj1, fbcrvj1, fbcrwj1, fbcej1,
     &                fbcrj2, fbcruj2, fbcrvj2, fbcrwj2, fbcej2,
     &                fbcrk1, fbcruk1, fbcrvk1, fbcrwk1, fbcek1,
     &                fbcrk2, fbcruk2, fbcrvk2, fbcrwk2, fbcek2 )
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


      Double  val(5)
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

c ----------------------------------------------------------------------
c
      integer function rbuild_direct( f, sctype, dt, dxx,
     &                fbcri1, fbcrui1, fbcrvi1, fbcrwi1, fbcei1,
     &                fbcri2, fbcrui2, fbcrvi2, fbcrwi2, fbcei2,
     &                fbcrj1, fbcruj1, fbcrvj1, fbcrwj1, fbcej1,
     &                fbcrj2, fbcruj2, fbcrvj2, fbcrwj2, fbcej2,
     &                fbcrk1, fbcruk1, fbcrvk1, fbcrwk1, fbcek1,
     &                fbcrk2, fbcruk2, fbcrvk2, fbcrwk2, fbcek2 )
c      integer function rbuild_direct( f, sctype, dt, dr, dru, 
c     &                drv, drw, de, 
c     &                fbcri1, fbcrui1, fbcrvi1, fbcrwi1, fbcei1,
c     &                fbcri2, fbcrui2, fbcrvi2, fbcrwi2, fbcei2,
c     &                fbcrj1, fbcruj1, fbcrvj1, fbcrwj1, fbcej1,
c     &                fbcrj2, fbcruj2, fbcrvj2, fbcrwj2, fbcej2,
c     &                fbcrk1, fbcruk1, fbcrvk1, fbcrwk1, fbcek1,
c     &                fbcrk2, fbcruk2, fbcrvk2, fbcrwk2, fbcek2 )
c
c  rbuild - Assembles residual vector directly (without using
c           VecSetValues().
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
#include "ibcwork.h"


      Double  f(0:*)
      integer i,j,k,jk,sctype,ijkx,ik

      rbuild_direct = 0
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild_direct: invalid value of bctype flag!'
        rbuild_direct = -1
        return
      endif
      if (sctype .ne. DT_MULT .and. sctype .ne. DT_DIV) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild_direct: invalid value of sctype flag!'
        rbuild_direct = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. IMPLICIT) then

      if (zsf1 .eq. 1) then
        k=1
        do 100 j=ysf2,yef01
           jk = (j-ysf)*xm + (k-zsf)*xm*ym
        do 100 i=xsf2,xef01
           ijkx = nc * (i-xsf + jk)
           f(ijkx)   = fbcrk1(i,j) 
           f(ijkx+1) = fbcruk1(i,j)
           f(ijkx+2) = fbcrvk1(i,j)
           f(ijkx+3) = fbcrwk1(i,j)
           f(ijkx+4) = fbcek1(i,j)
 100    continue
      endif

      if (ysf1 .eq. 1) then
         j=1
         do 200 k=zsf2,zef01
           jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 200 i=xsf2,xef01
           ijkx = nc * (i-xsf + jk)
           f(ijkx)   = fbcrj1(i,k)
           f(ijkx+1) = fbcruj1(i,k)
           f(ijkx+2) = fbcrvj1(i,k)
           f(ijkx+3) = fbcrwj1(i,k)
           f(ijkx+4) = fbcej1(i,k)
 200    continue
      endif

      if (xsf1 .eq. 1) then
        i=1
        do 300 k=zsf2,zef01
           ik = i-xsf + (k-zsf)*xm*ym
        do 300 j=ysf2,yef01
           ijkx = nc * (ik + (j-ysf)*xm)
           f(ijkx)   = fbcri1(j,k)
           f(ijkx+1) = fbcrui1(j,k)
           f(ijkx+2) = fbcrvi1(j,k)
           f(ijkx+3) = fbcrwi1(j,k)
           f(ijkx+4) = fbcei1(j,k)
 300     continue
      endif

      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (sctype .eq. DT_MULT) then
         do 400 k=zsf2,zef01
         do 400 j=ysf2,yef01
             jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 400 i=xsf2,xef01
             ijkx      = nc * (i-xsf + jk)
             f(ijkx)   = dr(i,j,k)*dt(i,j,k)
             f(ijkx+1) = dru(i,j,k)*dt(i,j,k)
             f(ijkx+2) = drv(i,j,k)*dt(i,j,k)
             f(ijkx+3) = drw(i,j,k)*dt(i,j,k)
             f(ijkx+4) = de(i,j,k)*dt(i,j,k)
 400     continue
      else if (sctype .eq. DT_DIV) then
         do 450 k=zsf2,zef01
         do 450 j=ysf2,yef01
             jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 450 i=xsf2,xef01
             ijkx   = nc * (i-xsf + jk)
             f(ijkx)   = dr(i,j,k)
             f(ijkx+1) = dru(i,j,k)
             f(ijkx+2) = drv(i,j,k)
             f(ijkx+3) = drw(i,j,k)
             f(ijkx+4) = de(i,j,k)
 450     continue
      else
        if (rank .eq. 0) write(6,*) 'rbuild_direct: invalid sctype!'
        rbuild_direct = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. IMPLICIT) then

      if (xef01 .eq. ni) then
        i=ni1
        do 500 k=zsf2,zef01
           ik = i-xsf + (k-zsf)*xm*ym
        do 500 j=ysf2,yef01
           ijkx = nc * (ik + (j-ysf)*xm)
           f(ijkx)   = fbcri2(j,k)
           f(ijkx+1) = fbcrui2(j,k)
           f(ijkx+2) = fbcrvi2(j,k)
           f(ijkx+3) = fbcrwi2(j,k)
           f(ijkx+4) = fbcei2(j,k)
 500    continue
      endif

      if (yef01 .eq. nj) then
        j=nj1
        do 600 k=zsf2,zef01
           jk = (j-ysf)*xm + (k-zsf)*xm*ym
        do 600 i=xsf2,xef01
           ijkx = nc * (i-xsf + jk)
           f(ijkx)   = fbcrj2(i,k)
           f(ijkx+1) = fbcruj2(i,k)
           f(ijkx+2) = fbcrvj2(i,k)
           f(ijkx+3) = fbcrwj2(i,k)
           f(ijkx+4) = fbcej2(i,k)
 600    continue
      endif

      if (zef01 .eq. nk) then
        k=nk1
        do 700 j=ysf2,yef01
           jk = (j-ysf)*xm + (k-zsf)*xm*ym
        do 700 i=xsf2,xef01
           ijkx = nc * (i-xsf + jk)
           f(ijkx)   = fbcrk2(i,j)
           f(ijkx+1) = fbcruk2(i,j)
           f(ijkx+2) = fbcrvk2(i,j)
           f(ijkx+3) = fbcrwk2(i,j)
           f(ijkx+4) = fbcek2(i,j)
 700    continue
      endif
      endif

      return
      end

c -------------------------------------------------------------------

      integer function nzmat(format,ncd,nc_block,istart,iend,is1,
     &                       ltog,nloc,wkdim,nnz_d,nnz_o)
c
c  nnzmat - Determines nonzero structure of the (parallel) Jacobian
c           matrix in the arrays nnz_d and nnz_o.  These arrays can
c           then be passed to the PETSc MatCreateXXX() routines to
c           preallocate matrix memory. This routine supports several
c           matrix formats.  See the PETSc apps manual for info 
c           about the use of these arrays for matrix memory preallocation.
c
c  Input Parameters:
c    format   - PETSc matrix format
c    nc       - Dof per node
c    nc_block - block size (nc, except when experimenting with block size = 1)
c    is1      - index set with PETSc ordering numbers for certain
c               boundary nodes
c    bctype   - type of boundary condition formulation
c    istart, iend - starting and ending owned rows of matrix,
c                   as determined by the distributed array
c    ltog     - local-to-global mapping for distributed arrays
c    nloc     - number of elements in ltog mapping
c    wkdim    - length of arrays nnz_d and nnz_o
c    nnz_d    - array for diagonal submatrices
c    nnz_o    - array for off-diagonal submatrices
c
c  Notes:
c
c   - All PETSc matrices use 0-based indexing.  I.e., for
c     a matrix of dimension N, the matrix rows are 0 through N-1,
c     rather than the usual Fortran convention of 1 through N.
c
c   - Due to grid point reordering with DAs, we must always work
c     with the local grid points, then transform them to the new
c     global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
c     We cannot work directly with the global numbers for the original
c     uniprocessor grid!
c
c   - This method of counting nonzeros (ordering most rapidly by i,
c     then j, then k) corresponds with the DA partitioning/ordering
c     of the grid.
c
#include "param.h"
#include "wing.h"

C  Note: Can delete ncd

      integer ijk,ijkn,ncd,istart,iend,format,ijk_noghosts
      integer b1c,b2c,b3c,b4c,b5c,b6c,i,j,k,l,wkdim,nc_block
      integer col,ict,chunk,row,id,ijkid,is1(0:*)
      integer nloc,ltog(0:nloc-1),nnz_d(0:wkdim-1),nnz_o(0:wkdim-1)

      nzmat = 0

c  The same code supports both block and point storage formats
c  through the chunk variable.
      if (format .eq. MATMPIAIJ .or. format .eq. MATSEQAIJ
     &         .or. format .eq. MATMPIROWBS) then
        chunk = nc
      else if (format .eq. MATMPIBAIJ .or. format .eq. MATSEQBAIJ) then
        if (nc_block .eq. nc) then
           chunk = 1
        else if (nc_block .eq. 1) then
           chunk = nc
        else
           if (rank .eq. 0) write(6,*) 'nzmat: unsupported block size!'
           nzmat = -1
           return
        endif
      else
        if (rank .eq. 0) write(6,*) 'nzmat: unsupported matrix format!'
        nzmat = -1
        return
      endif
c
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)) then
        if (rank .eq. 0) 
     &     write(6,*) 'nzmat: invalid value of bctype flag!'
        nzmat = -1
        return
      endif

c  Initialize arrays to zero
      do 10 i=0,wkdim-1
         nnz_d(i) = 0
         nnz_o(i) = 0
 10   continue


c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Explicit boundary conditions:
c   - Use only interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. EXPLICIT) then

      ict = 0
      do 30 k=zsf2,zef01
      do 30 j=ysf2,yef01
      do 30 i=xsf2,xef01
         ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         b3c = ijk - nc*gxm*gym
         b2c = ijk - nc*gxm
         b1c = ijk - nc
         b4c = ijk + nc
         b5c = ijk + nc*gxm
         b6c = ijk + nc*gxm*gym
         do 20 l=1,chunk
            if (k .gt. 2) then
               col = ltog(b3c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            if (j .gt. 2)  then
               col = ltog(b2c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            if (i .gt. 2)  then
               col = ltog(b1c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            col = ltog(ijk+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(ict) = nnz_d(ict) + chunk
            else
               nnz_o(ict) = nnz_o(ict) + chunk
            endif
            if (i .lt. ni) then
               col = ltog(b4c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            if (j .lt. nj) then
               col = ltog(b5c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            if (k .lt. nk) then
               col = ltog(b6c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk
               else
                  nnz_o(ict) = nnz_o(ict) + chunk
               endif
            endif
            ict = ict + 1
 20     continue
 30   continue

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions:
c   - Use interior and boundary grid points 
c        (i=1,ni1; j=1,nj1; k=1,nk1)
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      else if (bctype .eq. IMPLICIT) then

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (zsf1 .eq. 1) then
         k=1
         do 100 j=ysf2,yef01
         do 100 i=xsf2,xef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijkn = ijk + nc*gxm*gym
c           ijkn = upper neighbor (z=2)
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            do 102 l=1,chunk
               row = ijk_noghosts+l
               col = ltog(ijk+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
 102        continue
 100     continue
      endif

      if (ysf1 .eq. 1) then
         j=1
         ict = 0
         do 200 k=zsf2,zef01
         do 200 i=xsf2,xef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk + nc*gxm
c           ijkn = northern neighbor (j=2)
            IF (K.GT.KTIP) GO TO 123
            IF (I.LE.ITL.OR.I.GT.ITU) GO TO 123
            do 202 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
                  write(6,*) 'nnz_D: i,j,k,row,ct = ',
     &                        rank,i,j,k,row,nnz_d(row)
               else
                  nnz_o(row) = nnz_o(row) + chunk
                  write(6,*) 'nnz_O: i,j,k,row,ct = ',
     &                    rank,i,j,k,row,nnz_o(row)
               endif
 202        continue
         goto 200
 123        continue
c
            ID=NI+1-I
c            ID=NI+2-I
c Note: This j value corresponds to j=2!!
            ijkid = nc * ( id-1 + j*ni1 + (k-1)*ni1*nj1 ) - 1
            do 124 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
c Note: We cannot call AOApplicationToPetsc() here, since this
c       parallel database routine MUST be called by all processors.
c       Luckily, we have already determined these column numbers
c       in the index set is1 when setting up the vector scatters,
c       so we can just use them here.
c              uniprocessor: col = ijkid + l
c               col = ijkid + l
c               call AOApplicationToPetsc(ao,1,col,ierr)
               col = is1(ict)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               ict = ict + 1
 124        continue
 200     continue
      endif

      if (xsf1 .eq. 1) then
         i=1
         do 300 k=zsf2,zef01
         do 300 j=ysf2,yef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk + nc
c           ijkn = eastern neighbor (i=2)
            do 302 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
 302        continue
 300     continue
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
      do 410 k=zsf2,zef01
      do 410 j=ysf2,yef01
      do 410 i=xsf2,xef01
         ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &              (k-zsf)*xm*ym) - 1
         b2c = ijk - nc*gxm
         b3c = ijk - nc*gxm*gym
         b5c = ijk + nc*gxm
         b6c = ijk + nc*gxm*gym
         b1c = ijk - nc
         b4c = ijk + nc
         do 402 l=1,chunk
            row = ijk_noghosts + l
            col = ltog(b3c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(b2c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(b1c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(ijk+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(b4c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(b5c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
            col = ltog(b6c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk
            else
               nnz_o(row) = nnz_o(row) + chunk
            endif
 402     continue
 410  continue

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (xef01 .eq. ni) then
         i=ni1
         do 500 k=zsf2,zef01
         do 500 j=ysf2,yef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - nc
c           ijkn = western neighbor (i=ni)
            do 502 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
 502        continue
 500     continue
      endif

      if (yef01 .eq. nj) then
         j=nj1
         do 600 k=zsf2,zef01
         do 600 i=xsf2,xef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - nc*gxm
c           ijkn = southern neighbor (k=nj)
            do 602 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
 602        continue
 600     continue
      endif

      if (zef01 .eq. nk) then
         k=nk1
         do 700 j=ysf2,yef01
         do 700 i=xsf2,xef01
            ijk = nc * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - nc*gxm*gym
c           ijkn = lower neighbor (k=nk)
            do 702 l=1,chunk
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk
               else
                  nnz_o(row) = nnz_o(row) + chunk
               endif
 702        continue
 700     continue
      endif

C  Remaining rows have just 1 nonzero on diagonal (these correspond to
C  nodes on the edges of the 3D domain).

      do 90 i=0,wkdim-1
         if (nnz_d(i) .eq. 0) nnz_d(i) = 1
 90   continue

      endif

      return
      end
