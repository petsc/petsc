      integer function buildbdmat( mat, sctype,
     &                     b1, b2, b3, b4, b5, b6, d, dt, ltog, nloc )
c
c Assemble matrix in block diagonal PETSc format
c
c Notes:
c   This routine has not been used in ages; it's inevitably broken.
c   Fixes needed:  use generao ndof instead of hardwired value.
c 
#include "param.h"
#include "matrixb.h"
#include "dt.h"
#include "diag.h"
#include "wing.h"
#include "bcimpl.h"

c      COMMON /BCINFO/ b1bc(5,5,4,nj,nk), b2bc(5,5,ni,4,nk),
c     &                b3bc(5,5,ni,nj,4), b2bc_tmp(5,5,ni,4,nk)
c      COMMON /WING/ ITL,ITU,ILE,KTIP

      integer          ltog(nloc), nloc
      integer          bctype,mat,row,ndof,i,j,k,m,l,sctype
      integer          nim,njm,nkm,km,jm,im,jdiag,ijid,ij,id,ifoo
      double precision val,scale
      logical totimpl

      buildbdmat = 0
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)
     &    .and. (bctype .ne. IMPLICIT_SIZE)) then
        if (rank .eq. 0)
     &     write(6,*) 'buildbdmat: invalid value of bctype flag!'
        buildbdmat = -1
        return
      endif

c
c  MUST FIX TO HANDLE SCTYPE!!!
c

      totimpl = .false.
c
c Set ifoo = 0, no implicit terms
c Note: If we add implicit terms, must scale b1bc, b2bc, b3bc, accordingly!
c Eventually, should remove scale variable from other assembly stuff
c
      ifoo = 0
      row = -1
c Use row/columns with 0 beginning indices (as in C)
      ndof = 5

      if (bctype .eq. EXPLICIT) then
         nim = ni - 1
         njm = nj - 1
         nkm = nk - 1
      else

c *** Set diagonal terms first (to avoid scaling and zeroing problems)
      do 400 k=2,nk
         km = k-1
      do 400 j=2,nj
         jm = j-1
      do 400 i=2,ni
c         scale = dt(i,j,k)
         scale = one
         im = i-1
         do 390 l=1,ndof
         do 390 m=1,ndof
            val = (b1(l,m,i,j,k) +
     &                          b2(l,m,i,j,k) +
     &                          b3(l,m,i,j,k) -
     &                          b4(l,m,im,j,k) -
     &                          b5(l,m,i,jm,k) -
     &                          b6(l,m,i,j,km) )*scale
c            if (l .eq. m) val = one + val
            if (l .eq. m) val = one/dt(i,j,k) + val
            d(l,m,i,j,k) = val
 390     continue
 400  continue

      if (ifoo .eq. 1) then
         k=1
         do 100 j=1,nj1
            jm = j-1
         do 100 i=1,ni1
            im = i-1
            do 102 l=1,ndof
            do 102 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (i .ne. 1) b1(l,m,im,j,k) = zero
               if (j .ne. 1) b2(l,m,i,jm,k) = zero
               b4(l,m,i,j,k) = zero
               b5(l,m,i,j,k) = zero
c              call SpSetValue( mat, b3bc(l,m,i,j,k), row, ij+m )
c              call SpSetValue( mat, b3bc(l,m,i,j,k+1), row,
c    &              ij + ndof*ni1*nj1 + m )
               if (i .ne. 1 .and. j .ne. 1 .and.
     &                       i .ne. ni1 .and. j .ne. nj1) then
                  d(l,m,i,j,k) = b3bc(l,m,i,j,k)
                  b6(l,m,i,j,k) = b3bc(l,m,i,j,k+1)
               else
                  b6(l,m,i,j,k) = zero
               endif
 102        continue
 100     continue

         j=1
         do 200 k=1,nk1
            km = k-1
         do 200 i=1,ni1
            im = i-1
            ij = ndof * ( i-1 + (j-1)*ni1 + (k-1)*ni1*nj1 ) - 1
            IF (K.GT.KTIP) GO TO 123
            IF (I.LE.ITL.OR.I.GT.ITU) GO TO 123
            do 202 l=1,ndof
               row = ij + l
            do 202 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (i .ne. 1) b1(l,m,im,j,k) = zero
               if (k .ne. 1) b3(l,m,i,j,km) = zero
               b4(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
c              call SpSetValue( mat, b2bc(l,m,i,1,k), row, ij+m )
c              call SpSetValue( mat, b2bc(l,m,i,2,k), row,
c    &              ij + ndof*ni1 + m )
               if (i .ne. 1 .and. k .ne. 1 .and.
     &                       i .ne. ni1 .and. k .ne. nk1) then
                  d(l,m,i,j,k) = b2bc(l,m,i,1,k)
                  b5(l,m,i,j,k) = b2bc(l,m,i,2,k)
               else
                  b5(l,m,i,j,k) = zero
               endif
 202        continue
         goto 200
 123        continue
            id = ni+2-i
            ijid = ndof * ( id-1 + (j-1)*ni1 + (k-1)*ni1*nj1 ) - 1
            do 124 l=1,ndof
            row = ij + l
            do 124 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (i .ne. 1) b1(l,m,im,j,k) = zero
               if (k .ne. 1) b3(l,m,i,j,km) = zero
               b4(l,m,i,j,k) = zero
               b5(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
c              call SpSetValue( mat, b2bc(l,m,i,1,k), row, ij+m )
c              call SpSetValue( mat, b2bc(l,m,id,2,k), row, ijid+m )
c
c     Since we're storing only 7 diagonal blocks, there's no space for
c     most of these nonzeros, which fill the diagonals 
c          -ni1+2, -ni1+4, ... -3, -1, 1, 3, ... ni1-4, ni1-2.  
c     For now, we handle these as explicit bc's.
c
               if (totimpl) then
                  if (i .ne. 1 .and. k .ne. 1 .and.
     &                          i .ne. ni1 .and. k .ne. nk1) then
                     d(l,m,i,j,k) = b2bc(l,m,i,1,k)
                     jdiag = row/ndof - (ijid+m)/ndof
                     if (jdiag .eq. ni1*nj1) then
                         b3(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. ni1) then
                         b2(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. 1) then
                         b1(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. 0) then
                         d(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. -1) then
                         b4(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. -ni1) then
                         b5(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else if (jdiag .eq. -ni1*nj1) then
                         b6(l,m,i,j,k) = b2bc(l,m,id,2,k)
                     else 
                  write(20,*) 'jdnew:  i,j,k,l,m,ij,row,col,jdiag ',
     &                      i, j, k, l, m, ij, row, ijid+m, jdiag
                     endif 
                  endif 
               endif
 124        continue
 200     continue

         i=1
         do 300 k=1,nk1
            km = k-1
         do 300 j=1,nj1
            jm = j-1
            do 302 l=1,ndof
            do 302 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               b5(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
               if (j .ne. 1) b2(l,m,i,jm,k) = zero
               if (k .ne. 1) b3(l,m,i,j,km) = zero
c              call SpSetValue( mat, b1bc(l,m,1,j,k), row, ij+m )
c              call SpSetValue( mat, b1bc(l,m,2,j,k), row, ij+ndof+m )
               if (k .ne. 1 .and. j .ne. 1 .and.
     &                       k .ne. nk1 .and. j .ne. nj1) then
                  d(l,m,i,j,k) = b1bc(l,m,1,j,k)
                  b4(l,m,i,j,k) = b1bc(l,m,2,j,k)
               else
                  b4(l,m,i,j,k) = zero
               endif
 302        continue
 300     continue

        endif
      endif

c
c Set interior points (for both implcit and explicit cases)
c
      do 410 k=nk,2,-1
         km = k-1
      do 410 j=nj,2,-1
         jm = j-1
      do 410 i=ni,2,-1
c         scale = dt(i,j,k)
         scale = one
         im = i-1
         do 402 l=1,ndof
         do 402 m=1,ndof
            if (k .gt. 2) then
               b3(l,m,i,j,km) = -b3(l,m,i,j,km)*scale
            else
               b3(l,m,i,j,km) = zero
            endif
            if (j .gt. 2)  then
               b2(l,m,i,jm,k) = -b2(l,m,i,jm,k)*scale
            else
               b2(l,m,i,jm,k) = zero
            endif
            if (i .gt. 2)  then
               b1(l,m,im,j,k) = -b1(l,m,im,j,k)*scale
            else
               b1(l,m,im,j,k) = zero
            endif
            if (i .lt. ni) then
               b4(l,m,i,j,k) = b4(l,m,i,j,k)*scale
            else
               b4(l,m,i,j,k) = zero
            endif
            if (j .lt. nj) then
               b5(l,m,i,j,k) = b5(l,m,i,j,k)*scale
            else
               b5(l,m,i,j,k) = zero
            endif
            if (k .lt. nk) then
               b6(l,m,i,j,k) = b6(l,m,i,j,k)*scale
            else
               b6(l,m,i,j,k) = zero
            endif
 402     continue
 410  continue

      if (bctype .eq. IMPLICIT) then
       if (ifoo .eq. 1) then

         i=ni1
         im = i-1
         do 500 k=1,nk1
            km = k-1
         do 500 j=1,nj1
            jm = j-1
            do 502 l=1,ndof
            do 502 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (j .ne. 1) b2(l,m,i,jm,k) = zero
               if (k .ne. 1) b3(l,m,i,j,km) = zero
               b4(l,m,i,j,k) = zero
               b5(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
c              call SpSetValue( mat, b1bc(l,m,4,j,k), row, ij+m )
c              call SpSetValue( mat, b1bc(l,m,3,j,k), row, ij-ndof+m )
               if (k .ne. 1 .and. j .ne. 1 .and.
     &                       k .ne. nk1 .and. j .ne. nj1) then
                  d(l,m,i,j,k) = b1bc(l,m,4,j,k)
                  b1(l,m,im,j,k) = b1bc(l,m,3,j,k)
               else
                  b1(l,m,im,j,k) = zero
               endif
 502        continue
 500     continue

         j=nj1
         jm = j-1
         do 600 k=1,nk1
            km = k-1
         do 600 i=1,ni1
            im = i-1
            do 602 l=1,ndof
            do 602 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (i .ne. 1) b1(l,m,im,j,k) = zero
               if (k .ne. 1) b3(l,m,i,j,km) = zero
               b4(l,m,i,j,k) = zero
               b5(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
c              call SpSetValue( mat, b2bc(l,m,i,4,k), row, ij+m )
c              call SpSetValue( mat, b2bc(l,m,i,3,k), row, 
c    &              ij - ndof*ni1 + m )
               if (i .ne. 1 .and. k .ne. 1 .and.
     &                       i .ne. ni1 .and. k .ne. nk1) then
                  d(l,m,i,j,k) = b2bc(l,m,i,4,k)
                  b2(l,m,i,jm,k) = b2bc(l,m,i,3,k)
               else
                  b2(l,m,i,jm,k) = zero
               endif
 602        continue
 600     continue

         k=nk1
         km = k-1
         do 700 j=1,nj1
            jm = j-1
         do 700 i=1,ni1
            im = i-1
            do 702 l=1,ndof
            do 702 m=1,ndof
               if (m .eq. l) then
                  d(l,m,i,j,k) = one
               else 
                  d(l,m,i,j,k) = zero
               endif 
               if (i .ne. 1) b1(l,m,im,j,k) = zero
               if (j .ne. 1) b2(l,m,i,jm,k) = zero
               b3(l,m,i,j,km) = zero
               b4(l,m,i,j,k) = zero
               b5(l,m,i,j,k) = zero
               b6(l,m,i,j,k) = zero
c              call SpSetValue( mat, b3bc(l,m,i,j,4), row, ij+m )
c              call SpSetValue( mat, b3bc(l,m,i,j,3), row,
c    &              ij - ndof*ni1*nj1 + m )
               if (i .ne. 1 .and. j .ne. 1 .and.
     &                       i .ne. ni1 .and. j .ne. nj1) then
                  d(l,m,i,j,k) = b3bc(l,m,i,j,4)
                  b3(l,m,i,j,km) = b3bc(l,m,i,j,3)
               else
                  b3(l,m,i,j,km) = zero
               endif
 702        continue
 700     continue

       endif
      endif

      return
      end
