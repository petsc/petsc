c
c  "$Id: mat.F,v 1.17 1997/10/11 18:39:18 curfman Exp curfman $";
c
c ----------------------------------------------------------------------
c
      integer function rbuild_direct_euler( f, sctype, dt, dxx )
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
      integer i,j,k,jk,sctype,ijkx

      rbuild_direct_euler = 0
      if ((bctype .ne. EXPLICIT) .and. (bctype .ne. IMPLICIT)) then
        if (rank .eq. 0) write(6,*) 
     &       'rbuild_direct_euler: invalid value of bctype flag!'
        rbuild_direct_euler = -1
        return
      endif
      if (sctype .ne. DT_MULT .and. sctype .ne. DT_DIV) then
        if (rank .eq. 0) 
     &    write(6,*) 'rbuild_direct: invalid value of sctype flag!'
        rbuild_direct_euler = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=1, j=1, k=1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c      if (bctype .eq. IMPLICIT) then

c      if (zsf1 .eq. 1) then
c        k=1
c        do 100 j=ysf2,yef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 100 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcrk1(i,j) 
c           f(ijkx+1) = fbcruk1(i,j)
c           f(ijkx+2) = fbcrvk1(i,j)
c           f(ijkx+3) = fbcrwk1(i,j)
c           f(ijkx+4) = fbcek1(i,j)
c 100    continue
c      endif

c      if (ysf1 .eq. 1) then
c         j=1
c         do 200 k=zsf2,zef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c         do 200 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcrj1(i,k)
c           f(ijkx+1) = fbcruj1(i,k)
c           f(ijkx+2) = fbcrvj1(i,k)
c           f(ijkx+3) = fbcrwj1(i,k)
c           f(ijkx+4) = fbcej1(i,k)
c 200    continue
c      endif

c      if (xsf1 .eq. 1) then
c        i=1
c        do 300 k=zsf2,zef01
c           ik = i-xsf + (k-zsf)*xm*ym
c        do 300 j=ysf2,yef01
c           ijkx = ndof * (ik + (j-ysf)*xm)
c           f(ijkx)   = fbcri1(j,k)
c           f(ijkx+1) = fbcrui1(j,k)
c           f(ijkx+2) = fbcrvi1(j,k)
c           f(ijkx+3) = fbcrwi1(j,k)
c           f(ijkx+4) = fbcei1(j,k)
c 300     continue
c      endif

c      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (sctype .eq. DT_MULT) then
         do 400 k=zsf2,zef01
         do 400 j=ysf2,yef01
             jk = (j-ysf)*xm + (k-zsf)*xm*ym
         do 400 i=xsf2,xef01
             ijkx      = ndof * (i-xsf + jk)
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
             ijkx   = ndof * (i-xsf + jk)
             f(ijkx)   = dr(i,j,k)
             f(ijkx+1) = dru(i,j,k)
             f(ijkx+2) = drv(i,j,k)
             f(ijkx+3) = drw(i,j,k)
             f(ijkx+4) = de(i,j,k)
 450     continue
      else
        if (rank .eq. 0) write(6,*) 'rbuild_direct: invalid sctype!'
        rbuild_direct_euler = -1
        return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Implicit boundary conditions for i=ni1, j=nj1, k=nk1
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c      if (bctype .eq. IMPLICIT) then

c      if (xef01 .eq. ni) then
c        i=ni1
c        do 500 k=zsf2,zef01
c           ik = i-xsf + (k-zsf)*xm*ym
c        do 500 j=ysf2,yef01
c           ijkx = ndof * (ik + (j-ysf)*xm)
c           f(ijkx)   = fbcri2(j,k)
c           f(ijkx+1) = fbcrui2(j,k)
c           f(ijkx+2) = fbcrvi2(j,k)
c           f(ijkx+3) = fbcrwi2(j,k)
c           f(ijkx+4) = fbcei2(j,k)
c 500    continue
c      endif

c      if (yef01 .eq. nj) then
c        j=nj1
c        do 600 k=zsf2,zef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 600 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcrj2(i,k)
c           f(ijkx+1) = fbcruj2(i,k)
c           f(ijkx+2) = fbcrvj2(i,k)
c           f(ijkx+3) = fbcrwj2(i,k)
c           f(ijkx+4) = fbcej2(i,k)
c 600    continue
c      endif

c      if (zef01 .eq. nk_boundary) then
c        k=nk1_boundary
c        do 700 j=ysf2,yef01
c           jk = (j-ysf)*xm + (k-zsf)*xm*ym
c        do 700 i=xsf2,xef01
c           ijkx = ndof * (i-xsf + jk)
c           f(ijkx)   = fbcrk2(i,j)
c           f(ijkx+1) = fbcruk2(i,j)
c           f(ijkx+2) = fbcrvk2(i,j)
c           f(ijkx+3) = fbcrwk2(i,j)
c           f(ijkx+4) = fbcek2(i,j)
c 700    continue
c      endif
c      endif

      return
      end

c -------------------------------------------------------------------

      integer function nzmat(format,mmtype,ndof_ee,ndof_block,istart,
     &                   iend,is1,ltog,nloc,wkdim,nnz_d,nnz_o,ao)
c
c  nnzmat - Determines nonzero structure of the (parallel) Jacobian
c           matrix in the arrays nnz_d and nnz_o.  These arrays can
c           then be passed to the PETSc MatCreateXXX() routines to
c           preallocate matrix memory. This routine supports several
c           matrix formats.  See the PETSc users manual for info about
c           the use of these arrays for matrix memory preallocation.
c
c  Input Parameters:
c    format       - PETSc matrix format
c    mmtype       - type of multi-model (Euler, FP, or hybrid)
c    ndof         - Dof per node
c    ndof_block   - block size (ndof, except when experimenting with
c                   block size = 1)
c    is1          - index set with PETSc ordering numbers for certain
c                   boundary nodes
c    bctype       - type of boundary condition formulation
c    istart, iend - starting and ending owned rows of matrix,
c                   as determined by the distributed array
c    ltog         - local-to-global mapping for distributed arrays
c    nloc         - number of elements in ltog mapping
c    wkdim        - length of arrays nnz_d and nnz_o
c    nnz_d        - array for diagonal submatrices
c    nnz_o        - array for off-diagonal submatrices
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

      integer ijk,ijkn,ndof_ee,istart,iend,format,ijk_noghosts
      integer b1c,b2c,b3c,b4c,b5c,b6c,i,j,k,l,wkdim,ndof_block
      integer col,ict,row,id,ijkid,is1(0:*),ierr,ao,m, mmtype
      integer nloc,ltog(0:nloc-1),nnz_d(0:wkdim-1),nnz_o(0:wkdim-1)
      integer chunk_loc,chunk_alc

      nzmat = 0

c  The same code supports both block and point storage formats
c  through the chunk variables.
      if (format .eq. MATMPIAIJ .or. format .eq. MATSEQAIJ
     &         .or. format .eq. MATMPIROWBS) then
        chunk_alc = ndof_ee
        chunk_loc = ndof
      else if (format .eq. MATMPIBAIJ .or. format .eq. MATSEQBAIJ) then
        if (ndof_block .eq. ndof_ee .or. ndof_block .eq. ndof) then
           chunk_alc = 1
           chunk_loc = 1
        else if (ndof_block .eq. 1) then
           chunk_alc = ndof_ee
           chunk_loc = ndof
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

C  Set nonzeros for full potential model; just dummy values for now.
      if (mmtype .eq. MMFP) then
         do 5 i=0,wkdim-1
            nnz_d(i) = 1
 5       continue
         return
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Explicit boundary conditions:
c   - Use only interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (bctype .eq. EXPLICIT) then

      ict = 0
      do 30 k=zsf2,zef01
      do 30 j=ysf2,yef01
      do 30 i=xsf2,xef01
         ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         b3c = ijk - ndof*gxm*gym
         b2c = ijk - ndof*gxm
         b1c = ijk - ndof
         b4c = ijk + ndof
         b5c = ijk + ndof*gxm
         b6c = ijk + ndof*gxm*gym
         do 20 l=1,chunk_alc
            if (k .gt. 2) then
               col = ltog(b3c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
               endif
            endif
            if (j .gt. 2)  then
               col = ltog(b2c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
               endif
            endif
            if (i .gt. 2)  then
               col = ltog(b1c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
               endif
            endif
            col = ltog(ijk+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(ict) = nnz_d(ict) + chunk_alc
            else
               nnz_o(ict) = nnz_o(ict) + chunk_alc
            endif
            if (i .lt. ni) then
               col = ltog(b4c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
               endif
            endif
            if (j .lt. nj) then
               col = ltog(b5c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
               endif
            endif
            if (k .lt. nk) then
               col = ltog(b6c+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(ict) = nnz_d(ict) + chunk_alc
               else
                  nnz_o(ict) = nnz_o(ict) + chunk_alc
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
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijkn = ijk + ndof*gxm*gym
c           ijkn = upper neighbor (z=2)
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            do 102 l=1,chunk_alc
               row = ijk_noghosts+l
               col = ltog(ijk+l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 102        continue
 100     continue
      endif

      if (ysf1 .eq. 1) then
         j=1
         ict = 0
         do 200 k=zsf2,zef01
         do 200 i=xsf2,xef01
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk + ndof*gxm
c           ijkn = northern neighbor (j=2)
            IF (K.GT.KTIP) GO TO 123
            IF (I.LE.ITL.OR.I.GT.ITU) GO TO 123
            do 202 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 202        continue
         goto 200
 123        continue
c
            ID=NI+1-I
c            ID=NI+2-I
c Note: This j value corresponds to j=2!!
            ijkid = ndof * ( id-1 + j*ni1 + (k-1)*ni1*nj1 ) - 1
            do 124 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
c Note: We cannot call AOApplicationToPetsc() here, since this
c       parallel database routine MUST be called by all processors.
c       Luckily, we have already determined these column numbers
c       in the index set is1 when setting up the vector scatters,
c       so we can just use them here.
c              uniprocessor: col = ijkid + l
            do 124 m=1,ndof
               col = ijkid + m
               call AOApplicationToPetsc(ao,1,col,ierr)
c               col1 = is1(ict)
c               if (col .ne. col1)
c     &           write(6,*) 'ALLOC: k,j,i,ict,col, col1 = ',
c     &                       k,j,i,ict,col,col1
               if (nowake .eq. 1) goto 122
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + 1
               else
                  nnz_o(row) = nnz_o(row) + 1
               endif
 122           continue
               ict = ict + 1
 124        continue
 200     continue
      endif

      if (xsf1 .eq. 1) then
         i=1
         do 300 k=zsf2,zef01
         do 300 j=ysf2,yef01
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk + ndof
c           ijkn = eastern neighbor (i=2)
            do 302 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 302        continue
 300     continue
      endif

c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c  Interior grid points i=2,ni; j=2,nj; k=2,nk
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      do 410 k=zsf2,zef01
      do 410 j=ysf2,yef01
      do 410 i=xsf2,xef01
         ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &              (k-zsf)*xm*ym) - 1
         b2c = ijk - ndof*gxm
         b3c = ijk - ndof*gxm*gym
         b5c = ijk + ndof*gxm
         b6c = ijk + ndof*gxm*gym
         b1c = ijk - ndof
         b4c = ijk + ndof
         do 402 l=1,chunk_alc
            row = ijk_noghosts + l
            col = ltog(b3c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(b2c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(b1c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(ijk+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(b4c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(b5c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
            endif
            col = ltog(b6c+l)
            if (col .ge. istart .and. col .lt. iend) then
               nnz_d(row) = nnz_d(row) + chunk_alc
            else
               nnz_o(row) = nnz_o(row) + chunk_alc
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
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - ndof
c           ijkn = western neighbor (i=ni)
            do 502 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 502        continue
 500     continue
      endif

      if (yef01 .eq. nj) then
         j=nj1
         do 600 k=zsf2,zef01
         do 600 i=xsf2,xef01
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - ndof*gxm
c           ijkn = southern neighbor (k=nj)
            do 602 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 602        continue
 600     continue
      endif

      if (zef01 .eq. nk_boundary) then
         k=nk1_boundary
         do 700 j=ysf2,yef01
         do 700 i=xsf2,xef01
            ijk = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &                 (k-gzsf)*gxm*gym) - 1
            ijk_noghosts = chunk_loc * ((i-xsf) + (j-ysf)*xm + 
     &                 (k-zsf)*xm*ym) - 1
            ijkn = ijk - ndof*gxm*gym
c           ijkn = lower neighbor (k=nk)
            do 702 l=1,chunk_alc
               row = ijk_noghosts + l
               col = ltog(ijk + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
               col = ltog(ijkn + l)
               if (col .ge. istart .and. col .lt. iend) then
                  nnz_d(row) = nnz_d(row) + chunk_alc
               else
                  nnz_o(row) = nnz_o(row) + chunk_alc
               endif
 702        continue
 700     continue
      endif

C  Remaining rows have just 1 nonzero on diagonal (these correspond to
C  nodes on the edges of the 3D domain).

      do 90 i=0,wkdim-1
         if (nnz_d(i) .eq. 0) nnz_d(i) = 1
c         write(6,*) i,nnz_d(i)
 90   continue

      endif

      return
      end
