      integer function mpir_iargc()
C
      use f90_unix

      mpir_iargc = iargc()
      return
      end
c     
      subroutine mpir_getarg( i, s )
C
      use f90_unix

      integer       i
      character*(*) s
      call getarg(i,s)
      return
      end
