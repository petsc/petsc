!
!
      module junk
      type abc
      integer a
      double precision,pointer :: b(:)
      integer c
      end type
      end module

      program main 
      use junk

      type (abc) x

      ALLOCATE(x%b(5))

      x%a = 1
      x%b(1) = 11.0
      x%c = 111

      call c_routine(x)

      write (6,100) x%a, x%b(1), x%c
 100  format("From Fortran Main:",i3," ",1pe8.2," ",i3)

      DEALLOCATE(x%b)

      end program


      subroutine fortran_routine(x)
      use junk

      type (abc) x

      write (6,110) x%a, x%b(1), x%c
 110  format("From Fortran routine called by C:",i3," ",1pe8.2," ",i3)
      x%a = 3
      x%b(1) = 33.0
      x%c = 333
      end

