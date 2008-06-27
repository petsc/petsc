! file: del2lib.f90

! to build a Python module, use this:
! $$ f2py -m del2lib -c del2lib.f90

subroutine del2apply (n, F, x, y)

  !f2py intent(hide) :: n=shape(F,0)-2
  integer      :: n
  real(kind=8) :: F(0:n+1,0:n+1,0:n+1)
  real(kind=8) :: x(n,n,n)
  real(kind=8) :: y(n,n,n)

  F(1:n,1:n,1:n) = x

  y(:,:,:) = 6.0 * F(1:n,1:n,1:n) &
           - F(0:n-1,1:n,1:n)     &
           - F(2:n+1,1:n,1:n)     &
           - F(1:n,0:n-1,1:n)     &
           - F(1:n,2:n+1,1:n)     &
           - F(1:n,1:n,0:n-1)     &
           - F(1:n,1:n,2:n+1)

end subroutine del2apply
