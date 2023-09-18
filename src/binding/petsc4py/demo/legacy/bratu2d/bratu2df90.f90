! file: bratu2df90.f90
! to build a Python module, use this:
! $$ f2py -m bratu2df90 -c bratu2df90.f90

subroutine bratu2d (m, n, alpha, x, f)
  !f2py intent(hide) :: m = shape(x,0)
  !f2py intent(hide) :: n = shape(x,1)
  integer                          :: m, n
  real(kind=8)                     :: alpha
  real(kind=8), intent(in), target :: x(m,n)
  real(kind=8), intent(inout)      :: f(m,n)
  real(kind=8) :: hx, hy
  real(kind=8), pointer, &
       dimension(:,:) :: u, uN, uS, uE, uW
  ! setup 5-points stencil
  u  => x(2:m-1, 2:n-1) ! center
  uN => x(2:m-1, 1:n-2) ! north
  uS => x(2:m-1, 3:n  ) ! south
  uW => x(1:m-2, 2:n-1) ! west
  uE => x(3:m,   2:n-1) ! east
  ! compute nonlinear function
  hx = 1.0/(m-1) ! x grid spacing
  hy = 1.0/(n-1) ! y grid spacing
  f(:,:) = x
  f(2:m-1, 2:n-1) =  &
         (2*u - uE - uW) * (hy/hx) &
       + (2*u - uN - uS) * (hx/hy) &
       - alpha * exp(u)  * (hx*hy)
end subroutine bratu2d
