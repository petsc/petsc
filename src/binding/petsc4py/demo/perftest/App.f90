subroutine formFunction_C(nx, ny, nz, h, t, x, xdot, f) &
     bind(C, name="formFunction")
  use ISO_C_BINDING, only: C_INT, C_DOUBLE
  implicit none
  integer(kind=C_INT), intent(in)    :: nx, ny, nz
  real(kind=C_DOUBLE), intent(in)    :: h(3), t
  real(kind=C_DOUBLE), intent(in)    :: x(nx,ny,nz), xdot(nx,ny,nz)
  real(kind=C_DOUBLE), intent(inout) :: f(nx,ny,nz)
  call formfunction_f(nx, ny, nz, h, t, x, xdot, f)
end subroutine formFunction_C

subroutine formInitial_C(nx, ny, nz, h, t, x) &
     bind(C, name="formInitial")
  use ISO_C_BINDING, only: C_INT, C_DOUBLE
  implicit none
  integer(kind=C_INT), intent(in)    :: nx, ny, nz
  real(kind=C_DOUBLE), intent(in)    :: h(3), t
  real(kind=C_DOUBLE), intent(inout) :: x(nx,ny,nz)
  call forminitial_f(nx, ny, nz, h, t, x)
end subroutine formInitial_C

subroutine evalK (P, K)
  real(kind=8), intent(in)  :: P
  real(kind=8), intent(out) :: K
  if (P >= 0.0) then
     K = 1.0
  else
     K = 1.0/(1+P**2)
  end if
end subroutine evalK

subroutine fillK (P, K)
  real(kind=8), intent(in)  :: P(-1:1)
  real(kind=8), intent(out) :: K(-1:1)
  real(kind=8)  Ka, Kb
  call evalK((P(-1)+P( 0))/2.0, Ka)
  call evalK((P( 0)+P( 1))/2.0, Kb)
  K(-1) = -Ka
  K( 0) = Ka + Kb
  K(+1) = -Kb
end subroutine fillK

subroutine forminitial_f(nx, ny, nz, h, t, x)
  implicit none
  integer, intent(in)         :: nx, ny, nz
  real(kind=8), intent(in)    :: h(3), t
  real(kind=8), intent(inout) :: x(nx,ny,nz)
  !
  x(:,:,:) = 0.0
end subroutine forminitial_f

subroutine formfunction_f(nx, ny, nz, h, t, x, xdot, f)
  implicit none
  integer, intent(in)         :: nx, ny, nz
  real(kind=8), intent(in)    :: h(3), t
  real(kind=8), intent(in)    :: x(nx,ny,nz), xdot(nx,ny,nz)
  real(kind=8), intent(inout) :: f(nx,ny,nz)
  !
  integer      :: i,j,k,ii(-1:1),jj(-1:1),kk(-1:1)
  real(kind=8) :: K1(-1:1), K2(-1:1), K3(-1:1)
  real(kind=8) :: P1(-1:1), P2(-1:1), P3(-1:1)
  !
  do k=1,nz
     do j=1,ny
        do i=1,nx
           !
           ii = (/i-1, i, i+1/)
           if (i == 1) then
              ii(-1) = i
           else if (i == nx) then
              ii(+1) = i
           endif
           !
           jj = (/j-1, j, j+1/)
           if (j == 1) then
              jj(-1) = j
           else if (j == ny) then
              jj(+1) = j
           end if
           !
           kk = (/k-1, k, k+1/)
           if (k == 1) then
              kk(-1) = k
           else if (k == nz) then
              kk(+1) = k
           end if
           !
           P1 = x(ii,j,k)
           P2 = x(i,jj,k)
           P3 = x(i,j,kk)
           call fillK(P1,K1)
           call fillK(P2,K2)
           call fillK(P3,K3)
           f(i,j,k) =                &
                xdot(i,j,k)        + &
                sum(K1*P1)/h(1)**2 + &
                sum(K2*P2)/h(2)**2 + &
                sum(K3*P3)/h(3)**2
           !
        end do !i
     end do !j
  end do !k
  !
  i = nx/4+1
  j = ny/4+1
  k = nz/2+1
  f(i,j,k:nz) = f(i,j,k:nz) + 300.0
  !
end subroutine formfunction_f
