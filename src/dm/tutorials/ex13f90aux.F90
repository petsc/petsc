module ex13f90aux
  implicit none
contains
  !
  ! A subroutine which returns the boundary conditions.
  !
  subroutine get_boundary_cond(b_x,b_y,b_z)
#include <petsc/finclude/petscdm.h>
    use petscdm
    DMBoundaryType,intent(inout) :: b_x,b_y,b_z

    ! Here you may set the BC types you want
    b_x = DM_BOUNDARY_GHOSTED
    b_y = DM_BOUNDARY_GHOSTED
    b_z = DM_BOUNDARY_GHOSTED

  end subroutine get_boundary_cond
  !
  ! A function which returns the RHS of the equation we are solving
  !
  function dfdt_vdp(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n,f)
    !
    ! Right-hand side for the van der Pol oscillator.  Very simple system of two
    ! ODEs.  See Iserles, eq (5.2).
    !
    PetscReal, intent(in) :: t,dt
    PetscInt, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n
    PetscReal, dimension(n,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: f
    PetscReal, dimension(n,imax,jmax,kmax) :: dfdt_vdp
    PetscReal, parameter :: mu=1.4, one=1.0
    !
    dfdt_vdp(1,:,:,:) = f(2,1,1,1)
    dfdt_vdp(2,:,:,:) = mu*(one - f(1,1,1,1)**2)*f(2,1,1,1) - f(1,1,1,1)
  end function dfdt_vdp
  !
  ! The standard Forward Euler time-stepping method.
  !
  recursive subroutine forw_euler(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,neq,y,dfdt)
    PetscReal, intent(in) :: t,dt
    PetscInt, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,neq
    PetscReal, dimension(neq,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: y
    !
    ! Define the right-hand side function
    !
    interface
      function dfdt(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n,f)
        PetscReal, intent(in) :: t,dt
        PetscInt, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n
        PetscReal, dimension(n,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: f
        PetscReal, dimension(n,imax,jmax,kmax) :: dfdt
      end function dfdt
    end interface
    !--------------------------------------------------------------------------
    !
    y(:,1:imax,1:jmax,1:kmax) = y(:,1:imax,1:jmax,1:kmax)  + dt*dfdt(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,neq,y)
  end subroutine forw_euler
  !
  ! The following 4 subroutines handle the mapping of coordinates. I'll explain
  ! this in detail:
  !    PETSc gives you local arrays which are indexed using the global indices.
  ! This is probably handy in some cases, but when you are re-writing an
  ! existing serial code and want to use DMDAs, you have tons of loops going
  ! from 1 to imax etc. that you don't want to change.
  !    These subroutines re-map the arrays so that all the local arrays go from
  ! 1 to the (local) imax.
  !
  subroutine petsc_to_local(da,vec,array,f,dof,stw)
    use petscdmda
    DM                                                            :: da
    Vec,intent(in)                                                :: vec
    PetscReal, pointer                                            :: array(:,:,:,:)
    PetscInt,intent(in)                                           :: dof,stw
    PetscReal, intent(inout), dimension(:,1-stw:,1-stw:,1-stw:) :: f
    PetscErrorCode                                                :: ierr
    !
    call DMDAVecGetArrayF90(da,vec,array,ierr);PetscCall(ierr);
    call transform_petsc_us(array,f,stw)
  end subroutine petsc_to_local
  subroutine transform_petsc_us(array,f,stw)
    !Note: this assumed shape-array is what does the "coordinate transformation"
    PetscInt,intent(in)                                   :: stw
    PetscReal, intent(in), dimension(:,1-stw:,1-stw:,1-stw:)  :: array
    PetscReal,intent(inout),dimension(:,1-stw:,1-stw:,1-stw:) :: f
    f(:,:,:,:) = array(:,:,:,:)
  end subroutine transform_petsc_us
  subroutine local_to_petsc(da,vec,array,f,dof,stw)
    use petscdmda
    DM                                                    :: da
    Vec,intent(inout)                                     :: vec
    PetscReal, pointer                                    :: array(:,:,:,:)
    PetscInt,intent(in)                                    :: dof,stw
    PetscReal,intent(inout),dimension(:,1-stw:,1-stw:,1-stw:)  :: f
    PetscErrorCode                                        :: ierr
    call transform_us_petsc(array,f,stw)
    call DMDAVecRestoreArrayF90(da,vec,array,ierr);PetscCall(ierr);
  end subroutine local_to_petsc
  subroutine transform_us_petsc(array,f,stw)
    !Note: this assumed shape-array is what does the "coordinate transformation"
    PetscInt,intent(in)                                     :: stw
    PetscReal, intent(inout), dimension(:,1-stw:,1-stw:,1-stw:) :: array
    PetscReal, intent(in),dimension(:,1-stw:,1-stw:,1-stw:)      :: f
    array(:,:,:,:) = f(:,:,:,:)
  end subroutine transform_us_petsc
end module ex13f90aux
