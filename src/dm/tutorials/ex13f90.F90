program main
!
! This example intends to show how DMDA is used to solve a PDE on a decomposed
! domain. The equation we are solving is not a PDE, but a toy example: van der
! Pol's 2-variable ODE duplicated onto a 3D grid:
! dx/dt = y
! dy/dt = mu(1-x**2)y - x
!
! So we are solving the same equation on all grid points, with no spatial
! dependencies. Still we tell PETSc to communicate (stencil width >0) so we
! have communication between different parts of the domain.
!
! The example is structured so that one can replace the RHS function and
! the forw_euler routine with a suitable RHS and a suitable time-integration
! scheme and make little or no modifications to the DMDA parts. In particular,
! the "inner" parts of the RHS and time-integration do not "know about" the
! decomposed domain.
!
!     See:     http://dx.doi.org/10.6084/m9.figshare.1368581
!
!     Contributed by Aasmund Ervik (asmunder at pvv.org)
!


  use ex13f90aux

#include <petsc/finclude/petscdmda.h>
  use petscdmda

  PetscErrorCode   ierr
  PetscMPIInt      rank,size
  MPI_Comm         comm
  Vec              Lvec,coords
  DM               SolScal,CoordDM
  DMBoundaryType b_x,b_y,b_z
  PetscReal, pointer :: array(:,:,:,:)
  PetscReal :: t,tend,dt,xmin,xmax,ymin,ymax,zmin,zmax,xgmin,xgmax,ygmin,ygmax,zgmin,zgmax
  PetscReal, allocatable :: f(:,:,:,:), grid(:,:,:,:)
  PetscInt :: i,j,k,igmax,jgmax,kgmax,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,itime,maxstep,nscreen,dof,stw,ndim

  ! Fire up PETSc:
  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr .ne. 0) then
    print*,'Unable to initialize PETSc'
    stop
  endif
  comm = PETSC_COMM_WORLD
  call MPI_Comm_rank(comm,rank,ierr);CHKERRA(ierr)
  call MPI_Comm_size(comm,size,ierr);CHKERRA(ierr)
  if (rank == 0) then
    write(*,*) 'Hi! We are solving van der Pol using ',size,' processes.'
    write(*,*) ' '
    write(*,*) '  t     x1         x2'
  endif

  ! Set up the global grid
  igmax = 50
  jgmax = 50
  kgmax = 50
  xgmin = 0.0
  ygmin = 0.0
  zgmin = 0.0
  xgmax = 1.0
  ygmax = 1.0
  zgmax = 1.0
  stw = 1 ! stencil width
  dof = 2 ! number of variables in this DA
  ndim = 3 ! 3D code

  ! Get the BCs and create the DMDA
  call get_boundary_cond(b_x,b_y,b_z);CHKERRA(ierr)
  call DMDACreate3d(comm,b_x,b_y,b_z,DMDA_STENCIL_STAR,igmax,jgmax,kgmax,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stw,  &
                    PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,SolScal,ierr);CHKERRA(ierr)
  call DMSetFromOptions(SolScal,ierr);CHKERRA(ierr)
  call DMSetUp(SolScal,ierr);CHKERRA(ierr)

  ! Set global coordinates, get a global and a local work vector
  call DMDASetUniformCoordinates(SolScal,xgmin,xgmax,ygmin,ygmax,zgmin,zgmax,ierr);CHKERRA(ierr)
  call DMCreateLocalVector(SolScal,Lvec,ierr);CHKERRA(ierr)

  ! Get ib1,imax,ibn etc. of the local grid.
  ! Our convention is:
  ! the first local ghost cell is ib1
  ! the first local       cell is 1
  ! the last  local       cell is imax
  ! the last  local ghost cell is ibn.
  !
  ! i,j,k must be in this call, but are not used
  call DMDAGetCorners(SolScal,i,j,k,imax,jmax,kmax,ierr);CHKERRA(ierr)
  ib1=1-stw
  jb1=1-stw
  kb1=1-stw
  ibn=imax+stw
  jbn=jmax+stw
  kbn=kmax+stw
  allocate(f(dof,ib1:ibn,jb1:jbn,kb1:kbn))
  allocate(grid(ndim,ib1:ibn,jb1:jbn,kb1:kbn))

  ! Get xmin,xmax etc. for the local grid
  ! The "coords" local vector here is borrowed, so we shall not destroy it.
  call DMGetCoordinatesLocal(SolScal,coords,ierr);CHKERRA(ierr)
  ! We need a new DM for coordinate stuff since PETSc supports unstructured grid
  call DMGetCoordinateDM(SolScal,CoordDM,ierr);CHKERRA(ierr)
  ! petsc_to_local and local_to_petsc are convenience functions, see
  ! ex13f90aux.F90.
  call petsc_to_local(CoordDM,coords,array,grid,ndim,stw);CHKERRA(ierr)
  xmin=grid(1,1,1,1)
  ymin=grid(2,1,1,1)
  zmin=grid(3,1,1,1)
  xmax=grid(1,imax,jmax,kmax)
  ymax=grid(2,imax,jmax,kmax)
  zmax=grid(3,imax,jmax,kmax)
  call local_to_petsc(CoordDM,coords,array,grid,ndim,stw);CHKERRA(ierr)

  ! Note that we never use xmin,xmax in this example, but the preceding way of
  ! getting the local xmin,xmax etc. from PETSc for a structured uniform grid
  ! is not documented in any other examples I could find.

  ! Set up the time-stepping
  t = 0.0
  tend = 100.0
  dt = 1e-3
  maxstep=ceiling((tend-t)/dt)
  ! Write output every second (of simulation-time)
  nscreen = int(1.0/dt)+1

  ! Set initial condition
  call DMDAVecGetArrayF90(SolScal,Lvec,array,ierr);CHKERRA(ierr)
  array(0,:,:,:) = 0.5
  array(1,:,:,:) = 0.5
  call DMDAVecRestoreArrayF90(SolScal,Lvec,array,ierr);CHKERRA(ierr)

  ! Initial set-up finished.
  ! Time loop
  maxstep = 5
  do itime=1,maxstep

    ! Communicate such that everyone has the correct values in ghost cells
    call DMLocalToLocalBegin(SolScal,Lvec,INSERT_VALUES,Lvec,ierr);CHKERRA(ierr)
    call DMLocalToLocalEnd(SolScal,Lvec,INSERT_VALUES,Lvec,ierr);CHKERRA(ierr)

    ! Get the old solution from the PETSc data structures
    call petsc_to_local(SolScal,Lvec,array,f,dof,stw);CHKERRA(ierr)

    ! Do the time step
    call forw_euler(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,dof,f,dfdt_vdp)
    t=t+dt

    ! Write result to screen (if main process and it's time to)
    if (rank == 0 .and. mod(itime,nscreen) == 0) then
      write(*,101) t,f(1,1,1,1),f(2,1,1,1)
    endif

    ! Put our new solution in the PETSc data structures
    call local_to_petsc(SolScal,Lvec,array,f,dof,stw)
  end do

  ! Deallocate and finalize
  call DMRestoreLocalVector(SolScal,Lvec,ierr);CHKERRA(ierr)
  call DMDestroy(SolScal,ierr);CHKERRA(ierr)
  deallocate(f)
  deallocate(grid)
  call PetscFinalize(ierr)

  ! Format for writing output to screen
101 format(F5.1,2F11.6)

end program main

!/*TEST
!
!   build:
!     requires: !complex
!     depends:  ex13f90aux.F90
!
!   test:
!     nsize: 4
!
!TEST*/
