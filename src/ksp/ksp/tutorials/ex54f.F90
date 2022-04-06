!
!   Description: Solve Ax=b.  A comes from an anisotropic 2D thermal problem with Q1 FEM on domain (-1,1)^2.
!       Material conductivity given by tensor:
!
!       D = | 1 0       |
!           | 0 epsilon |
!
!    rotated by angle 'theta' (-theta <90> in degrees) with anisotropic parameter 'epsilon' (-epsilon <0.0>).
!    Blob right hand side centered at C (-blob_center C(1),C(2) <0,0>)
!    Dirichlet BCs on y=-1 face.
!
!    -out_matlab will generate binary files for A,x,b and a ex54f.m file that reads them and plots them in matlab.
!
!    User can change anisotropic shape with function ex54_psi().  Negative theta will switch to a circular anisotropy.
!

! -----------------------------------------------------------------------
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

      Vec              xvec,bvec,uvec
      Mat              Amat
      KSP              ksp
      PetscErrorCode   ierr
      PetscViewer viewer
      PetscInt qj,qi,ne,M,Istart,Iend,geq
      PetscInt ki,kj,nel,ll,j1,i1,ndf,f4
      PetscInt f2,f9,f6,one
      PetscInt :: idx(4)
      PetscBool  flg,out_matlab
      PetscMPIInt size,rank
      PetscScalar::ss(4,4),val
      PetscReal::shp(3,9),sg(3,9)
      PetscReal::thk,a1,a2
      PetscReal, external :: ex54_psi
      PetscReal::theta,eps,h,x,y,xsj
      PetscReal::coord(2,4),dd(2,2),ev(3),blb(2)

      common /ex54_theta/ theta
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      one = 1
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 set parameters
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      f4 = 4
      f2 = 2
      f9 = 9
      f6 = 6
      ne = 9
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-ne',ne,flg,ierr)
      h = 2.0/real(ne)
      M = (ne+1)*(ne+1)
      theta = 90.0
!     theta is input in degrees
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-theta',theta,flg,ierr)
      theta = theta / 57.2957795
      eps = 1.0
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-epsilon',eps,flg,ierr)
      ki = 2
      call PetscOptionsGetRealArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-blob_center',blb,ki,flg,ierr)
      if (.not. flg) then
         blb(1) = 0.0
         blb(2) = 0.0
      else if (ki .ne. 2) then
         print *, 'error: ', ki,' arguments read for -blob_center.  Needs to be two.'
      endif
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-out_matlab',out_matlab,flg,ierr)
      if (.not.flg) out_matlab = PETSC_FALSE;

      ev(1) = 1.0
      ev(2) = eps*ev(1)
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Compute the matrix and right-hand-side vector that define
!     the linear system, Ax = b.
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix.  When using MatCreate(), the matrix format can
!  be specified at runtime.
      call MatCreate(PETSC_COMM_WORLD,Amat,ierr)
      call MatSetSizes( Amat,PETSC_DECIDE, PETSC_DECIDE, M, M, ierr)
      call MatSetType( Amat, MATAIJ, ierr)
      call MatSetOption(Amat,MAT_SPD,PETSC_TRUE,ierr)
      if (size == 1) then
         call MatSetType( Amat, MATAIJ, ierr)
      else
         call MatSetType( Amat, MATMPIAIJ, ierr)
      endif
      call MatMPIAIJSetPreallocation(Amat,f9,PETSC_NULL_INTEGER,f6,PETSC_NULL_INTEGER, ierr)
      call MatSetFromOptions( Amat, ierr)
      call MatSetUp( Amat, ierr)
      call MatGetOwnershipRange( Amat, Istart, Iend, ierr)
!  Create vectors.  Note that we form 1 vector from scratch and
!  then duplicate as needed.
      call MatCreateVecs( Amat, PETSC_NULL_VEC, xvec, ierr)
      call VecSetFromOptions( xvec, ierr)
      call VecDuplicate( xvec, bvec, ierr)
      call VecDuplicate( xvec, uvec, ierr)
!  Assemble matrix.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array "col").
      thk = 1.0              ! thickness
      nel = 4                   ! nodes per element (quad)
      ndf = 1
      call int2d(f2,sg)
      do geq=Istart,Iend-1,1
         qj = geq/(ne+1); qi = mod(geq,(ne+1))
         x = h*qi - 1.0; y = h*qj - 1.0 ! lower left corner (-1,-1)
         if (qi < ne .and. qj < ne) then
            coord(1,1) = x;   coord(2,1) = y
            coord(1,2) = x+h; coord(2,2) = y
            coord(1,3) = x+h; coord(2,3) = y+h
            coord(1,4) = x;   coord(2,4) = y+h
! form stiff
            ss = 0.0
            do ll = 1,4
               call shp2dquad(sg(1,ll),sg(2,ll),coord,shp,xsj,f2)
               xsj = xsj*sg(3,ll)*thk
               call thfx2d(ev,coord,shp,dd,f2,f2,f4,ex54_psi)
               j1 = 1
               do kj = 1,nel
                  a1 = (dd(1,1)*shp(1,kj) + dd(1,2)*shp(2,kj))*xsj
                  a2 = (dd(2,1)*shp(1,kj) + dd(2,2)*shp(2,kj))*xsj
!     Compute residual
!                  p(j1) = p(j1) - a1*gradt(1) - a2*gradt(2)
!     Compute tangent
                  i1 = 1
                  do ki = 1,nel
                     ss(i1,j1) = ss(i1,j1) + a1*shp(1,ki) + a2*shp(2,ki)
                     i1 = i1 + ndf
                  end do
                  j1 = j1 + ndf
               end do
            enddo

            idx(1) = geq; idx(2) = geq+1; idx(3) = geq+(ne+1)+1
            idx(4) = geq+(ne+1)
            if (qj > 0) then
               call MatSetValues(Amat,f4,idx,f4,idx,ss,ADD_VALUES,ierr)
            else                !     a BC
               do ki=1,4,1
                  do kj=1,4,1
                     if (ki<3 .or. kj<3) then
                        if (ki==kj) then
                           ss(ki,kj) = .1*ss(ki,kj)
                        else
                           ss(ki,kj) = 0.0
                        endif
                     endif
                  enddo
               enddo
               call MatSetValues(Amat,f4,idx,f4,idx,ss,ADD_VALUES,ierr)
            endif               ! BC
         endif                  ! add element
         if (qj > 0) then      ! set rhs
            val = h*h*exp(-100*((x+h/2)-blb(1))**2)*exp(-100*((y+h/2)-blb(2))**2)
            call VecSetValues(bvec,one,geq,val,INSERT_VALUES,ierr)
         endif
      enddo
      call MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY,ierr)
      call VecAssemblyBegin(bvec,ierr)
      call VecAssemblyEnd(bvec,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!          Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      call KSPSetOperators(ksp,Amat,Amat,ierr)

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

      call KSPSetFromOptions(ksp,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call KSPSolve(ksp,bvec,xvec,ierr)
      CHKERRA(ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      output
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (out_matlab) then
         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Amat',FILE_MODE_WRITE,viewer,ierr)
         call MatView(Amat,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Bvec',FILE_MODE_WRITE,viewer,ierr)
         call VecView(bvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Xvec',FILE_MODE_WRITE,viewer,ierr)
         call VecView(xvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call MatMult(Amat,xvec,uvec,ierr)
         val = -1.0
         call VecAXPY(uvec,val,bvec,ierr)
         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Rvec',FILE_MODE_WRITE,viewer,ierr)
         call VecView(uvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         if (rank == 0) then
            open(1,file='ex54f.m', FORM='formatted')
            write (1,*) 'A = PetscBinaryRead(''Amat'');'
            write (1,*) '[m n] = size(A);'
            write (1,*) 'mm = sqrt(m);'
            write (1,*) 'b = PetscBinaryRead(''Bvec'');'
            write (1,*) 'x = PetscBinaryRead(''Xvec'');'
            write (1,*) 'r = PetscBinaryRead(''Rvec'');'
            write (1,*) 'bb = reshape(b,mm,mm);'
            write (1,*) 'xx = reshape(x,mm,mm);'
            write (1,*) 'rr = reshape(r,mm,mm);'
!            write (1,*) 'imagesc(bb')'
!            write (1,*) 'title('RHS'),'
            write (1,*) 'figure,'
            write (1,*) 'imagesc(xx'')'
            write (1,2002) eps,theta*57.2957795
            write (1,*) 'figure,'
            write (1,*) 'imagesc(rr'')'
            write (1,*) 'title(''Residual''),'
            close(1)
         endif
      endif
 2002 format('title(''Solution: esp='',d9.3,'', theta='',g8.3,''),')
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(xvec,ierr)
      call VecDestroy(bvec,ierr)
      call VecDestroy(uvec,ierr)
      call MatDestroy(Amat,ierr)
      call KSPDestroy(ksp,ierr)
      call PetscFinalize(ierr)

      end

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     thfx2d - compute material tensor
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Compute thermal gradient and flux

      subroutine thfx2d(ev,xl,shp,dd,ndm,ndf,nel,dir)
      implicit  none

      PetscInt   ndm,ndf,nel,i
      PetscReal ev(2),xl(ndm,nel),shp(3,*),dir
      PetscReal xx,yy,psi,cs,sn,c2,s2,dd(2,2)

      xx       = 0.0
      yy       = 0.0
      do i = 1,nel
        xx       = xx       + shp(3,i)*xl(1,i)
        yy       = yy       + shp(3,i)*xl(2,i)
      end do
      psi = dir(xx,yy)
!     Compute thermal flux
      cs  = cos(psi)
      sn  = sin(psi)
      c2  = cs*cs
      s2  = sn*sn
      cs  = cs*sn

      dd(1,1) = c2*ev(1) + s2*ev(2)
      dd(2,2) = s2*ev(1) + c2*ev(2)
      dd(1,2) = cs*(ev(1) - ev(2))
      dd(2,1) = dd(1,2)

!      flux(1) = -dd(1,1)*gradt(1) - dd(1,2)*gradt(2)
!      flux(2) = -dd(2,1)*gradt(1) - dd(2,2)*gradt(2)

      end

!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     shp2dquad - shape functions - compute derivatives w/r natural coords.
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       subroutine shp2dquad(s,t,xl,shp,xsj,ndm)
!-----[--.----+----.----+----.-----------------------------------------]
!      Purpose: Shape function routine for 4-node isoparametric quads
!
!      Inputs:
!         s,t       - Natural coordinates of point
!         xl(ndm,*) - Nodal coordinates for element
!         ndm       - Spatial dimension of mesh

!      Outputs:
!         shp(3,*)  - Shape functions and derivatives at point
!                     shp(1,i) = dN_i/dx  or dN_i/dxi_1
!                     shp(2,i) = dN_i/dy  or dN_i/dxi_2
!                     shp(3,i) = N_i
!         xsj       - Jacobian determinant at point
!-----[--.----+----.----+----.-----------------------------------------]
      implicit  none
      PetscInt  ndm
      PetscReal xo,xs,xt, yo,ys,yt, xsm,xsp,xtm
      PetscReal xtp, ysm,ysp,ytm,ytp
      PetscReal s,t, xsj,xsj1, sh,th,sp,tp,sm
      PetscReal tm, xl(ndm,4),shp(3,4)

!     Set up interpolations

      sh = 0.5*s
      th = 0.5*t
      sp = 0.5 + sh
      tp = 0.5 + th
      sm = 0.5 - sh
      tm = 0.5 - th
      shp(3,1) =   sm*tm
      shp(3,2) =   sp*tm
      shp(3,3) =   sp*tp
      shp(3,4) =   sm*tp

!     Set up natural coordinate functions (times 4)

      xo =  xl(1,1)-xl(1,2)+xl(1,3)-xl(1,4)
      xs = -xl(1,1)+xl(1,2)+xl(1,3)-xl(1,4) + xo*t
      xt = -xl(1,1)-xl(1,2)+xl(1,3)+xl(1,4) + xo*s
      yo =  xl(2,1)-xl(2,2)+xl(2,3)-xl(2,4)
      ys = -xl(2,1)+xl(2,2)+xl(2,3)-xl(2,4) + yo*t
      yt = -xl(2,1)-xl(2,2)+xl(2,3)+xl(2,4) + yo*s

!     Compute jacobian (times 16)

      xsj1 = xs*yt - xt*ys

!     Divide jacobian by 16 (multiply by .0625)

      xsj = 0.0625*xsj1
      if (xsj1.eq.0.0) then
         xsj1 = 1.0
      else
         xsj1 = 1.0/xsj1
      endif

!     Divide functions by jacobian

      xs  = (xs+xs)*xsj1
      xt  = (xt+xt)*xsj1
      ys  = (ys+ys)*xsj1
      yt  = (yt+yt)*xsj1

!     Multiply by interpolations

      ytm =  yt*tm
      ysm =  ys*sm
      ytp =  yt*tp
      ysp =  ys*sp
      xtm =  xt*tm
      xsm =  xs*sm
      xtp =  xt*tp
      xsp =  xs*sp

!     Compute shape functions

      shp(1,1) = - ytm+ysm
      shp(1,2) =   ytm+ysp
      shp(1,3) =   ytp-ysp
      shp(1,4) = - ytp-ysm
      shp(2,1) =   xtm-xsm
      shp(2,2) = - xtm-xsp
      shp(2,3) = - xtp+xsp
      shp(2,4) =   xtp+xsm

      end

!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     int2d
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      subroutine int2d(l,sg)
!-----[--.----+----.----+----.-----------------------------------------]
!     Purpose: Form Gauss points and weights for two dimensions

!     Inputs:
!     l       - Number of points/direction

!     Outputs:
!     sg(3,*) - Array of points and weights
!-----[--.----+----.----+----.-----------------------------------------]
      implicit  none
      PetscInt   l,i,lr(9),lz(9)
      PetscReal    g,third,sg(3,*)
      data      lr/-1,1,1,-1,0,1,0,-1,0/,lz/-1,-1,1,1,-1,0,1,0,0/
      data      third / 0.3333333333333333 /

!     2x2 integration
      g = sqrt(third)
      do i = 1,4
         sg(1,i) = g*lr(i)
         sg(2,i) = g*lz(i)
         sg(3,i) = 1.0
      end do

      end

!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     ex54_psi - anusotropic material direction
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscReal function ex54_psi(x,y)
      implicit  none
      PetscReal x,y,theta
      common /ex54_theta/ theta
      ex54_psi = theta
      if (theta < 0.) then     ! circular
         if (y==0) then
            ex54_psi = 2.0*atan(1.0)
         else
            ex54_psi = atan(-x/y)
         endif
      endif
      end

!
!/*TEST
!
!   build:
!
!   test:
!      nsize: 4
!      args: -ne 39 -theta 30.0 -epsilon 1.e-1 -blob_center 0.,0. -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mat_coarsen_type hem -pc_gamg_square_graph 0 -ksp_monitor_short -pc_gamg_esteig_ksp_max_it 5
!      requires: !single
!
!TEST*/
