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
!/*T
!   Concepts: KSP^solving a system of linear equations
!T*/
! -----------------------------------------------------------------------
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscksp.h>
#include <finclude/petscpc.h>
#include <finclude/petscviewer.h>
#include <finclude/petscviewer.h90>
      Vec              xvec,bvec,uvec
      Mat              Amat 
      KSP              ksp
      PC               pc
      PetscErrorCode   ierr
      PetscViewer viewer
      PetscInt qj,qi,ne,M,Istart,Iend,geq,ix
      PetscInt ki,kj,lint,nel,ll,j1,i1,ndf
      PetscInt :: idx(4)
      PetscBool  flg,out_matlab
      PetscMPIInt npe,mype
      PetscScalar::ss(4,4),res(4),val
      PetscReal::shp(3,9),sg(3,9),flux(2)
      PetscReal::thk,a1,a2
      PetscReal, external :: ex54_psi
      PetscReal::norm,tol,theta,eps,h,x,y,xsj
      PetscReal::coord(2,4),dd(2,2),ev(3),blb(2)
      
      common /ex54_theta/ theta 
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,npe,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,mype,ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 set parameters
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      ne = 9
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-ne',ne,flg,ierr)
      h = 2.d0/ne
      M = (ne+1)*(ne+1)
      theta = 90.d0
!     theta is input in degrees
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-theta',theta,flg,
     $     ierr)
      theta = theta / 57.2957795
      eps = 1.d0      
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-epsilon',eps,flg,
     $     ierr)
      ki = 2
      call PetscOptionsGetRealArray(PETSC_NULL_CHARACTER,'-blob_center',
     $     blb,ki,flg,ierr)
      if( .not. flg ) then
         blb(1) = 0.d0
         blb(2) = 0.d0
      else if( ki .ne. 2 ) then
         print *, 'error: ', ki,
     $        ' arguments read for -blob_center.  Needs to be two.'
      endif
      call PetscOptionsGetBool(PETSC_NULL_CHARACTER,'-out_matlab',
     $     out_matlab,flg,ierr)

      ev(1) = 1.d0
      ev(2) = eps*ev(1)
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Compute the matrix and right-hand-side vector that define
!     the linear system, Ax = b.
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Create matrix.  When using MatCreate(), the matrix format can
!  be specified at runtime.
      call MatCreate(PETSC_COMM_WORLD,Amat,ierr)
      call MatSetSizes( Amat,PETSC_DECIDE, PETSC_DECIDE, M, M, ierr )
      if( npe == 1 ) then
         call MatSetType( Amat, MATAIJ, ierr )
      else
         call MatSetType( Amat, MATMPIAIJ, ierr )
      endif
      call MatMPIAIJSetPreallocation(Amat,9,PETSC_NULL_INTEGER,6,
     $     PETSC_NULL_INTEGER, ierr)
      call MatSetFromOptions( Amat, ierr )
      call MatSetUp( Amat, ierr )
      call MatGetOwnershipRange( Amat, Istart, Iend, ierr )
!  Create vectors.  Note that we form 1 vector from scratch and
!  then duplicate as needed.
      call MatGetVecs( Amat, PETSC_NULL_OBJECT, xvec, ierr ) 
      call VecSetFromOptions( xvec, ierr )
      call VecDuplicate( xvec, bvec, ierr )
      call VecDuplicate( xvec, uvec, ierr )
!  Assemble matrix.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array "col").
      thk = 1.d0              ! thickness
      nel = 4                   ! nodes per element (quad)
      ndf = 1
      call int2d(2,lint,sg)
      ix = 0
      do geq=Istart,Iend-1,1
         qj = geq/(ne+1); qi = mod(geq,(ne+1))
         x = h*qi - 1.d0; y = h*qj - 1.d0 ! lower left corner (-1,-1)
         if( qi < ne .and. qj < ne ) then
            coord(1,1) = x;   coord(2,1) = y 
            coord(1,2) = x+h; coord(2,2) = y 
            coord(1,3) = x+h; coord(2,3) = y+h 
            coord(1,4) = x;   coord(2,4) = y+h
! form stiff
            ss = 0.d0
            do ll = 1,lint
               call shp2dquad(sg(1,ll),sg(2,ll),coord,shp,xsj,2)
               xsj = xsj*sg(3,ll)*thk
               call thfx2d(ev,coord,shp,dd,2,2,4,ex54_psi)
               j1 = 1
               do kj = 1,nel
                  a1 = (dd(1,1)*shp(1,kj) + dd(1,2)*shp(2,kj))*xsj
                  a2 = (dd(2,1)*shp(1,kj) + dd(2,2)*shp(2,kj))*xsj
c     Compute residual                 
c                  p(j1) = p(j1) - a1*gradt(1) - a2*gradt(2)
c     Compute tangent
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
            if( qj > 0 ) then
               call MatSetValues(Amat,4,idx,4,idx,ss,ADD_VALUES,ierr)
            else                !     a BC 
               do ki=1,4,1
                  do kj=1,4,1
                     if(ki<3 .or. kj<3 ) then
                        if( ki==kj ) then
                           ss(ki,kj) = .1d0*ss(ki,kj)
                        else
                           ss(ki,kj) = 0.d0
                        endif
                     endif
                  enddo
               enddo
               call MatSetValues(Amat,4,idx,4,idx,ss,ADD_VALUES,ierr)
            endif               ! BC
         endif                  ! add element
         if( qj > 0 ) then      ! set rhs

            val = h*h*exp(-1.d2*((x+h/2)-blb(1))**2)*
     $           exp(-1.d2*((y+h/2)-blb(2))**2)
            call VecSetValues(bvec,1,geq,val,INSERT_VALUES,ierr)
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

      call KSPSetOperators(ksp,Amat,Amat,DIFFERENT_NONZERO_PATTERN,ierr)

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


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                      output
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      if( out_matlab ) then
         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Amat',
     $        FILE_MODE_WRITE,viewer,ierr)
         call MatView(Amat,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Bvec',
     $        FILE_MODE_WRITE,viewer,ierr)
         call VecView(bvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Xvec',
     $        FILE_MODE_WRITE,viewer,ierr)
         call VecView(xvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         call MatMult(Amat,xvec,uvec,ierr)
         val = -1.d0
         call VecAXPY(uvec,val,bvec,ierr)
         call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'Rvec',
     $        FILE_MODE_WRITE,viewer,ierr)
         call VecView(uvec,viewer,ierr)
         call PetscViewerDestroy(viewer,ierr)

         if( mype == 0 ) then
            open(1,file="ex54f.m", FORM="formatted")
            write (1,*) "A = PetscBinaryRead('Amat');"
            write (1,*) "[m n] = size(A);"
            write (1,*) "mm = sqrt(m);"
            write (1,*) "b = PetscBinaryRead('Bvec');"
            write (1,*) "x = PetscBinaryRead('Xvec');"
            write (1,*) "r = PetscBinaryRead('Rvec');"
            write (1,*) "bb = reshape(b,mm,mm);"
            write (1,*) "xx = reshape(x,mm,mm);"
            write (1,*) "rr = reshape(r,mm,mm);"
c            write (1,*) "imagesc(bb')"
c            write (1,*) "title('RHS'),"
            write (1,*) "figure,"
            write (1,*) "imagesc(xx')"
            write (1,2002) eps,theta*57.2957795
            write (1,*) "figure,"
            write (1,*) "imagesc(rr')"
            write (1,*) "title('Residual'),"
            close(1)
         endif
      endif
 2002 format("title('Solution: esp=",d9.3,", theta=",g8.3,"'),")
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

      subroutine thfx2d(ev,xl,shp,dd,ndm,ndf,nel,dir)
      
c     Compute thermal gradient and flux
      
      implicit  none

      integer   ndm,ndf,nel,i
      PetscReal ev(2),xl(ndm,nel),shp(3,*),dir
      PetscReal xx,yy,psi,cs,sn,c2,s2,dd(2,2)

c      temp     = 0.0d0
c      gradt(1) = 0.0d0
c      gradt(2) = 0.0d0
      xx       = 0.0d0
      yy       = 0.0d0
      do i = 1,nel
c        gradt(1) = gradt(1) + shp(1,i)*ul(1,i)
c        gradt(2) = gradt(2) + shp(2,i)*ul(1,i)
c        temp     = temp     + shp(3,i)*ul(1,i)
        xx       = xx       + shp(3,i)*xl(1,i)
        yy       = yy       + shp(3,i)*xl(2,i)
      end do
      psi = dir(xx,yy)
c     Compute thermal flux
      cs  = cos(psi)
      sn  = sin(psi)
      c2  = cs*cs
      s2  = sn*sn
      cs  = cs*sn

      dd(1,1) = c2*ev(1) + s2*ev(2)
      dd(2,2) = s2*ev(1) + c2*ev(2)
      dd(1,2) = cs*(ev(1) - ev(2))
      dd(2,1) = dd(1,2)

c      flux(1) = -dd(1,1)*gradt(1) - dd(1,2)*gradt(2)
c      flux(2) = -dd(2,1)*gradt(1) - dd(2,2)*gradt(2)

      end

!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     shp2dquad - shape functions - compute derivatives w/r natural coords.
!     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       subroutine shp2dquad(s,t,xl,shp,xsj,ndm)
c-----[--.----+----.----+----.-----------------------------------------]
c      Purpose: Shape function routine for 4-node isoparametric quads
c
c      Inputs:
c         s,t       - Natural coordinates of point
c         xl(ndm,*) - Nodal coordinates for element
c         ndm       - Spatial dimension of mesh

c      Outputs:
c         shp(3,*)  - Shape functions and derivatives at point
c                     shp(1,i) = dN_i/dx  or dN_i/dxi_1
c                     shp(2,i) = dN_i/dy  or dN_i/dxi_2
c                     shp(3,i) = N_i
c         xsj       - Jacobian determinant at point
c-----[--.----+----.----+----.-----------------------------------------]
      implicit  none
      integer   ndm
      real*8    xo,xs,xt, yo,ys,yt, xsm,xsp,xtm,xtp, ysm,ysp,ytm,ytp
      real*8    s,t, xsj,xsj1, sh,th,sp,tp,sm,tm, xl(ndm,4),shp(3,4)

c     Set up interpolations

      sh = 0.5d0*s
      th = 0.5d0*t
      sp = 0.5d0 + sh
      tp = 0.5d0 + th
      sm = 0.5d0 - sh
      tm = 0.5d0 - th
      shp(3,1) =   sm*tm
      shp(3,2) =   sp*tm
      shp(3,3) =   sp*tp
      shp(3,4) =   sm*tp

c     Set up natural coordinate functions (times 4)

      xo =  xl(1,1)-xl(1,2)+xl(1,3)-xl(1,4)
      xs = -xl(1,1)+xl(1,2)+xl(1,3)-xl(1,4) + xo*t
      xt = -xl(1,1)-xl(1,2)+xl(1,3)+xl(1,4) + xo*s
      yo =  xl(2,1)-xl(2,2)+xl(2,3)-xl(2,4)
      ys = -xl(2,1)+xl(2,2)+xl(2,3)-xl(2,4) + yo*t
      yt = -xl(2,1)-xl(2,2)+xl(2,3)+xl(2,4) + yo*s

c     Compute jacobian (times 16)

      xsj1 = xs*yt - xt*ys

c     Divide jacobian by 16 (multiply by .0625)

      xsj = 0.0625d0*xsj1
      if(xsj1.eq.0.0d0) then
         xsj1 = 1.0d0
      else
         xsj1 = 1.0d0/xsj1
      endif
      
c     Divide functions by jacobian
      
      xs  = (xs+xs)*xsj1
      xt  = (xt+xt)*xsj1
      ys  = (ys+ys)*xsj1
      yt  = (yt+yt)*xsj1
      
c     Multiply by interpolations
      
      ytm =  yt*tm
      ysm =  ys*sm
      ytp =  yt*tp
      ysp =  ys*sp
      xtm =  xt*tm
      xsm =  xs*sm
      xtp =  xt*tp
      xsp =  xs*sp
      
c     Compute shape functions
      
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
      subroutine int2d(l,lint,sg)
c-----[--.----+----.----+----.-----------------------------------------]
c     Purpose: Form Gauss points and weights for two dimensions
      
c     Inputs:
c     l       - Number of points/direction
      
c     Outputs:
c     lint    - Total number of points
c     sg(3,*) - Array of points and weights
c-----[--.----+----.----+----.-----------------------------------------]
      implicit  none
      integer   l,i,lint,lr(9),lz(9)
      real*8    g,third,sg(3,*)
      data      lr/-1,1,1,-1,0,1,0,-1,0/,lz/-1,-1,1,1,-1,0,1,0,0/
      data      third / 0.3333333333333333d0 /
      
c     Set number of total points
      
      lint = l*l
      
c     2x2 integration
      g = sqrt(third)
      do i = 1,4
         sg(1,i) = g*lr(i)
         sg(2,i) = g*lz(i)
         sg(3,i) = 1.d0
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
      if( theta < 0. ) then     ! circular
         if(y==0) then
            ex54_psi = 2.d0*atan(1.d0)
         else
            ex54_psi = atan(-x/y)
         endif
      endif
      end
