
      program main
#include <petsc/finclude/petsc.h>
      use petsc
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!
      SNES     snes
      Vec      x,r,lb,ub
      Mat      J
      PetscErrorCode  ierr
      PetscInt i2
      PetscMPIInt size
      PetscScalar,pointer :: xx(:)
      PetscScalar zero,big
      SNESLineSearch ls

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormJacobian
      external ShashiPostCheck

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Macro definitions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Macros to make clearer the process of setting values in vectors and
!  getting values from vectors.  These vectors are used in the routines
!  FormFunction() and FormJacobian().
!   - The element lx_a(ib) is element ib in the vector x
!
#define lx_a(ib) lx_v(lx_i + (ib))
#define lf_a(ib) lf_v(lf_i + (ib))
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      if (size .ne. 1) then; SETERRA(PETSC_COMM_WORLD,1,'requires one process'); endif

      big  = 2.88
      big  = PETSC_INFINITY
      zero = 0.0
      i2  = 26
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix and vector data structures; set corresponding routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create vectors for solution and nonlinear function

      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,i2,x,ierr))
      PetscCallA(VecDuplicate(x,r,ierr))

!  Create Jacobian matrix data structure

      PetscCallA(MatCreateDense(PETSC_COMM_SELF,26,26,26,26,PETSC_NULL_SCALAR,J,ierr))

!  Set function evaluation routine and vector

      PetscCallA(SNESSetFunction(snes,r,FormFunction,0,ierr))

!  Set Jacobian matrix data structure and Jacobian evaluation routine

      PetscCallA(SNESSetJacobian(snes,J,J,FormJacobian,0,ierr))

      PetscCallA(VecDuplicate(x,lb,ierr))
      PetscCallA(VecDuplicate(x,ub,ierr))
      PetscCallA(VecSet(lb,zero,ierr))
!      PetscCallA(VecGetArrayF90(lb,xx,ierr))
!      PetscCallA(ShashiLowerBound(xx))
!      PetscCallA(VecRestoreArrayF90(lb,xx,ierr))
      PetscCallA(VecSet(ub,big,ierr))
!      PetscCallA(SNESVISetVariableBounds(snes,lb,ub,ierr))

      PetscCallA(SNESGetLineSearch(snes,ls,ierr))
      PetscCallA(SNESLineSearchSetPostCheck(ls,ShashiPostCheck,0,ierr))
      PetscCallA(SNESSetType(snes,SNESVINEWTONRSLS,ierr))

      PetscCallA(SNESSetFromOptions(snes,ierr))

!     set initial guess

      PetscCallA(VecGetArrayF90(x,xx,ierr))
      PetscCallA(ShashiInitialGuess(xx))
      PetscCallA(VecRestoreArrayF90(x,xx,ierr))

      PetscCallA(SNESSolve(snes,PETSC_NULL_VEC,x,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(VecDestroy(lb,ierr))
      PetscCallA(VecDestroy(ub,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(MatDestroy(J,ierr))
      PetscCallA(SNESDestroy(snes,ierr))
      PetscCallA(PetscFinalize(ierr))
      end
!
! ------------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameter:
!  f - function vector
!
      subroutine FormFunction(snes,x,f,dummy,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      implicit none
      SNES     snes
      Vec      x,f
      PetscErrorCode ierr
      integer dummy(*)

!  Declarations for use with local arrays

      PetscScalar  lx_v(2),lf_v(2)
      PetscOffset  lx_i,lf_i

!  Get pointers to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the Fortran chapter of the users manual for details.

      PetscCall(VecGetArrayRead(x,lx_v,lx_i,ierr))
      PetscCall(VecGetArray(f,lf_v,lf_i,ierr))
      PetscCall(ShashiFormFunction(lx_a(1),lf_a(1)))
      PetscCall(VecRestoreArrayRead(x,lx_v,lx_i,ierr))
      PetscCall(VecRestoreArray(f,lf_v,lf_i,ierr))

      return
      end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameters:
!  A - Jacobian matrix
!  B - optionally different preconditioning matrix
!  flag - flag indicating matrix structure
!
      subroutine FormJacobian(snes,X,jac,B,dummy,ierr)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      implicit none
      SNES         snes
      Vec          X
      Mat          jac,B
      PetscErrorCode ierr
      integer dummy(*)

!  Declarations for use with local arrays

      PetscScalar lx_v(1),lf_v(1)
      PetscOffset lx_i,lf_i

!  Get pointer to vector data

      PetscCall(VecGetArrayRead(x,lx_v,lx_i,ierr))
      PetscCall(MatDenseGetArray(B,lf_v,lf_i,ierr))
      PetscCall(ShashiFormJacobian(lx_a(1),lf_a(1)))
      PetscCall(MatDenseRestoreArray(B,lf_v,lf_i,ierr))
      PetscCall(VecRestoreArrayRead(x,lx_v,lx_i,ierr))

!  Assemble matrix

      PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))

      return
      end

            subroutine ShashiLowerBound(an_r)
!        implicit PetscScalar (a-h,o-z)
        implicit none
        PetscScalar an_r(26)
        PetscInt i

        do i=2,26
           an_r(i) = 1000.0/6.023D+23
        enddo
        return
        end

            subroutine ShashiInitialGuess(an_r)
!        implicit PetscScalar (a-h,o-z)
        implicit none
        PetscScalar an_c_additive
        PetscScalar       an_h_additive
        PetscScalar an_o_additive
        PetscScalar   atom_c_init
        PetscScalar atom_h_init
        PetscScalar atom_n_init
        PetscScalar atom_o_init
        PetscScalar h_init
        PetscScalar p_init
        PetscInt nfuel
        PetscScalar temp,pt
        PetscScalar an_r(26)
        PetscInt an_h(1),an_c(1)

        pt = 0.1
        atom_c_init =6.7408177364816552D-022
        atom_h_init = 2.0
        atom_o_init = 1.0
        atom_n_init = 3.76
        nfuel = 1
        an_c(1) = 1
        an_h(1) = 4
        an_c_additive = 2
        an_h_additive = 6
        an_o_additive = 1
        h_init = 128799.7267952987
        p_init = 0.1
        temp = 1500

      an_r( 1) =     1.66000D-24
      an_r( 2) =     1.66030D-22
      an_r( 3) =     5.00000D-01
      an_r( 4) =     1.66030D-22
      an_r( 5) =     1.66030D-22
      an_r( 6) =     1.88000D+00
      an_r( 7) =     1.66030D-22
      an_r( 8) =     1.66030D-22
      an_r( 9) =     1.66030D-22
      an_r(10) =     1.66030D-22
      an_r(11) =     1.66030D-22
      an_r(12) =     1.66030D-22
      an_r(13) =     1.66030D-22
      an_r(14) =     1.00000D+00
      an_r(15) =     1.66030D-22
      an_r(16) =     1.66030D-22
      an_r(17) =     1.66000D-24
      an_r(18) =     1.66030D-24
      an_r(19) =     1.66030D-24
      an_r(20) =     1.66030D-24
      an_r(21) =     1.66030D-24
      an_r(22) =     1.66030D-24
      an_r(23) =     1.66030D-24
      an_r(24) =     1.66030D-24
      an_r(25) =     1.66030D-24
      an_r(26) =     1.66030D-24

      an_r = 0
      an_r( 3) =     .5
      an_r( 6) =     1.88000
      an_r(14) =     1.

#if defined(solution)
      an_r( 1) =      3.802208D-33
      an_r( 2) =      1.298287D-29
      an_r( 3) =      2.533067D-04
      an_r( 4) =      6.865078D-22
      an_r( 5) =      9.993125D-01
      an_r( 6) =      1.879964D+00
      an_r( 7) =      4.449489D-13
      an_r( 8) =      3.428687D-07
      an_r( 9) =      7.105138D-05
      an_r(10) =      1.094368D-04
      an_r(11) =      2.362305D-06
      an_r(12) =      1.107145D-09
      an_r(13) =      1.276162D-24
      an_r(14) =      6.315538D-04
      an_r(15) =      2.356540D-09
      an_r(16) =      2.048248D-09
      an_r(17) =      1.966187D-22
      an_r(18) =      7.856497D-29
      an_r(19) =      1.987840D-36
      an_r(20) =      8.182441D-22
      an_r(21) =      2.684880D-16
      an_r(22) =      2.680473D-16
      an_r(23) =      6.594967D-18
      an_r(24) =      2.509714D-21
      an_r(25) =      3.096459D-21
      an_r(26) =      6.149551D-18
#endif

      return
      end

      subroutine ShashiFormFunction(an_r,f_eq)
!       implicit PetscScalar (a-h,o-z)
        implicit none
        PetscScalar an_c_additive
        PetscScalar       an_h_additive
        PetscScalar an_o_additive
        PetscScalar   atom_c_init
        PetscScalar atom_h_init
        PetscScalar atom_n_init
        PetscScalar atom_o_init
        PetscScalar h_init
        PetscScalar p_init
        PetscInt nfuel
        PetscScalar temp,pt
        PetscScalar an_r(26),k_eq(23),f_eq(26)
        PetscScalar H_molar(26)
        PetscInt an_h(1),an_c(1)
        PetscScalar part_p(26),idiff
        PetscInt i_cc,i_hh,i_h2o
        PetscScalar an_t,sum_h
        PetscScalar a_io2
        PetscInt i,ip
        pt = 0.1
        atom_c_init =6.7408177364816552D-022
        atom_h_init = 2.0
        atom_o_init = 1.0
        atom_n_init = 3.76
        nfuel = 1
        an_c(1) = 1
        an_h(1) = 4
        an_c_additive = 2
        an_h_additive = 6
        an_o_additive = 1
        h_init = 128799.7267952987
        p_init = 0.1
        temp = 1500

       k_eq( 1) =     1.75149D-05
       k_eq( 2) =     4.01405D-06
       k_eq( 3) =     6.04663D-14
       k_eq( 4) =     2.73612D-01
       k_eq( 5) =     3.25592D-03
       k_eq( 6) =     5.33568D+05
       k_eq( 7) =     2.07479D+05
       k_eq( 8) =     1.11841D-02
       k_eq( 9) =     1.72684D-03
       k_eq(10) =     1.98588D-07
       k_eq(11) =     7.23600D+27
       k_eq(12) =     5.73926D+49
       k_eq(13) =     1.00000D+00
       k_eq(14) =     1.64493D+16
       k_eq(15) =     2.73837D-29
       k_eq(16) =     3.27419D+50
       k_eq(17) =     1.72447D-23
       k_eq(18) =     4.24657D-06
       k_eq(19) =     1.16065D-14
       k_eq(20) =     3.28020D+25
       k_eq(21) =     1.06291D+00
       k_eq(22) =     9.11507D+02
       k_eq(23) =     6.02837D+03

       H_molar( 1) =     3.26044D+03
       H_molar( 2) =    -8.00407D+04
       H_molar( 3) =     4.05873D+04
       H_molar( 4) =    -3.31849D+05
       H_molar( 5) =    -1.93654D+05
       H_molar( 6) =     3.84035D+04
       H_molar( 7) =     4.97589D+05
       H_molar( 8) =     2.74483D+05
       H_molar( 9) =     1.30022D+05
       H_molar(10) =     7.58429D+04
       H_molar(11) =     2.42948D+05
       H_molar(12) =     1.44588D+05
       H_molar(13) =    -7.16891D+04
       H_molar(14) =     3.63075D+04
       H_molar(15) =     9.23880D+04
       H_molar(16) =     6.50477D+04
       H_molar(17) =     3.04310D+05
       H_molar(18) =     7.41707D+05
       H_molar(19) =     6.32767D+05
       H_molar(20) =     8.90624D+05
       H_molar(21) =     2.49805D+04
       H_molar(22) =     6.43473D+05
       H_molar(23) =     1.02861D+06
       H_molar(24) =    -6.07503D+03
       H_molar(25) =     1.27020D+05
       H_molar(26) =    -1.07011D+05
!=============
      an_t = 0
      sum_h = 0

      do i = 1,26
         an_t = an_t + an_r(i)
      enddo

        f_eq(1) = atom_h_init                                           &
     &          - (an_h(1)*an_r(1) + an_h_additive*an_r(2)              &
     &              + 2*an_r(5) + an_r(10) + an_r(11) + 2*an_r(14)      &
     &              + an_r(16)+ 2*an_r(17) + an_r(19)                   &
     &              +an_r(20) + 3*an_r(22)+an_r(26))

        f_eq(2) = atom_o_init                                           &
     &          - (an_o_additive*an_r(2) + 2*an_r(3)                    &
     &             + 2*an_r(4) + an_r(5)                                &
     &             + an_r(8) + an_r(9) + an_r(10) + an_r(12) + an_r(13) &
     &             + 2*an_r(15) + 2*an_r(16)+ an_r(20) + an_r(22)       &
     &             + an_r(23) + 2*an_r(24) + 1*an_r(25)+an_r(26))

        f_eq(3) = an_r(2)-1.0d-150

        f_eq(4) = atom_c_init                                           &
     &          - (an_c(1)*an_r(1) + an_c_additive * an_r(2)            &
     &          + an_r(4) + an_r(13)+ 2*an_r(17) + an_r(18)             &
     &          + an_r(19) + an_r(20))

        do ip = 1,26
           part_p(ip) = (an_r(ip)/an_t) * pt
        enddo

        i_cc      = an_c(1)
        i_hh      = an_h(1)
        a_io2      = i_cc + i_hh/4.0
        i_h2o     = i_hh/2
        idiff     = (i_cc + i_h2o) - (a_io2 + 1)

        f_eq(5) = k_eq(11)*an_r(1)*an_r(3)**a_io2                       &
     &          - (an_r(4)**i_cc)*(an_r(5)**i_h2o)*((pt/an_t)**idiff)
!           write(6,*)f_eq(5),an_r(1), an_r(3), an_r(4),an_r(5),' II'
!          stop
        f_eq(6) = atom_n_init                                           &
     &          - (2*an_r(6) + an_r(7) + an_r(9) + 2*an_r(12)           &
     &              + an_r(15)                                          &
     &              + an_r(23))

      f_eq( 7) = part_p(11)                                             &
     &         - (k_eq( 1) * sqrt(part_p(14)+1d-23))
      f_eq( 8) = part_p( 8)                                             &
     &         - (k_eq( 2) * sqrt(part_p( 3)+1d-23))

      f_eq( 9) = part_p( 7)                                             &
     &         - (k_eq( 3) * sqrt(part_p( 6)+1d-23))

      f_eq(10) = part_p(10)                                             &
     &         - (k_eq( 4) * sqrt(part_p( 3)+1d-23))                    &
     &         * sqrt(part_p(14))

      f_eq(11) = part_p( 9)                                             &
     &         - (k_eq( 5) * sqrt(part_p( 3)+1d-23))                    &
     &         * sqrt(part_p( 6)+1d-23)
      f_eq(12) = part_p( 5)                                             &
     &         - (k_eq( 6) * sqrt(part_p( 3)+1d-23))                    &
     &         * (part_p(14))

      f_eq(13) = part_p( 4)                                             &
     &         - (k_eq( 7) * sqrt(part_p(3)+1.0d-23))                   &
     &         * (part_p(13))

      f_eq(14) = part_p(15)                                             &
     &         - (k_eq( 8) * sqrt(part_p(3)+1.0d-50))                   &
     &         * (part_p( 9))
      f_eq(15) = part_p(16)                                             &
     &         - (k_eq( 9) * part_p( 3))                                &
     &         * sqrt(part_p(14)+1d-23)

      f_eq(16) = part_p(12)                                             &
     &         - (k_eq(10) * sqrt(part_p( 3)+1d-23))                    &
     &         * (part_p( 6))

      f_eq(17) = part_p(14)*part_p(18)**2                               &
     &         - (k_eq(15) * part_p(17))

      f_eq(18) = (part_p(13)**2)                                        &
     &     - (k_eq(16) * part_p(3)*part_p(18)**2)
      print*,f_eq(18),part_p(3),part_p(18),part_p(13),k_eq(16)

      f_eq(19) = part_p(19)*part_p(3) - k_eq(17)*part_p(13)*part_p(10)

      f_eq(20) = part_p(21)*part_p(20) - k_eq(18)*part_p(19)*part_p(8)

      f_eq(21) = part_p(21)*part_p(23) - k_eq(19)*part_p(7)*part_p(8)

      f_eq(22) = part_p(5)*part_p(11) - k_eq(20)*part_p(21)*part_p(22)

      f_eq(23) = part_p(24) - k_eq(21)*part_p(21)*part_p(3)

      f_eq(24) =  part_p(3)*part_p(25) - k_eq(22)*part_p(24)*part_p(8)

      f_eq(25) =  part_p(26) - k_eq(23)*part_p(21)*part_p(10)

      f_eq(26) = -(an_r(20) + an_r(22) + an_r(23))                      &
     &          +(an_r(21) + an_r(24) + an_r(25) + an_r(26))

             do i = 1,26
                 write(44,*)i,f_eq(i)
              enddo

      return
      end

      subroutine ShashiFormJacobian(an_r,d_eq)
!        implicit PetscScalar (a-h,o-z)
        implicit none
        PetscScalar an_c_additive
        PetscScalar       an_h_additive
        PetscScalar an_o_additive
        PetscScalar   atom_c_init
        PetscScalar atom_h_init
        PetscScalar atom_n_init
        PetscScalar atom_o_init
        PetscScalar h_init
        PetscScalar p_init
        PetscInt nfuel
        PetscScalar temp,pt
        PetscScalar an_t,ai_o2
        PetscScalar an_tot1_d,an_tot1
        PetscScalar an_tot2_d,an_tot2
        PetscScalar const5,const3,const2
        PetscScalar const_cube
        PetscScalar const_five
        PetscScalar const_four
        PetscScalar const_six
        PetscInt jj,jb,ii3,id,ib,i
        PetscScalar pt2,pt1
        PetscScalar an_r(26),k_eq(23)
        PetscScalar d_eq(26,26),H_molar(26)
        PetscInt an_h(1),an_c(1)
        PetscScalar ai_pwr1,idiff
        PetscInt i_cc,i_hh
        PetscInt i_h2o,i_pwr2,i_co2_h2o
        PetscScalar pt_cube,pt_five
        PetscScalar pt_four
        PetscScalar pt_val1,pt_val2
        PetscInt j

        pt = 0.1
        atom_c_init =6.7408177364816552D-022
        atom_h_init = 2.0
        atom_o_init = 1.0
        atom_n_init = 3.76
        nfuel = 1
        an_c(1) = 1
        an_h(1) = 4
        an_c_additive = 2
        an_h_additive = 6
        an_o_additive = 1
        h_init = 128799.7267952987
        p_init = 0.1
        temp = 1500

       k_eq( 1) =     1.75149D-05
       k_eq( 2) =     4.01405D-06
       k_eq( 3) =     6.04663D-14
       k_eq( 4) =     2.73612D-01
       k_eq( 5) =     3.25592D-03
       k_eq( 6) =     5.33568D+05
       k_eq( 7) =     2.07479D+05
       k_eq( 8) =     1.11841D-02
       k_eq( 9) =     1.72684D-03
       k_eq(10) =     1.98588D-07
       k_eq(11) =     7.23600D+27
       k_eq(12) =     5.73926D+49
       k_eq(13) =     1.00000D+00
       k_eq(14) =     1.64493D+16
       k_eq(15) =     2.73837D-29
       k_eq(16) =     3.27419D+50
       k_eq(17) =     1.72447D-23
       k_eq(18) =     4.24657D-06
       k_eq(19) =     1.16065D-14
       k_eq(20) =     3.28020D+25
       k_eq(21) =     1.06291D+00
       k_eq(22) =     9.11507D+02
       k_eq(23) =     6.02837D+03

       H_molar( 1) =     3.26044D+03
       H_molar( 2) =    -8.00407D+04
       H_molar( 3) =     4.05873D+04
       H_molar( 4) =    -3.31849D+05
       H_molar( 5) =    -1.93654D+05
       H_molar( 6) =     3.84035D+04
       H_molar( 7) =     4.97589D+05
       H_molar( 8) =     2.74483D+05
       H_molar( 9) =     1.30022D+05
       H_molar(10) =     7.58429D+04
       H_molar(11) =     2.42948D+05
       H_molar(12) =     1.44588D+05
       H_molar(13) =    -7.16891D+04
       H_molar(14) =     3.63075D+04
       H_molar(15) =     9.23880D+04
       H_molar(16) =     6.50477D+04
       H_molar(17) =     3.04310D+05
       H_molar(18) =     7.41707D+05
       H_molar(19) =     6.32767D+05
       H_molar(20) =     8.90624D+05
       H_molar(21) =     2.49805D+04
       H_molar(22) =     6.43473D+05
       H_molar(23) =     1.02861D+06
       H_molar(24) =    -6.07503D+03
       H_molar(25) =     1.27020D+05
       H_molar(26) =    -1.07011D+05

!
!=======
      do jb = 1,26
         do ib = 1,26
            d_eq(ib,jb) = 0.0d0
         end do
      end do

        an_t = 0.0
      do id = 1,26
         an_t = an_t + an_r(id)
      enddo

!====
!====
        d_eq(1,1) = -an_h(1)
        d_eq(1,2) = -an_h_additive
        d_eq(1,5) = -2
        d_eq(1,10) = -1
        d_eq(1,11) = -1
        d_eq(1,14) = -2
        d_eq(1,16) = -1
        d_eq(1,17) = -2
        d_eq(1,19) = -1
        d_eq(1,20) = -1
        d_eq(1,22) = -3
        d_eq(1,26) = -1

        d_eq(2,2) = -1*an_o_additive
        d_eq(2,3) = -2
        d_eq(2,4) = -2
        d_eq(2,5) = -1
        d_eq(2,8) = -1
        d_eq(2,9) = -1
        d_eq(2,10) = -1
        d_eq(2,12) = -1
        d_eq(2,13) = -1
        d_eq(2,15) = -2
        d_eq(2,16) = -2
        d_eq(2,20) = -1
        d_eq(2,22) = -1
        d_eq(2,23) = -1
        d_eq(2,24) = -2
        d_eq(2,25) = -1
        d_eq(2,26) = -1

        d_eq(6,6) = -2
        d_eq(6,7) = -1
        d_eq(6,9) = -1
        d_eq(6,12) = -2
        d_eq(6,15) = -1
        d_eq(6,23) = -1

        d_eq(4,1) = -an_c(1)
        d_eq(4,2) = -an_c_additive
        d_eq(4,4) = -1
        d_eq(4,13) = -1
        d_eq(4,17) = -2
        d_eq(4,18) = -1
        d_eq(4,19) = -1
        d_eq(4,20) = -1

!----------
        const2 = an_t*an_t
        const3 = (an_t)*sqrt(an_t)
        const5 = an_t*const3

           const_cube =  an_t*an_t*an_t
           const_four =  const2*const2
           const_five =  const2*const_cube
           const_six  = const_cube*const_cube
           pt_cube = pt*pt*pt
           pt_four = pt_cube*pt
           pt_five = pt_cube*pt*pt

           i_cc = an_c(1)
           i_hh = an_h(1)
           ai_o2     = i_cc + float(i_hh)/4.0
           i_co2_h2o = i_cc + i_hh/2
           i_h2o     = i_hh/2
           ai_pwr1  = 1 + i_cc + float(i_hh)/4.0
           i_pwr2  = i_cc + i_hh/2
           idiff     = (i_cc + i_h2o) - (ai_o2 + 1)

           pt1   = pt**(ai_pwr1)
           an_tot1 = an_t**(ai_pwr1)
           pt_val1 = (pt/an_t)**(ai_pwr1)
           an_tot1_d = an_tot1*an_t

           pt2   = pt**(i_pwr2)
           an_tot2 = an_t**(i_pwr2)
           pt_val2 = (pt/an_t)**(i_pwr2)
           an_tot2_d = an_tot2*an_t

           d_eq(5,1) =                                                  &
     &           -(an_r(4)**i_cc)*(an_r(5)**i_h2o)                      &
     &           *((pt/an_t)**idiff) *(-idiff/an_t)

           do jj = 2,26
              d_eq(5,jj) = d_eq(5,1)
           enddo

           d_eq(5,1) = d_eq(5,1) + k_eq(11)* (an_r(3) **ai_o2)

           d_eq(5,3) = d_eq(5,3) + k_eq(11)* (ai_o2*an_r(3)**(ai_o2-1)) &
     &                           * an_r(1)

           d_eq(5,4) = d_eq(5,4) - (i_cc*an_r(4)**(i_cc-1))*            &
     &                           (an_r(5)**i_h2o)* ((pt/an_t)**idiff)
           d_eq(5,5) = d_eq(5,5)                                        &
     &               - (i_h2o*(an_r(5)**(i_h2o-1)))                     &
     &               * (an_r(4)**i_cc)* ((pt/an_t)**idiff)

           d_eq(3,1) = -(an_r(4)**2)*(an_r(5)**3)*(pt/an_t)*(-1.0/an_t)
           do jj = 2,26
              d_eq(3,jj) = d_eq(3,1)
           enddo

           d_eq(3,2) = d_eq(3,2) + k_eq(12)* (an_r(3)**3)

           d_eq(3,3) = d_eq(3,3) + k_eq(12)* (3*an_r(3)**2)*an_r(2)

           d_eq(3,4) = d_eq(3,4) - 2*an_r(4)*(an_r(5)**3)*(pt/an_t)

           d_eq(3,5) = d_eq(3,5) - 3*(an_r(5)**2)*(an_r(4)**2)*(pt/an_t)
!     &                           *(pt_five/const_five)

           do ii3 = 1,26
              d_eq(3,ii3) = 0.0d0
           enddo
           d_eq(3,2) = 1.0d0

        d_eq(7,1) = pt*an_r(11)*(-1.0)/const2                           &
     &            -k_eq(1)*sqrt(pt)*sqrt(an_r(14)+1d-50)*(-0.5/const3)

        do jj = 2,26
           d_eq(7,jj) = d_eq(7,1)
        enddo

        d_eq(7,11) = d_eq(7,11) + pt/an_t
        d_eq(7,14) = d_eq(7,14)                                         &
     &            - k_eq(1)*sqrt(pt)*(0.5/(sqrt((an_r(14)+1d-50)*an_t)))

        d_eq(8,1) = pt*an_r(8)*(-1.0)/const2                            &
     &            -k_eq(2)*sqrt(pt)*sqrt(an_r(3)+1.0d-50)*(-0.5/const3)

        do jj = 2,26
           d_eq(8,jj) = d_eq(8,1)
        enddo

        d_eq(8,3) = d_eq(8,3)                                           &
     &            -k_eq(2)*sqrt(pt)*(0.5/(sqrt((an_r(3)+1.0d-50)*an_t)))
        d_eq(8,8) = d_eq(8,8) + pt/an_t

        d_eq(9,1) = pt*an_r(7)*(-1.0)/const2                            &
     &            -k_eq(3)*sqrt(pt)*sqrt(an_r(6))*(-0.5/const3)

        do jj = 2,26
           d_eq(9,jj) = d_eq(9,1)
        enddo

        d_eq(9,7) = d_eq(9,7) + pt/an_t
        d_eq(9,6) = d_eq(9,6)                                           &
     &             -k_eq(3)*sqrt(pt)*(0.5/(sqrt(an_r(6)*an_t)))

        d_eq(10,1) = pt*an_r(10)*(-1.0)/const2                          &
     &             -k_eq(4)*(pt)*sqrt((an_r(3)+1.0d-50)                 &
     &       *an_r(14))*(-1.0/const2)
        do jj = 2,26
           d_eq(10,jj) = d_eq(10,1)
        enddo

        d_eq(10,3) = d_eq(10,3)                                         &
     &           -k_eq(4)*(pt)*sqrt(an_r(14))                           &
     &           *(0.5/(sqrt(an_r(3)+1.0d-50)*an_t))
        d_eq(10,10) = d_eq(10,10) + pt/an_t
        d_eq(10,14) = d_eq(10,14)                                       &
     &           -k_eq(4)*(pt)*sqrt(an_r(3)+1.0d-50)                    &
     &            *(0.5/(sqrt(an_r(14)+1.0d-50)*an_t))

        d_eq(11,1) = pt*an_r(9)*(-1.0)/const2                           &
     &             -k_eq(5)*(pt)*sqrt((an_r(3)+1.0d-50)*an_r(6))        &
     &             *(-1.0/const2)

        do jj = 2,26
           d_eq(11,jj) = d_eq(11,1)
        enddo

        d_eq(11,3) = d_eq(11,3)                                         &
     &            -k_eq(5)*(pt)*sqrt(an_r(6))*(0.5/                     &
     &       (sqrt(an_r(3)+1.0d-50)*an_t))
        d_eq(11,6) = d_eq(11,6)                                         &
     &            -k_eq(5)*(pt)*sqrt(an_r(3)+1.0d-50)                   &
     &       *(0.5/(sqrt(an_r(6))*an_t))
        d_eq(11,9) = d_eq(11,9) + pt/an_t

        d_eq(12,1) = pt*an_r(5)*(-1.0)/const2                           &
     &             -k_eq(6)*(pt**1.5)*sqrt(an_r(3)+1.0d-50)             &
     &             *(an_r(14))*(-1.5/const5)

        do jj = 2,26
           d_eq(12,jj) = d_eq(12,1)
        enddo

        d_eq(12,3) = d_eq(12,3)                                         &
     &            -k_eq(6)*(pt**1.5)*((an_r(14)+1.0d-50)/const3)        &
     &            *(0.5/sqrt(an_r(3)+1.0d-50))

        d_eq(12,5) = d_eq(12,5) + pt/an_t
        d_eq(12,14) = d_eq(12,14)                                       &
     &            -k_eq(6)*(pt**1.5)*(sqrt(an_r(3)+1.0d-50)/const3)

        d_eq(13,1) = pt*an_r(4)*(-1.0)/const2                           &
     &             -k_eq(7)*(pt**1.5)*sqrt(an_r(3)+1.0d-50)             &
     &             *(an_r(13))*(-1.5/const5)

        do jj = 2,26
           d_eq(13,jj) = d_eq(13,1)
        enddo

        d_eq(13,3) = d_eq(13,3)                                         &
     &            -k_eq(7)*(pt**1.5)*(an_r(13)/const3)                  &
     &            *(0.5/sqrt(an_r(3)+1.0d-50))

        d_eq(13,4) = d_eq(13,4) + pt/an_t
        d_eq(13,13) = d_eq(13,13)                                       &
     &            -k_eq(7)*(pt**1.5)*(sqrt(an_r(3)+1.0d-50)/const3)

        d_eq(14,1) = pt*an_r(15)*(-1.0)/const2                          &
     &             -k_eq(8)*(pt**1.5)*sqrt(an_r(3)+1.0d-50)             &
     &             *(an_r(9))*(-1.5/const5)

        do jj = 2,26
           d_eq(14,jj) = d_eq(14,1)
        enddo

        d_eq(14,3) = d_eq(14,3)                                         &
     &            -k_eq(8)*(pt**1.5)*(an_r(9)/const3)                   &
     &            *(0.5/sqrt(an_r(3)+1.0d-50))
        d_eq(14,9) = d_eq(14,9)                                         &
     &            -k_eq(8)*(pt**1.5)*(sqrt(an_r(3)+1.0d-50)/const3)
        d_eq(14,15) = d_eq(14,15)+ pt/an_t

        d_eq(15,1) = pt*an_r(16)*(-1.0)/const2                          &
     &             -k_eq(9)*(pt**1.5)*sqrt(an_r(14)+1.0d-50)            &
     &             *(an_r(3))*(-1.5/const5)

        do jj = 2,26
           d_eq(15,jj) = d_eq(15,1)
        enddo

        d_eq(15,3) = d_eq(15,3)                                         &
     &            -k_eq(9)*(pt**1.5)*(sqrt(an_r(14)+1.0d-50)/const3)
        d_eq(15,14) = d_eq(15,14)                                       &
     &            -k_eq(9)*(pt**1.5)*(an_r(3)/const3)                   &
     &            *(0.5/sqrt(an_r(14)+1.0d-50))
        d_eq(15,16) = d_eq(15,16) + pt/an_t

        d_eq(16,1) = pt*an_r(12)*(-1.0)/const2                          &
     &             -k_eq(10)*(pt**1.5)*sqrt(an_r(3)+1.0d-50)            &
     &             *(an_r(6))*(-1.5/const5)

        do jj = 2,26
           d_eq(16,jj) = d_eq(16,1)
        enddo

        d_eq(16,3) = d_eq(16,3)                                         &
     &             -k_eq(10)*(pt**1.5)*(an_r(6)/const3)                 &
     &             *(0.5/sqrt(an_r(3)+1.0d-50))

        d_eq(16,6) = d_eq(16,6)                                         &
     &             -k_eq(10)*(pt**1.5)*(sqrt(an_r(3)+1.0d-50)/const3)
        d_eq(16,12) = d_eq(16,12) + pt/an_t

        const_cube =  an_t*an_t*an_t
        const_four =  const2*const2

        d_eq(17,1) = an_r(14)*an_r(18)*an_r(18)*(pt**3)*(-3/const_four) &
     &             - k_eq(15) * an_r(17)*pt * (-1/const2)
        do jj = 2,26
           d_eq(17,jj) = d_eq(17,1)
        enddo
        d_eq(17,14) = d_eq(17,14) + an_r(18)*an_r(18)*(pt**3)/const_cube
        d_eq(17,17) = d_eq(17,17) - k_eq(15)*pt/an_t
        d_eq(17,18) = d_eq(17,18) + 2*an_r(18)*an_r(14)                 &
     &                            *(pt**3)/const_cube

        d_eq(18,1) = an_r(13)*an_r(13)*(pt**2)*(-2/const_cube)          &
     &             - k_eq(16) * an_r(3)*an_r(18)*an_r(18)               &
     &              * (pt*pt*pt) * (-3/const_four)
        do jj = 2,26
           d_eq(18,jj) = d_eq(18,1)
        enddo
        d_eq(18,3) = d_eq(18,3)                                         &
     &             - k_eq(16) *an_r(18)* an_r(18)*pt*pt*pt /const_cube
        d_eq(18,13) = d_eq(18,13)                                       &
     &              + 2* an_r(13)*pt*pt /const2
        d_eq(18,18) = d_eq(18,18) -k_eq(16)*an_r(3)                     &
     &              * 2*an_r(18)*pt*pt*pt/const_cube

!====for eq 19

        d_eq(19,1) = an_r(3)*an_r(19)*(pt**2)*(-2/const_cube)           &
     &             - k_eq(17)*an_r(13)*an_r(10)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(19,jj) = d_eq(19,1)
        enddo
        d_eq(19,13) = d_eq(19,13)                                       &
     &             - k_eq(17) *an_r(10)*pt*pt /const2
        d_eq(19,10) = d_eq(19,10)                                       &
     &             - k_eq(17) *an_r(13)*pt*pt /const2
        d_eq(19,3) = d_eq(19,3) + an_r(19)*pt*pt/const2
        d_eq(19,19) = d_eq(19,19) + an_r(3)*pt*pt/const2
!====for eq 20

        d_eq(20,1) = an_r(21)*an_r(20)*(pt**2)*(-2/const_cube)          &
     &             - k_eq(18) * an_r(19)*an_r(8)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(20,jj) = d_eq(20,1)
        enddo
        d_eq(20,8) = d_eq(20,8)                                         &
     &             - k_eq(18) *an_r(19)*pt*pt /const2
        d_eq(20,19) = d_eq(20,19)                                       &
     &             - k_eq(18) *an_r(8)*pt*pt /const2
        d_eq(20,20) = d_eq(20,20) + an_r(21)*pt*pt/const2
        d_eq(20,21) = d_eq(20,21) + an_r(20)*pt*pt/const2

!========
!====for eq 21

        d_eq(21,1) = an_r(21)*an_r(23)*(pt**2)*(-2/const_cube)          &
     &             - k_eq(19)*an_r(7)*an_r(8)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(21,jj) = d_eq(21,1)
        enddo
        d_eq(21,7) = d_eq(21,7)                                         &
     &             - k_eq(19) *an_r(8)*pt*pt /const2
        d_eq(21,8) = d_eq(21,8)                                         &
     &             - k_eq(19) *an_r(7)*pt*pt /const2
        d_eq(21,21) = d_eq(21,21) + an_r(23)*pt*pt/const2
        d_eq(21,23) = d_eq(21,23) + an_r(21)*pt*pt/const2

!========
!  for 22
        d_eq(22,1) = an_r(5)*an_r(11)*(pt**2)*(-2/const_cube)           &
     &         -k_eq(20)*an_r(21)*an_r(22)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(22,jj) = d_eq(22,1)
        enddo
        d_eq(22,21) = d_eq(22,21)                                       &
     &             - k_eq(20) *an_r(22)*pt*pt /const2
        d_eq(22,22) = d_eq(22,22)                                       &
     &             - k_eq(20) *an_r(21)*pt*pt /const2
        d_eq(22,11) = d_eq(22,11) + an_r(5)*pt*pt/(const2)
        d_eq(22,5) = d_eq(22,5) + an_r(11)*pt*pt/(const2)

!========
!  for 23

        d_eq(23,1) = an_r(24)*(pt)*(-1/const2)                          &
     &             - k_eq(21)*an_r(21)*an_r(3)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(23,jj) = d_eq(23,1)
        enddo
        d_eq(23,3) = d_eq(23,3)                                         &
     &             - k_eq(21) *an_r(21)*pt*pt /const2
        d_eq(23,21) = d_eq(23,21)                                       &
     &             - k_eq(21) *an_r(3)*pt*pt /const2
        d_eq(23,24) = d_eq(23,24) + pt/(an_t)

!========
!  for 24
        d_eq(24,1) = an_r(3)*an_r(25)*(pt**2)*(-2/const_cube)           &
     &             - k_eq(22)*an_r(24)*an_r(8)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(24,jj) = d_eq(24,1)
        enddo
        d_eq(24,8) = d_eq(24,8)                                         &
     &             - k_eq(22) *an_r(24)*pt*pt /const2
        d_eq(24,24) = d_eq(24,24)                                       &
     &             - k_eq(22) *an_r(8)*pt*pt /const2
        d_eq(24,3) = d_eq(24,3) + an_r(25)*pt*pt/const2
        d_eq(24,25) = d_eq(24,25) + an_r(3)*pt*pt/const2

!========
!for 25

        d_eq(25,1) = an_r(26)*(pt)*(-1/const2)                          &
     &       - k_eq(23)*an_r(21)*an_r(10)*pt*pt * (-2/const_cube)
        do jj = 2,26
           d_eq(25,jj) = d_eq(25,1)
        enddo
        d_eq(25,10) = d_eq(25,10)                                       &
     &             - k_eq(23) *an_r(21)*pt*pt /const2
        d_eq(25,21) = d_eq(25,21)                                       &
     &             - k_eq(23) *an_r(10)*pt*pt /const2
        d_eq(25,26) = d_eq(25,26) + pt/(an_t)

!============
!  for 26
        d_eq(26,20) = -1
        d_eq(26,22) = -1
        d_eq(26,23) = -1
        d_eq(26,21) = 1
        d_eq(26,24) = 1
        d_eq(26,25) = 1
        d_eq(26,26) = 1

           do j = 1,26
         do i = 1,26
                write(44,*)i,j,d_eq(i,j)
              enddo
           enddo

        return
        end

      subroutine ShashiPostCheck(ls,X,Y,W,c_Y,c_W,dummy)
#include <petsc/finclude/petscsnes.h>
      use petscsnes
      implicit none
      SNESLineSearch ls
      PetscErrorCode ierr
      Vec X,Y,W
      PetscObject dummy
      PetscBool c_Y,c_W
      PetscScalar,pointer :: xx(:)
      PetscInt i
      PetscCall(VecGetArrayF90(W,xx,ierr))
      do i=1,26
         if (xx(i) < 0.0) then
            xx(i) = 0.0
            c_W = PETSC_TRUE
         endif
        if (xx(i) > 3.0) then
           xx(i) = 3.0
        endif
      enddo
      PetscCall(VecRestoreArrayF90(W,xx,ierr))
      return
      end
