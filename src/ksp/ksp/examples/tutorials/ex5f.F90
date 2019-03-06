      !Solves two linear systems in parallel with KSP.  The code
      !illustrates repeated solution of linear systems with the same preconditioner
      !method but different matrices (having the same nonzero structure).  The code
      !also uses multiple profiling stages.  Input arguments are
      !  -m <size> : problem size
      !  -mat_nonsym : use nonsymmetric matrix (default is symmetric)

      !Concepts: KSP^repeatedly solving linear systems;
      !Concepts: PetscLog^profiling multiple stages of code;
      !Processors: n

program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      
      implicit none
      KSP            :: myKsp            ! linear solver context 
      Mat            :: C,Ctmp           ! matrix 
      Vec            :: x,u,b            ! approx solution, RHS, exact solution 
      PetscReal      :: norm             ! norm of solution error 
      PetscScalar    :: v
      PetscScalar, parameter :: myNone = -1.0
      PetscInt       :: Ii,JJ,ldim,low,high,iglobal,Istart,Iend
      PetscErrorCode :: ierr
      PetscInt       :: i,j,its,n
      PetscInt, parameter :: m = 3
      PetscMPIInt    :: mySize,myRank
      PetscBool, parameter :: &
        testnewC         = PETSC_FALSE, &
        testscaledMat    = PETSC_FALSE, &
        mat_nonsymmetric = PETSC_FALSE
      PetscBool      :: flg
      PetscRandom    :: rctx
      PetscLogStage,dimension(2) :: stages
      character(len=80)          :: outputString
      
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr /= 0) then
        write(6,*)'Unable to initialize PETSc'
        stop
      endif
      
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      CHKERRA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,myRank,ierr)
      CHKERRA(ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr)
      CHKERRA(ierr)
      n=2*mySize

      
      ! Set flag if we are doing a nonsymmetric problem; the default is symmetric.
      
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-mat_nonsym",mat_nonsymmetric,flg,ierr)
      CHKERRA(ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-test_scaledMat",testscaledMat,flg,ierr)
      CHKERRA(ierr)
      
      ! Register two stages for separate profiling of the two linear solves.
      ! Use the runtime option -log_view for a printout of performance
      ! statistics at the program's conlusion.

      call PetscLogStageRegister("Original Solve",stages(1),ierr)
      CHKERRA(ierr)
      call PetscLogStageRegister("Second Solve",stages(2),ierr)
      CHKERRA(ierr)
      
      ! -------------- Stage 0: Solve Original System ---------------------- 
      ! Indicate to PETSc profiling that we're beginning the first stage
      
      call PetscLogStagePush(stages(1),ierr)
      CHKERRA(ierr)
      
      ! Create parallel matrix, specifying only its global dimensions.
      ! When using MatCreate(), the matrix format can be specified at
      ! runtime. Also, the parallel partitioning of the matrix is
      ! determined by PETSc at runtime.
      
      call MatCreate(PETSC_COMM_WORLD,C,ierr)
      CHKERRA(ierr)
      call MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr)
      CHKERRA(ierr)
      call MatSetFromOptions(C,ierr)
      CHKERRA(ierr)
      call MatSetUp(C,ierr)
      CHKERRA(ierr)

      ! Currently, all PETSc parallel matrix formats are partitioned by
      ! contiguous chunks of rows across the processors.  Determine which
      ! rows of the matrix are locally owned.
      
      call MatGetOwnershipRange(C,Istart,Iend,ierr)
      
      ! Set matrix entries matrix in parallel.
      ! - Each processor needs to insert only elements that it owns
      ! locally (but any non-local elements will be sent to the
      ! appropriate processor during matrix assembly).
      !- Always specify global row and columns of matrix entries.

      intitializeC: do Ii=Istart,Iend-1
          v =-1.0; i = Ii/n; j = Ii - i*n
          if (i>0) then
            JJ = Ii - n
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
          
          if (i<m-1) then
            JJ = Ii + n
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
      
          if (j>0) then
            JJ = Ii - 1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
      
          if (j<n-1) then
            JJ = Ii + 1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
          
          v=4.0
          call MatSetValues(C,1,Ii,1,Ii,v,ADD_VALUES,ierr)
          CHKERRA(ierr)
          
      enddo intitializeC
      
      ! Make the matrix nonsymmetric if desired
      if (mat_nonsymmetric) then
        do Ii=Istart,Iend-1
          v=-1.5; i=Ii/n
          if (i>1) then
            JJ=Ii-n-1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
        enddo
      else
        call MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE,ierr)
        CHKERRA(ierr)
        call MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE,ierr)
        CHKERRA(ierr)
      endif
      
      ! Assemble matrix, using the 2-step process:
      ! MatAssemblyBegin(), MatAssemblyEnd()
      ! Computations can be done while messages are in transition
      ! by placing code between these two statements.
      
      call MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRA(ierr)
      call MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRA(ierr)

      ! Create parallel vectors.
      ! - When using VecSetSizes(), we specify only the vector's global
      !   dimension; the parallel partitioning is determined at runtime.
      ! - Note: We form 1 vector from scratch and then duplicate as needed.
      
      call  VecCreate(PETSC_COMM_WORLD,u,ierr)
      call  VecSetSizes(u,PETSC_DECIDE,m*n,ierr)
      call  VecSetFromOptions(u,ierr)
      call  VecDuplicate(u,b,ierr)
      call  VecDuplicate(b,x,ierr)

      ! Currently, all parallel PETSc vectors are partitioned by
      ! contiguous chunks across the processors.  Determine which
      ! range of entries are locally owned.
      
      call  VecGetOwnershipRange(x,low,high,ierr)
      CHKERRA(ierr)

      !Set elements within the exact solution vector in parallel.
      ! - Each processor needs to insert only elements that it owns
      ! locally (but any non-local entries will be sent to the
      ! appropriate processor during vector assembly).
      ! - Always specify global locations of vector entries.

      call VecGetLocalSize(x,ldim,ierr) 
      CHKERRA(ierr)
      do i=0,ldim-1
        iglobal = i + low
        v = real(i + 100*myRank)
        call VecSetValues(u,1,iglobal,v,INSERT_VALUES,ierr)
        CHKERRA(ierr)
      enddo

      ! Assemble vector, using the 2-step process:
      ! VecAssemblyBegin(), VecAssemblyEnd()
      ! Computations can be done while messages are in transition,
      ! by placing code between these two statements.      
      
      call  VecAssemblyBegin(u,ierr)
      CHKERRA(ierr)
      call  VecAssemblyEnd(u,ierr)
      CHKERRA(ierr)
      
      ! Compute right-hand-side vector
      
      call  MatMult(C,u,b,ierr)

      CHKERRA(ierr)
      
      ! Create linear solver context

      call  KSPCreate(PETSC_COMM_WORLD,myKsp,ierr)
      CHKERRA(ierr)
      ! Set operators. Here the matrix that defines the linear system
      ! also serves as the preconditioning matrix.

      call  KSPSetOperators(myKsp,C,C,ierr)
      CHKERRA(ierr)
      ! Set runtime options (e.g., -ksp_type <type> -pc_type <type>)

      call  KSPSetFromOptions(myKsp,ierr)
      CHKERRA(ierr)
      ! Solve linear system.  Here we explicitly call KSPSetUp() for more
      ! detailed performance monitoring of certain preconditioners, such
      ! as ICC and ILU.  This call is optional, as KSPSetUp() will
      ! automatically be called within KSPSolve() if it hasn't been
      ! called already.

      call  KSPSetUp(myKsp,ierr)
      CHKERRA(ierr)

      call  KSPSolve(myKsp,b,x,ierr)
      CHKERRA(ierr)

      ! Check the error
      
      call VecAXPY(x,myNone,u,ierr)
      call VecNorm(x,NORM_2,norm,ierr)

      call KSPGetIterationNumber(myKsp,its,ierr)
      if (.not. testscaledMat .or. norm > 1.e-7) then
        write(outputString,*)'Norm of error',norm,'Iterations',its,'\n'
        call PetscPrintf(PETSC_COMM_WORLD,outputString,ierr)
      endif
      
      ! -------------- Stage 1: Solve Second System ---------------------- 
      
      ! Solve another linear system with the same method.  We reuse the KSP
      ! context, matrix and vector data structures, and hence save the
      ! overhead of creating new ones.

      ! Indicate to PETSc profiling that we're concluding the first
      ! stage with PetscLogStagePop(), and beginning the second stage with
      ! PetscLogStagePush().
      
      call PetscLogStagePop(ierr)
      CHKERRA(ierr)      
      call PetscLogStagePush(stages(2),ierr)
      CHKERRA(ierr)
      
      ! Initialize all matrix entries to zero.  MatZeroEntries() retains the
      ! nonzero structure of the matrix for sparse formats.
      
      call MatZeroEntries(C,ierr)
      CHKERRA(ierr)
      
      ! Assemble matrix again.  Note that we retain the same matrix data
      ! structure and the same nonzero pattern; we just change the values
      ! of the matrix entries.
      
      do i=0,m-1
        do j=2*myRank,2*myRank+1
          v =-1.0; Ii=j + n*i
          if (i>0) then
            JJ = Ii - n
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
          
          if (i<m-1) then
            JJ = Ii + n
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
      
          if (j>0) then
            JJ = Ii - 1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
      
          if (j<n-1) then
            JJ = Ii + 1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
          
          v=6.0
          call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
          CHKERRA(ierr)
          
        enddo
      enddo
        
      ! Make the matrix nonsymmetric if desired
      
      if (mat_nonsymmetric) then
        do Ii=Istart,Iend-1
          v=-1.5;  i=Ii/n
          if (i>1) then
            JJ=Ii-n-1
            call MatSetValues(C,1,Ii,1,JJ,v,ADD_VALUES,ierr)
            CHKERRA(ierr)
          endif
        enddo
      else 
          call MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE,ierr)
          CHKERRA(ierr)
          call MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE,ierr)
          CHKERRA(ierr)
      endif
      
      ! Assemble matrix, using the 2-step process:
      ! MatAssemblyBegin(), MatAssemblyEnd()
      ! Computations can be done while messages are in transition
      ! by placing code between these two statements.
      
      call MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRA(ierr)
      call MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr)
      CHKERRA(ierr)
      
      if (testscaledMat) then
        ! Scale a(0,0) and a(M-1,M-1)

        if (myRank /= 0) then
          v = 6.0*0.00001; Ii = 0; JJ = 0
          call MatSetValues(C,1,Ii,1,JJ,v,INSERT_VALUES,ierr)
          CHKERRA(ierr)
        elseif (myRank == mySize -1) then
          v = 6.0*0.00001; Ii = m*n-1; JJ = m*n-1
          call MatSetValues(C,1,Ii,1,JJ,v,INSERT_VALUES,ierr)
          
        endif
        
        call MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr)
        call MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr)
        
        ! Compute a new right-hand-side vector
        
        call  VecDestroy(u,ierr)
        call  VecCreate(PETSC_COMM_WORLD,u,ierr)
        call  VecSetSizes(u,PETSC_DECIDE,m*n,ierr)
        call  VecSetFromOptions(u,ierr)

        call  PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr)
        call  PetscRandomSetFromOptions(rctx,ierr)
        call  VecSetRandom(u,rctx,ierr)
        call  PetscRandomDestroy(rctx,ierr)
        call  VecAssemblyBegin(u,ierr)
        call  VecAssemblyEnd(u,ierr)
        
      endif
      
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-test_newMat",testnewC,flg,ierr)
      CHKERRA(ierr)
      
      if (testnewC) then
      ! User may use a new matrix C with same nonzero pattern, e.g.
      ! ex5 -ksp_monitor -mat_type sbaij -pc_type cholesky -pc_factor_mat_solver_type mumps -test_newMat

        call  MatDuplicate(C,MAT_COPY_VALUES,Ctmp,ierr)
        call  MatDestroy(C,ierr)
        call  MatDuplicate(Ctmp,MAT_COPY_VALUES,C,ierr)
        call  MatDestroy(Ctmp,ierr)
      endif
      
      call MatMult(C,u,b,ierr);CHKERRA(ierr)

      ! Set operators. Here the matrix that defines the linear system
      ! also serves as the preconditioning matrix.
      
      call KSPSetOperators(myKsp,C,C,ierr);CHKERRA(ierr)
      
      ! Solve linear system
       
      call  KSPSetUp(myKsp,ierr); CHKERRA(ierr)
      call  KSPSolve(myKsp,b,x,ierr); CHKERRA(ierr)
      
      ! Check the error
      
      call VecAXPY(x,myNone,u,ierr); CHKERRA(ierr)
      call VecNorm(x,NORM_2,norm,ierr); CHKERRA(ierr)
      call KSPGetIterationNumber(myKsp,its,ierr); CHKERRA(ierr)
      if (.not. testscaledMat .or. norm > 1.e-7) then
        write(outputString,*)'Norm of error',norm,'Iterations',its,'\n'
        call PetscPrintf(PETSC_COMM_WORLD,outputString,ierr)
      endif
      
      ! Free work space.  All PETSc objects should be destroyed when they
      ! are no longer needed.
      
      call  KSPDestroy(myKsp,ierr); CHKERRA(ierr)
      call  VecDestroy(u,ierr); CHKERRA(ierr)
      call  VecDestroy(x,ierr); CHKERRA(ierr) 
      call  VecDestroy(b,ierr); CHKERRA(ierr)
      call  MatDestroy(C,ierr); CHKERRA(ierr)
      
      ! Indicate to PETSc profiling that we're concluding the second stage
      
      call PetscLogStagePop(ierr) 
      CHKERRA(ierr)
      
      call PetscFinalize(ierr)
      

end program main
