!
!  $Id: petsclog.h,v 1.19 2001/04/10 23:57:15 balay Exp $;

#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Include file for Fortran use of the Plog package in PETSc
!

       integer    MAT_Mult  
       parameter (MAT_Mult = 0)
       integer    MAT_MatrixFreeMult 
       parameter (MAT_MatrixFreeMult = 1)
       integer    MAT_AssemblyBegin
       parameter (MAT_AssemblyBegin = 2)
       integer    MAT_AssemblyEnd
       parameter (MAT_AssemblyEnd = 3)
       integer    MAT_GetOrdering
       parameter (MAT_GetOrdering = 4)
       integer    MAT_MultTranspose
       parameter (MAT_MultTranspose = 5)
       integer    MAT_MultAdd
       parameter (MAT_MultAdd = 6)
       integer    MAT_MultTransposeAdd
       parameter (MAT_MultTransposeAdd = 7)
       integer    MAT_LUFactor
       parameter (MAT_LUFactor = 8)
       integer    MAT_CholeskyFactor
       parameter (MAT_CholeskyFactor = 9)
       integer    MAT_LUFactorSymbolic
       parameter (MAT_LUFactorSymbolic = 10)
       integer    MAT_ILUFactorSymbolic
       parameter (MAT_ILUFactorSymbolic = 11)
       integer    MAT_CholeskyFactorSymbolic
       parameter (MAT_CholeskyFactorSymbolic = 12)
       integer    MAT_ICCFactorSymbolic
       parameter (MAT_ICCFactorSymbolic = 13)
       integer    MAT_LUFactorNumeric
       parameter (MAT_LUFactorNumeric = 14)
       integer    MAT_CholeskyFactorNumeric
       parameter (MAT_CholeskyFactorNumeric = 15)
       integer    MAT_Relax
       parameter (MAT_Relax = 16)
       integer    MAT_Copy
       parameter (MAT_Copy = 17)
       integer    MAT_Convert
       parameter (MAT_Convert = 18)
       integer    MAT_Scale
       parameter (MAT_Scale = 19)
       integer    MAT_ZeroEntries
       parameter (MAT_ZeroEntries = 20)
       integer    MAT_Solve
       parameter (MAT_Solve = 21)
       integer    MAT_SolveAdd
       parameter (MAT_SolveAdd = 22)
       integer    MAT_SolveTranspose   
       parameter (MAT_SolveTranspose = 23)
       integer    MAT_SolveTransposeAdd  
       parameter (MAT_SolveTransposeAdd = 24)
       integer    MAT_SetValues
       parameter (MAT_SetValues = 25)
       integer    MAT_ForwardSolve
       parameter (MAT_ForwardSolve = 26)
       integer    MAT_BackwardSolve 
       parameter (MAT_BackwardSolve  = 27)
       integer    MAT_Load
       parameter (MAT_Load = 28)
       integer    MAT_View
       parameter (MAT_View = 29)
       integer    MAT_ILUFactor
       parameter (MAT_ILUFactor = 30)
       integer    MAT_GetColoring
       parameter (MAT_GetColoring = 31)
       integer    MAT_GetSubMatrices
       parameter (MAT_GetSubMatrices = 32)
       integer    MAT_GetValues
       parameter (MAT_GetValues = 33)
       integer    MAT_IncreaseOverlap
       parameter (MAT_IncreaseOverlap = 34)
       integer    MAT_GetRow
       parameter (MAT_GetRow = 35)
       integer    MAT_GetPartitioning
       parameter (MAT_GetPartitioning = 36)

       integer    MAT_FDColoringApply
       parameter (MAT_FDColoringApply = 38)

       integer    MAT_FDColoringCreate
       parameter (MAT_FDColoringCreate = 41)

       integer    VEC_ReduceArithmetic
       parameter (VEC_ReduceArithmetic = 37)
       integer    VEC_View
       parameter (VEC_View = 39)

       integer    VEC_Max
       parameter (VEC_Max = 42)
       integer    VEC_Min
       parameter (VEC_Min = 43)
       integer    VEC_TDot
       parameter (VEC_TDot = 44)
       integer    VEC_Scale
       parameter (VEC_Scale = 45)
       integer    VEC_Copy
       parameter (VEC_Copy = 46)
       integer    VEC_Set
       parameter (VEC_Set = 47)
       integer    VEC_AXPY
       parameter (VEC_AXPY = 48)
       integer    VEC_AYPX
       parameter (VEC_AYPX = 49)
       integer    VEC_Swap
       parameter (VEC_Swap = 50)
       integer    VEC_WAXPY
       parameter (VEC_WAXPY = 51)
       integer    VEC_AssemblyBegin
       parameter (VEC_AssemblyBegin = 52)
       integer    VEC_AssemblyEnd
       parameter (VEC_AssemblyEnd = 53)
       integer    VEC_MTDot
       parameter (VEC_MTDot = 54)

       integer    VEC_MAXPY
       parameter (VEC_MAXPY = 56)
       integer    VEC_PMult
       parameter (VEC_PMult = 57)
       integer    VEC_SetValues
       parameter (VEC_SetValues = 58)
       integer    VEC_Load
       parameter (VEC_Load = 59)
       integer    VEC_ScatterBarrier
       parameter (VEC_ScatterBarrier = 60)
       integer    VEC_ScatterBegin
       parameter (VEC_ScatterBegin = 61)

       integer    VEC_ScatterEnd
       parameter (VEC_ScatterEnd = 62)
       integer    VEC_SetRandom
       parameter (VEC_SetRandom = 63)
       integer    VEC_NormBarrier
       parameter (VEC_NormBarrier = 64)    
       integer    VEC_Norm
       parameter (VEC_Norm = 65)    
       integer    VEC_DotBarrier
       parameter (VEC_DotBarrier = 66)    
       integer    VEC_Dot
       parameter (VEC_Dot = 67)    
       integer    VEC_MDotBarrier
       parameter (VEC_MDotBarrier = 68)    
       integer    VEC_MDot
       parameter (VEC_MDot = 69)    


       integer    SLES_Solve
       parameter (SLES_Solve = 70)
       integer    SLES_SetUp
       parameter (SLES_SetUp = 71)
       integer    KSP_GMRESOrthogonalization
       parameter (KSP_GMRESOrthogonalization = 72)
       integer    PC_ApplyCoarse
       parameter (PC_ApplyCoarse = 73)
       integer    PC_ModifySubMatrices
       parameter (PC_ModifySubMatrices = 74)
       integer    PC_SetUp
       parameter (PC_SetUp = 75)
       integer    PC_SetUpOnBlocks
       parameter (PC_SetUpOnBlocks = 76)
       integer    PC_Apply
       parameter (PC_Apply = 77)
       integer    PC_ApplySymmetricLeft
       parameter (PC_ApplySymmetricLeft = 78)
       integer    PC_ApplySymmetricRight
       parameter (PC_ApplySymmetricRight = 79)

       integer    SNES_Solve
       parameter (SNES_Solve = 80)
       integer    SNES_LineSearch
       parameter (SNES_LineSearch = 81)
       integer    SNES_FunctionEval
       parameter (SNES_FunctionEval = 82)
       integer    SNES_JacobianEval
       parameter (SNES_JacobianEval = 83)
       integer    SNES_MinimizationFunctionEval
       parameter (SNES_MinimizationFunctionEval = 84)
       integer    SNES_GradientEval
       parameter (SNES_GradientEval = 85)
       integer    SNES_HessianEval
       parameter (SNES_HessianEval = 86)

       integer    VEC_ReduceBarrier
       parameter (VEC_ReduceBarrier = 87)
       integer    VEC_ReduceComm
       parameter (VEC_ReduceComm = 88)

       integer    TS_Step
       parameter (TS_Step = 90)
       integer    TS_PseudoComputeTimeStep
       parameter (TS_PseudoComputeTimeStep = 91)
       integer    TS_FunctionEval
       parameter (TS_FunctionEval = 92)
       integer    TS_JacobianEval
       parameter (TS_JacobianEval = 93)

       integer    Petsc_Barrier
       parameter (Petsc_Barrier = 100)

       integer    EC_SetUp
       parameter (EC_SetUp = 105)
       integer    EC_Solve
       parameter (EC_Solve = 106)

#endif
