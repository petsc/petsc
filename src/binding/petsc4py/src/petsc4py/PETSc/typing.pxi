cdef Any
cdef Self
cdef Union
cdef Literal
cdef Optional
cdef NoReturn

cdef Callable
cdef Hashable
cdef Iterable
cdef Iterator
cdef Sequence
cdef Mapping

cdef PathLike

cdef Scalar
cdef ArrayInt
cdef ArrayReal
cdef ArrayComplex
cdef ArrayScalar

cdef DimsSpec
cdef AccessModeSpec
cdef InsertModeSpec
cdef ScatterModeSpec
cdef LayoutSizeSpec
cdef NormTypeSpec

# --- Mat ---

cdef MatAssemblySpec
cdef MatSizeSpec
cdef MatBlockSizeSpec
cdef CSRIndicesSpec
cdef CSRSpec
cdef NNZSpec

# --- MatNullSpace ---

cdef MatNullFunction

# --- DM ---

cdef DMCoarsenHookFunction
cdef DMRestrictHookFunction

# --- KSP ---

cdef KSPRHSFunction
cdef KSPOperatorsFunction
cdef KSPConvergenceTestFunction
cdef KSPMonitorFunction

# --- TS ---

cdef TSRHSFunction
cdef TSRHSJacobian
cdef TSRHSJacobianP
cdef TSIFunction
cdef TSIJacobian
cdef TSIJacobianP
cdef TSI2Function
cdef TSI2Jacobian
cdef TSMonitorFunction
cdef TSEventHandlerFunction
cdef TSPostEventFunction
cdef TSPreStepFunction
cdef TSPostStepFunction

# --- SNES ---

cdef SNESMonitorFunction
cdef SNESObjFunction
cdef SNESFunction
cdef SNESJacobianFunction
cdef SNESGuessFunction
cdef SNESUpdateFunction
cdef SNESLSPreFunction
cdef SNESNGSFunction
cdef SNESConvergedFunction

# --- TAO ---

cdef TAOObjectiveFunction
cdef TAOGradientFunction
cdef TAOObjectiveGradientFunction
cdef TAOHessianFunction
cdef TAOUpdateFunction
cdef TAOMonitorFunction
cdef TAOConvergedFunction
cdef TAOJacobianFunction
cdef TAOResidualFunction
cdef TAOJacobianResidualFunction
cdef TAOVariableBoundsFunction
cdef TAOConstraintsFunction

# --- MPI ---

cdef Intracomm
cdef Datatype
cdef Op

