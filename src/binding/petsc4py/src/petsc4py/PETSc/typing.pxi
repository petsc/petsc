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
