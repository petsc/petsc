# --------------------------------------------------------------------

DECIDE    = PETSC_DECIDE
DEFAULT   = PETSC_DEFAULT
DETERMINE = PETSC_DETERMINE

# --------------------------------------------------------------------

INFINITY  = toReal(PETSC_INFINITY)
NINFINITY = toReal(PETSC_NINFINITY)
PINFINITY = toReal(PETSC_INFINITY)

# --------------------------------------------------------------------

class InsertMode(object):
    # native
    NOT_SET_VALUES    = PETSC_NOT_SET_VALUES
    INSERT_VALUES     = PETSC_INSERT_VALUES
    ADD_VALUES        = PETSC_ADD_VALUES
    MAX_VALUES        = PETSC_MAX_VALUES
    INSERT_ALL_VALUES = PETSC_INSERT_ALL_VALUES
    ADD_ALL_VALUES    = PETSC_ADD_ALL_VALUES
    INSERT_BC_VALUES  = PETSC_INSERT_BC_VALUES
    ADD_BC_VALUES     = PETSC_ADD_BC_VALUES
    # aliases
    INSERT     = INSERT_VALUES
    ADD        = ADD_VALUES
    MAX        = MAX_VALUES
    INSERT_ALL = INSERT_ALL_VALUES
    ADD_ALL    = ADD_ALL_VALUES
    INSERT_BC  = INSERT_BC_VALUES
    ADD_BC     = ADD_BC_VALUES

# --------------------------------------------------------------------

class ScatterMode(object):
    # native
    SCATTER_FORWARD       = PETSC_SCATTER_FORWARD
    SCATTER_REVERSE       = PETSC_SCATTER_REVERSE
    SCATTER_FORWARD_LOCAL = PETSC_SCATTER_FORWARD_LOCAL
    SCATTER_REVERSE_LOCAL = PETSC_SCATTER_REVERSE_LOCAL
    SCATTER_LOCAL         = PETSC_SCATTER_LOCAL
    # aliases
    FORWARD       = SCATTER_FORWARD
    REVERSE       = SCATTER_REVERSE
    FORWARD_LOCAL = SCATTER_FORWARD_LOCAL
    REVERSE_LOCAL = SCATTER_REVERSE_LOCAL

# --------------------------------------------------------------------

class NormType(object):
    # native
    NORM_1         = PETSC_NORM_1
    NORM_2         = PETSC_NORM_2
    NORM_1_AND_2   = PETSC_NORM_1_AND_2
    NORM_FROBENIUS = PETSC_NORM_FROBENIUS
    NORM_INFINITY  = PETSC_NORM_INFINITY
    NORM_MAX       = PETSC_NORM_MAX
    # aliases
    N1        = NORM_1
    N2        = NORM_2
    N12       = NORM_1_AND_2
    MAX       = NORM_MAX
    FROBENIUS = NORM_FROBENIUS
    INFINITY  = NORM_INFINITY
    # extra aliases
    FRB = FROBENIUS
    INF = INFINITY

# --------------------------------------------------------------------
