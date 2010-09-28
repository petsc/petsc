# --------------------------------------------------------------------

DECIDE    = PETSC_DECIDE
IGNORE    = PETSC_IGNORE
DEFAULT   = PETSC_DEFAULT
DETERMINE = PETSC_DETERMINE

# --------------------------------------------------------------------

class InsertMode(object):
    # native
    NOT_SET_VALUES = PETSC_NOT_SET_VALUES
    INSERT_VALUES  = PETSC_INSERT_VALUES
    ADD_VALUES     = PETSC_ADD_VALUES
    MAX_VALUES     = PETSC_MAX_VALUES
    # aliases
    INSERT  = INSERT_VALUES
    ADD     = ADD_VALUES
    MAX     = MAX_VALUES

# --------------------------------------------------------------------

class ScatterMode(object):
    # native
    SCATTER_FORWARD = PETSC_SCATTER_FORWARD
    SCATTER_REVERSE = PETSC_SCATTER_REVERSE
    # aliases
    FORWARD = SCATTER_FORWARD
    REVERSE = SCATTER_REVERSE

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
