#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h>
#endif
#define _append_pforest(a) a ## _p8est
#include "pforest.c"
