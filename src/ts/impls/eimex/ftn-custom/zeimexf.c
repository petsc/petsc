#include <petsc-private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tseimexsetmaxrows_                TSEIMEXSETMAXROWS
#define tseimexsetrowcol_                 TSEIMEXSETROWCOL
#define tseimexsetordadapt_               TSEIMEXSETORDADAPT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tseimexsetmaxrows_                tseimexsetmaxrows
#define tseimexsetrowcol_                 tseimexsetrowcol
#define tseimexsetordadapt_               tseimexsetordadapt
#endif

EXTERN_C_BEGIN


EXTERN_C_END
