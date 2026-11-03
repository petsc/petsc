#pragma once

#define KSPGMRES_NO_MACROS
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

typedef struct {
  KSPGMRESHEADER
} KSP_PGMRES;

#define HH(a, b) (pgmres->hh_origin + (b) * (pgmres->restart_size + 2) + (a))
/* HH will be size (restart_size+2)*(restart_size+1)  -  think of HH as being stored columnwise for access purposes. */
#define HES(a, b) (pgmres->hes_origin + (b) * (pgmres->restart_size + 1) + (a))
/* HES will be size (restart_size + 1) * (restart_size + 1) -  again, think of HES as being stored columnwise */
#define CC(a) (pgmres->cc_origin + (a)) /* CC will be length (restart_size+1) - cosines */
#define SS(a) (pgmres->ss_origin + (a)) /* SS will be length (restart_size+1) - sines */
#define RS(a) (pgmres->rs_origin + (a)) /* RS will be length (restart_size+2) - rt side */

/* vector names */
#define VEC_OFFSET     2
#define VEC_TEMP       pgmres->vecs[0]                /* work space */
#define VEC_TEMP_MATOP pgmres->vecs[1]                /* work space */
#define VEC_VV(i)      pgmres->vecs[VEC_OFFSET + (i)] /* use to access othog basis vectors */
