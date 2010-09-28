#if !defined(PETSC4PY_COMPAT_H)
#define PETSC4PY_COMPAT_H

#include "compat/mpi.h"
#include "compat/petsc.h"
#include "compat/petscsys.h"
#include "compat/petscfwk.h"
#include "compat/petscviewer.h"
#include "compat/petscis.h"
#include "compat/petscvec.h"
#include "compat/petscmat.h"
#include "compat/petscpc.h"
#include "compat/petscksp.h"
#include "compat/petscsnes.h"
#include "compat/petscts.h"
#include "compat/petscao.h"
#include "compat/petscda.h"

#if !defined(WITH_THREAD)
#undef  PyGILState_Ensure
#define PyGILState_Ensure() ((PyGILState_STATE)0)
#undef  PyGILState_Release
#define PyGILState_Release(state) (state)=((PyGILState_STATE)0)
#undef  Py_BLOCK_THREADS
#define Py_BLOCK_THREADS (_save)=(PyThreadState*)0;
#undef  Py_UNBLOCK_THREADS
#define Py_UNBLOCK_THREADS (_save)=(PyThreadState*)0;
#endif

#endif /* !PETSC4PY_COMPAT_H */
