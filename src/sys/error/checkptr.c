#include <petsc/private/petscimpl.h>

static PetscInt petsc_checkpointer_intensity = 1;

/*@
   PetscCheckPointerSetIntensity - An intense pointer check registers a signal handler and attempts to dereference to
   confirm whether the address is valid.  An intensity of 0 never uses signal handlers, 1 uses them when not in a "hot"
   function, and intensity of 2 always uses a signal handler.

   Not Collective

   Input Parameter:
.  intensity - how much to check pointers for validity

   Options Database Key:
.  -check_pointer_intensity - intensity (0, 1, or 2)

   Level: advanced

.seealso: `PetscCheckPointer()`, `PetscFunctionBeginHot()`
@*/
PetscErrorCode PetscCheckPointerSetIntensity(PetscInt intensity)
{
  PetscFunctionBegin;
  PetscCheck((intensity >= 0) && (intensity <= 2), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Intensity %" PetscInt_FMT " not in [0,2]", intensity);
  petsc_checkpointer_intensity = intensity;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------------------------------*/

#if PetscDefined(HAVE_SETJMP_H)
  #include <setjmp.h>
static jmp_buf   PetscSegvJumpBuf;
static PetscBool PetscSegvJumpBuf_set;

/*@C
  PetscSignalSegvCheckPointerOrMpi - To be called from a signal handler for SIGSEGV.

  Not Collective

  Notes:
  If the signal was received while executing PetscCheckPointer(), this function longjmps back
  there, otherwise returns with no effect. This function is called automatically by
  PetscSignalHandlerDefault().

  Level: developer

.seealso: `PetscPushSignalHandler()`
@*/
void PetscSignalSegvCheckPointerOrMpi(void)
{
  if (PetscSegvJumpBuf_set) longjmp(PetscSegvJumpBuf, 1);
}

/*@C
     PetscCheckPointer - Returns `PETSC_TRUE` if a pointer points to accessible data

   Not Collective

   Input Parameters:
+     ptr - the pointer
-     dtype - the type of data the pointer is suppose to point to

   Level: developer

   Note:
   This is a non-standard PETSc function in that it returns the result as the return code and does not return an error code

.seealso: `PetscCheckPointerSetIntensity()`
@*/
PetscBool PetscCheckPointer(const void *ptr, PetscDataType dtype)
{
  if (PETSC_RUNNING_ON_VALGRIND) return PETSC_TRUE;
  if (!ptr) return PETSC_FALSE;
  if (petsc_checkpointer_intensity < 1) return PETSC_TRUE;

  #if PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)
  /* Skip the verbose check if we are inside a hot function. */
  if (petscstack.hotdepth > 0 && petsc_checkpointer_intensity < 2) return PETSC_TRUE;
  #endif

  PetscSegvJumpBuf_set = PETSC_TRUE;

  if (setjmp(PetscSegvJumpBuf)) {
    /* A segv was triggered in the code below hence we return with an error code */
    PetscSegvJumpBuf_set = PETSC_FALSE;
    return PETSC_FALSE;
  } else {
    switch (dtype) {
    case PETSC_INT: {
      PETSC_UNUSED PetscInt x = (PetscInt) * (volatile PetscInt *)ptr;
      break;
    }
  #if defined(PETSC_USE_COMPLEX)
    case PETSC_SCALAR: { /* C++ is seriously dysfunctional with volatile std::complex. */
    #if defined(PETSC_USE_CXXCOMPLEX)
      PetscReal                         xreal = ((volatile PetscReal *)ptr)[0], ximag = ((volatile PetscReal *)ptr)[1];
      PETSC_UNUSED volatile PetscScalar x = xreal + PETSC_i * ximag;
    #else
      PETSC_UNUSED PetscScalar x = *(volatile PetscScalar *)ptr;
    #endif
      break;
    }
  #endif
    case PETSC_REAL: {
      PETSC_UNUSED PetscReal x = *(volatile PetscReal *)ptr;
      break;
    }
    case PETSC_BOOL: {
      PETSC_UNUSED PetscBool x = *(volatile PetscBool *)ptr;
      break;
    }
    case PETSC_ENUM: {
      PETSC_UNUSED PetscEnum x = *(volatile PetscEnum *)ptr;
      break;
    }
    case PETSC_CHAR: {
      PETSC_UNUSED char x = *(volatile char *)ptr;
      break;
    }
    case PETSC_OBJECT: {
      PETSC_UNUSED volatile PetscClassId classid = ((PetscObject)ptr)->classid;
      break;
    }
    default:;
    }
  }
  PetscSegvJumpBuf_set = PETSC_FALSE;
  return PETSC_TRUE;
}
#endif
