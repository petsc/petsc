#include <petsc/private/petscimpl.h>
#include <petscvalgrind.h>

static PetscInt petsc_checkpointer_intensity = 1;

/*@
   PetscCheckPointerSetIntensity - An intense pointer check registers a signal handler and attempts to dereference to
   confirm whether the address is valid.  An intensity of 0 never uses signal handlers, 1 uses them when not in a "hot"
   function, and intensity of 2 always uses a signal handler.

   Not Collective

   Input Arguments:
.  intensity - how much to check pointers for validity

   Level: advanced

.seealso: PetscCheckPointer(), PetscFunctionBeginHot
@*/
PetscErrorCode PetscCheckPointerSetIntensity(PetscInt intensity)
{

  PetscFunctionBegin;
  switch (intensity) {
  case 0:
  case 1:
  case 2:
    petsc_checkpointer_intensity = intensity;
    break;
  default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Intensity %D not in 0,1,2",intensity);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#if defined(PETSC_HAVE_SETJMP_H) && defined(PETSC_HAVE_SIGINFO_T)
#include <signal.h>
#include <setjmp.h>
PETSC_INTERN jmp_buf PetscSegvJumpBuf;
PETSC_INTERN void PetscSegv_sigaction(int, siginfo_t*, void *);

/*@C
     PetscCheckPointer - Returns PETSC_TRUE if a pointer points to accessible data

   Not Collective

   Input Parameters:
+     ptr - the pointer
-     dtype - the type of data the pointer is suppose to point to

   Level: developer

@*/
PetscBool PetscCheckPointer(const void *ptr,PetscDataType dtype)
{
  struct sigaction sa,oldsa;

  if (PETSC_RUNNING_ON_VALGRIND) return PETSC_TRUE;
  if (!ptr) return PETSC_FALSE;
  if (petsc_checkpointer_intensity < 1) return PETSC_TRUE;

  /* Skip the verbose check if we are inside a hot function. */
  if (petscstack && petscstack->hotdepth > 0 && petsc_checkpointer_intensity < 2) return PETSC_TRUE;

  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = PetscSegv_sigaction;
  sa.sa_flags   = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, &oldsa);

  if (setjmp(PetscSegvJumpBuf)) {
    /* A segv was triggered in the code below hence we return with an error code */
    sigaction(SIGSEGV, &oldsa, NULL);/* reset old signal hanlder */
    return PETSC_FALSE;
  } else {
    switch (dtype) {
    case PETSC_INT:{
      PETSC_UNUSED PetscInt x = (PetscInt)*(volatile PetscInt*)ptr;
      break;
    }
#if defined(PETSC_USE_COMPLEX)
    case PETSC_SCALAR:{         /* C++ is seriously dysfunctional with volatile std::complex. */
      PetscReal xreal = ((volatile PetscReal*)ptr)[0],ximag = ((volatile PetscReal*)ptr)[1];
      PETSC_UNUSED volatile PetscScalar x = xreal + PETSC_i*ximag;
      break;
    }
#endif
    case PETSC_REAL:{
      PETSC_UNUSED PetscReal x = *(volatile PetscReal*)ptr;
      break;
    }
    case PETSC_BOOL:{
      PETSC_UNUSED PetscBool x = *(volatile PetscBool*)ptr;
      break;
    }
    case PETSC_ENUM:{
      PETSC_UNUSED PetscEnum x = *(volatile PetscEnum*)ptr;
      break;
    }
    case PETSC_CHAR:{
      PETSC_UNUSED char x = *(volatile char*)ptr;
      break;
    }
    case PETSC_OBJECT:{
      PETSC_UNUSED volatile PetscClassId classid = ((PetscObject)ptr)->classid;
      break;
    }
    default:;
    }
  }
  sigaction(SIGSEGV, &oldsa, NULL); /* reset old signal hanlder */
  return PETSC_TRUE;
}
#else
PetscBool PetscCheckPointer(const void *ptr,PETSC_UNUSED PetscDataType dtype)
{
  if (!ptr) return PETSC_FALSE;
  return PETSC_TRUE;
}
#endif
