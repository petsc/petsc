#include <petsc-private/petscimpl.h>

/* ---------------------------------------------------------------------------------------*/
#if defined(PETSC_HAVE_SETJMP_H) && defined(PETSC_HAVE_SIGINFO_T)
#include <signal.h>
#include <setjmp.h>
PETSC_EXTERN jmp_buf PetscSegvJumpBuf;
PETSC_EXTERN void PetscSegv_sigaction(int, siginfo_t*, void *);
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
      PETSC_UNUSED char *x = *(char*volatile*)ptr;
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
