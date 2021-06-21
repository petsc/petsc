#include <../src/sys/classes/draw/impls/x/ximpl.h>         /*I  "petscsys.h" I*/

#if defined(PETSC_HAVE_SETJMP_H)

jmp_buf PetscXIOErrorHandlerJumpBuf;

void PetscXIOErrorHandlerJump(PETSC_UNUSED void *ctx)
{
  longjmp(PetscXIOErrorHandlerJumpBuf, 1);
}

PetscXIOErrorHandler PetscSetXIOErrorHandler(PetscXIOErrorHandler xioerrhdl)
{
  return (PetscXIOErrorHandler)XSetIOErrorHandler((XIOErrorHandler)xioerrhdl);
}

#endif
