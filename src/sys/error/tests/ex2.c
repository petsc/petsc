static char help[] = "Tests checking pointers.\n\n";

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petscvalgrind.h>

int main(int argc, char *args[])
{
  PetscErrorCode ierr;
  PetscInt *ptr;

  ierr = PetscInitialize(&argc, &args, (char*) 0, help);
  if (ierr) return ierr;
  if (!PETSC_RUNNING_ON_VALGRIND) {                         /* PetscCheckPointer always returns TRUE when running on Valgrind */
    ierr = PetscMalloc(1024 * 1024 * 8,&ptr);CHKERRQ(ierr); /* Almost certainly larger than MMAP_THRESHOLD (128 KiB by default) */
    if (!PetscCheckPointer(ptr,PETSC_INT)) {ierr = PetscPrintf(PETSC_COMM_SELF,"Mistook valid pointer %p for invalid pointer\n",(void*)ptr);CHKERRQ(ierr);}
    ierr = PetscFree(ptr);CHKERRQ(ierr);
    if (PetscCheckPointer(ptr,PETSC_INT)) {ierr = PetscPrintf(PETSC_COMM_SELF,"Mistook NULL pointer for valid pointer\n");CHKERRQ(ierr);}
    ptr = (PetscInt*) ~(PETSC_UINTPTR_T)0xf; /* Pointer will almost certainly be invalid */
    if (PetscCheckPointer(ptr,PETSC_INT)) {ierr = PetscPrintf(PETSC_COMM_SELF,"Mistook invalid pointer %p for valid\n",(void*)ptr);CHKERRQ(ierr);}
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -check_pointer_intensity 1
     TODO: reports Mistook invalid pointer 0xfffffffffffffff0 for valid or Free'd pointer is still accessible

TEST*/
