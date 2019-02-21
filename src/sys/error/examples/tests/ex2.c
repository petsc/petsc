static char help[] = "Tests checking pointers.\n\n";

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petscvalgrind.h>

int main(int argc, char *args[])
{
  PetscErrorCode ierr;
  PetscInt *ptr,*ptr2;

  ierr = PetscInitialize(&argc, &args, (char*) 0, help);
  if (ierr) return ierr;
  if (!PETSC_RUNNING_ON_VALGRIND) {                         /* PetscCheckPointer always returns TRUE when running on Valgrind */
    ierr = PetscMalloc(1024 * 1024 * 8,&ptr);CHKERRQ(ierr); /* Almost certainly larger than MMAP_THRESHOLD (128 KiB by default) */
    if (!PetscCheckPointer(ptr,PETSC_INT)) {ierr = PetscPrintf(PETSC_COMM_SELF,"Mistook valid pointer %p for invalid pointer\n",(void*)ptr);CHKERRQ(ierr);}
    ptr[0] = 0x12345678;
    ptr2 = ptr;
    ierr = PetscFree(ptr);CHKERRQ(ierr);
    if (PetscCheckPointer(ptr,PETSC_INT)) {ierr = PetscPrintf(PETSC_COMM_SELF,"Mistook NULL pointer for valid pointer\n");CHKERRQ(ierr);}
    if (PetscCheckPointer(ptr2,PETSC_INT)) {
      if (*(volatile PetscInt*)ptr2 == 0x12345678) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"Free'd pointer is still accessible\n");CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF,"Free'd pointer is still accessible, but contains corrupt data\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -check_pointer_intensity 1

TEST*/
