
static char help[] = "Test PetscFormatConvertGetSize().\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  size_t         sz;
  char           *newformatstr;
  const char     *formatstr = "Greetings %D %3.2f %g\n";

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscFormatConvertGetSize(formatstr,&sz);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INDICES)
  if (sz != 27) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Format size %d should be 27\n",sz);
#else
  if (sz != 29) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Format size %d should be 29\n",sz);
#endif
  ierr = PetscMalloc1(sz,&newformatstr);CHKERRQ(ierr);
  ierr = PetscFormatConvert(formatstr,newformatstr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,newformatstr,22,3.47,3.0);CHKERRQ(ierr);
  ierr = PetscFree(newformatstr);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
