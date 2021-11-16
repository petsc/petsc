
static char help[] = "Test PetscFormatConvertGetSize().\n";

#include <petscsys.h>
#include <petscviewer.h>

PetscErrorCode TestPetscVSNPrintf(char*,size_t,size_t*,const char*,...);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  size_t         sz,fullLength;
  char           *newformatstr,buffer[128],longstr[256],superlongstr[10000];
  const char     *formatstr = "Greetings %D %3.2f %g\n";
  PetscInt       i,twentytwo = 22;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* test that PetscFormatConvertGetSize() correctly counts needed amount of space */
  ierr = PetscFormatConvertGetSize(formatstr,&sz);CHKERRQ(ierr);
  if (PetscDefined(USE_64BIT_INDICES)) {
    if (sz != 29) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Format size %zu should be 29",sz);
  } else {
    if (sz != 27) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Format size %zu should be 27",sz);
  }
  ierr = PetscMalloc1(sz,&newformatstr);CHKERRQ(ierr);
  ierr = PetscFormatConvert(formatstr,newformatstr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,newformatstr,twentytwo,3.47,3.0);CHKERRQ(ierr);
  ierr = PetscFree(newformatstr);CHKERRQ(ierr);

  /* Test correct count is returned with %g format */
  ierr = PetscSNPrintfCount(buffer,sizeof(buffer),"Test %g %g\n",&sz,3.33,2.7);CHKERRQ(ierr);
  ierr = PetscStrlen(buffer,&fullLength);CHKERRQ(ierr);
  if (sz != fullLength+1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSNPrintfCount() count should be %d it is %d\n",(int)fullLength+1,(int)sz);

  /* test that TestPetscVSNPrintf() fullLength argument returns required space for the string when buffer is long enough */
  ierr = TestPetscVSNPrintf(buffer,sizeof(buffer),&fullLength,"Greetings %s","This is my string");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"buffer :%s: fullLength %d\n",buffer,(int)fullLength);CHKERRQ(ierr);

  /* test that TestPetscVSNPrintf() fullLength argument returns required space for the string when buffer is not long enough */
  for (i=0; i<255; i++) {longstr[i] = 's';} longstr[255] = 0;
  ierr = TestPetscVSNPrintf(buffer,sizeof(buffer),&fullLength,"Greetings %s",longstr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"longstr fullLength %d\n",(int)fullLength);CHKERRQ(ierr);

  /* test that PetscPrintf() works for strings longer than the default buffer size */
  for (i=0; i<9998; i++) {superlongstr[i] = 's';} superlongstr[9998] = 't'; superlongstr[9999] = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Greetings %s",superlongstr);CHKERRQ(ierr);

  /* test that PetscSynchronizedPrintf() works for strings longer than the default buffer size */
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings %s",superlongstr);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);

  /* test that PetscSynchronizedFPrintf() works for strings longer than the default buffer size */
  ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"Greetings %s",superlongstr);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout);CHKERRQ(ierr);

  /* test that PetscSynchronizedFPrintf() works for strings longer than the default buffer size */
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Greetings %s",superlongstr);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* add new line to end of file so that diff does not warn about it being missing */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode TestPetscVSNPrintf(char *str,size_t l_str,size_t *fullLength,const char* format,...)
{
  va_list        Argp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  va_start(Argp,format);
  ierr = PetscVSNPrintf(str,l_str,format,fullLength,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*TEST

   test:
     nsize: 2
     requires: defined(PETSC_HAVE_VA_COPY)

TEST*/
