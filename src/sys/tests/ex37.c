
static char help[] = "Test PetscFormatConvertGetSize().\n";

#include <petscsys.h>
#include <petscviewer.h>

PetscErrorCode TestPetscVSNPrintf(char *, size_t, size_t *, const char *, ...);

int main(int argc, char **argv)
{
  size_t      sz, fullLength;
  char       *newformatstr, buffer[128], longstr[256], superlongstr[10000];
  const char *formatstr = "Greetings %D %3.2f %g\n";
  PetscInt    i, twentytwo = 22;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* test that PetscFormatConvertGetSize() correctly counts needed amount of space */
  PetscCall(PetscFormatConvertGetSize(formatstr, &sz));
  if (PetscDefined(USE_64BIT_INDICES)) {
    PetscCheck(sz == 29, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Format size %zu should be 29", sz);
  } else {
    PetscCheck(sz == 27, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Format size %zu should be 27", sz);
  }
  PetscCall(PetscMalloc1(sz, &newformatstr));
  PetscCall(PetscFormatConvert(formatstr, newformatstr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, newformatstr, twentytwo, 3.47, 3.0));
  PetscCall(PetscFree(newformatstr));

  /* Test correct count is returned with %g format */
  PetscCall(PetscSNPrintfCount(buffer, sizeof(buffer), "Test %g %g\n", &sz, 3.33, 2.7));
  PetscCall(PetscStrlen(buffer, &fullLength));
  PetscCheck(sz == fullLength + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscSNPrintfCount() count should be %d it is %d", (int)fullLength + 1, (int)sz);

  /* test that TestPetscVSNPrintf() fullLength argument returns required space for the string when buffer is long enough */
  PetscCall(TestPetscVSNPrintf(buffer, sizeof(buffer), &fullLength, "Greetings %s", "This is my string"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "buffer :%s: fullLength %d\n", buffer, (int)fullLength));

  /* test that TestPetscVSNPrintf() fullLength argument returns required space for the string when buffer is not long enough */
  for (i = 0; i < 255; i++) longstr[i] = 's';
  longstr[255] = 0;
  PetscCall(TestPetscVSNPrintf(buffer, sizeof(buffer), &fullLength, "Greetings %s", longstr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "longstr fullLength %d\n", (int)fullLength));

  /* test that PetscPrintf() works for strings longer than the default buffer size */
  for (i = 0; i < 9998; i++) superlongstr[i] = 's';
  superlongstr[9998] = 't';
  superlongstr[9999] = 0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Greetings %s", superlongstr));

  /* test that PetscSynchronizedPrintf() works for strings longer than the default buffer size */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Greetings %s", superlongstr));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));

  /* test that PetscSynchronizedFPrintf() works for strings longer than the default buffer size */
  PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD, stdout, "Greetings %s", superlongstr));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));

  /* test that PetscSynchronizedFPrintf() works for strings longer than the default buffer size */
  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "Greetings %s", superlongstr));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  /* add new line to end of file so that diff does not warn about it being missing */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode TestPetscVSNPrintf(char *str, size_t l_str, size_t *fullLength, const char *format, ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscVSNPrintf(str, l_str, format, fullLength, Argp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*TEST

   test:
     nsize: 2
     requires: defined(PETSC_HAVE_VA_COPY)

TEST*/
