static char help[] = "Tests bounded stack storage and unwinding.\n\n";

// PetscStackCopy() is declared in this private header.
#include <petsc/private/petscimpl.h>
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  #include <omp.h>
#endif

static PetscErrorCode CheckStack(int depth, int remaining, PetscBool check)
{
  PetscStack copy = {{NULL}, {NULL}, {0}, {0}, 0, 0, PETSC_BOOL3_FALSE};

  PetscFunctionBeginUser;
  PetscCheck(petscstack.currentsize == depth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected stack depth %d, got %d", depth, petscstack.currentsize);
  PetscCheck(petscstack.check == PetscBoolToBool3(check), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack checking policy");
  PetscStackUpdateLine;
  PetscCheck(depth > PETSCSTACKSIZE || petscstack.line[depth - 1] == __LINE__ - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack line number");
  PetscCall(PetscStackCopy(&petscstack, &copy));
  PetscCheck(copy.currentsize == depth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack copy depth");
  for (int i = 0; i < PetscMin(depth, PETSCSTACKSIZE); ++i) {
    PetscCheck(copy.function[i] == petscstack.function[i] && copy.file[i] == petscstack.file[i] && copy.line[i] == petscstack.line[i] && copy.petscroutine[i] == petscstack.petscroutine[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect copied frame %d", i);
  }
  if (remaining) PetscCall(CheckStack(depth + 1, remaining - 1, check));
  else {
    PetscStack *stack;
    FILE       *file;
    char        line[1024];
    char       *ptr;
    int         frames = 0;
    PetscBool   mdebug, truncated = PETSC_FALSE;

    file = tmpfile();
    PetscCheck(file, PETSC_COMM_SELF, PETSC_ERR_SYS, "Cannot create temporary stack output file");
    PetscCall(PetscStackView(file));
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
      if (line[0] == '[') ++frames;
      if (strstr(line, "Only the first")) truncated = PETSC_TRUE;
    }
    PetscCheck(frames == PetscMin(depth, PETSCSTACKSIZE), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect number of viewed frames: %d", frames);
    PetscCheck(truncated == (depth > PETSCSTACKSIZE ? PETSC_TRUE : PETSC_FALSE), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect stack truncation diagnostic");
    PetscCall(PetscMallocGetDebug(&mdebug, NULL, NULL));
    PetscCall(PetscMalloc1(8, &ptr));
    PetscCall(PetscMallocGetStack(ptr, &stack));
    PetscCheck(mdebug ? stack && stack->currentsize == depth + 1 : !stack, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect allocation stack");
    PetscCall(PetscRealloc(16, &ptr));
    PetscCall(PetscMallocGetStack(ptr, &stack));
    PetscCheck(mdebug ? stack && stack->currentsize == depth : !stack, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect reallocation stack");
    PetscCall(PetscMallocDump(file));
    PetscCall(PetscFree(ptr));
    PetscCallExternal(fclose, file);
  }
  PetscStackPushExternal("external");
  PetscCheck(petscstack.currentsize == depth + 1 && petscstack.hotdepth == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect external stack frame");
  PetscStackPop;
  PetscCheck(petscstack.currentsize == depth && !petscstack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack was not restored");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscBool check    = PETSC_TRUE;
  const int depths[] = {2, PETSCSTACKSIZE - 1, PETSCSTACKSIZE, PETSCSTACKSIZE + 1, 2 * PETSCSTACKSIZE};
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  PetscStack *stacks[2];
#endif

  PetscFunctionBeginUser;
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  PetscCall(PetscSetMPIThreadRequiredType(MPI_THREAD_MULTIPLE));
#endif
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-checkstack", &check, NULL));
  /* Use private push/pop macros to test an empty local stack. The public macros operate on petscstack
     and cannot target this local stack. */
  for (int i = 0; i < 2; ++i) {
    PetscStack stack = {{NULL}, {NULL}, {0}, {0}, 0, 0, PETSC_BOOL3_FALSE};

    stack.check = i ? PETSC_BOOL3_UNKNOWN : PETSC_BOOL3_FALSE;
    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscCheck(!stack.currentsize && !stack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Empty stack was not preserved");
    PetscStackPush_Private(stack, __FILE__, PETSC_FUNCTION_NAME, __LINE__, 1, PETSC_TRUE);
    PetscCheck(stack.currentsize == 1 && stack.hotdepth == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack could not be reused after an empty pop");
    PetscStackPop_Private(stack, PETSC_FUNCTION_NAME);
    PetscCheck(!stack.currentsize && !stack.hotdepth, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack was not restored");
  }
  for (int i = 0; i < (int)PETSC_STATIC_ARRAY_LENGTH(depths); ++i) PetscCall(CheckStack(2, depths[i] - 2, check));
  petscstack.check = PetscBoolToBool3(PetscNot(check));
  PetscCall(CheckStack(2, 1, PetscNot(check)));
  petscstack.check = PetscBoolToBool3(check);
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  omp_set_dynamic(0);
  PetscPragmaOMP(parallel num_threads(2))
  {
    const int       thread = omp_get_thread_num(), depth = petscstack.currentsize;
    const PetscBool localcheck = thread ? PetscNot(check) : check;

    stacks[thread] = &petscstack;
    PetscCheckAbort(!thread || petscstack.check == PETSC_BOOL3_UNKNOWN, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Worker stack checking was already initialized");
    PetscPragmaOMP(barrier)
    for (int i = 0; i < (int)PETSC_STATIC_ARRAY_LENGTH(depths); ++i) PetscCallAbort(PETSC_COMM_SELF, CheckStack(depth + 1, depths[i] - 2, check));
    petscstack.check = PetscBoolToBool3(localcheck);
    PetscPragmaOMP(barrier)
    PetscCallAbort(PETSC_COMM_SELF, CheckStack(depth + 1, 1, localcheck));
    PetscCheckAbort(petscstackcheck == check, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Thread changed the process-wide default");
  }
  PetscCheck(stacks[0] != stacks[1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Threads share a stack");
#endif
  PetscCheck(petscstack.currentsize == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Main stack was not restored");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: debug

  test:
    suffix: serial
    requires: !threadsafety
    args: -checkstack {{0 1}}
    output_file: output/empty.out

  test:
    suffix: serial_no_malloc_debug
    requires: !threadsafety
    args: -malloc_debug 0
    output_file: output/empty.out

  test:
    suffix: threads
    requires: threadsafety openmp
    args: -checkstack {{0 1}}
    output_file: output/empty.out

TEST*/
