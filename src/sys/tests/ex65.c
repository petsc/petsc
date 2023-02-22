static const char help[] = "Test PetscFunctionList.\n";

#include <petscsys.h>
#include <petscviewer.h>

#define PETSC_DEFINE_FUNCTION_AND_STR(name) \
  static void name() \
  { \
    puts("called " PetscStringize(name) "()"); \
  } \
  static const char name##_str[] = PetscStringize(name)

PETSC_DEFINE_FUNCTION_AND_STR(foo);
PETSC_DEFINE_FUNCTION_AND_STR(bar);
PETSC_DEFINE_FUNCTION_AND_STR(baz);
PETSC_DEFINE_FUNCTION_AND_STR(bop);
PETSC_DEFINE_FUNCTION_AND_STR(qux);
PETSC_DEFINE_FUNCTION_AND_STR(quux);
PETSC_DEFINE_FUNCTION_AND_STR(blip);
PETSC_DEFINE_FUNCTION_AND_STR(blap);
PETSC_DEFINE_FUNCTION_AND_STR(blop);
PETSC_DEFINE_FUNCTION_AND_STR(graulty);
PETSC_DEFINE_FUNCTION_AND_STR(quix);

static const char *const all_names[]   = {foo_str, bar_str, baz_str, bop_str, qux_str, quux_str, blip_str, blap_str, blop_str, graulty_str, quix_str};
static void (*const all_funcs[])(void) = {foo, bar, baz, bop, qux, quux, blip, blap, blop, graulty, quix};
static const size_t num_names          = PETSC_STATIC_ARRAY_LENGTH(all_names);
static const size_t num_funcs          = PETSC_STATIC_ARRAY_LENGTH(all_funcs);

static PetscErrorCode TestPetscFunctionListCreate(PetscViewer viewer, PetscFunctionList *fl, PetscFunctionList *fl_dup)
{
  PetscFunctionBegin;
  // add the function
  PetscCall(PetscFunctionListAdd(fl, foo_str, foo));
  PetscCall(PetscFunctionListView(*fl, NULL));
  // remove it
  PetscCall(PetscFunctionListAdd(fl, foo_str, NULL));
  PetscCheck(*fl, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Emptying PetscFunctionList has destroyed it!");
  PetscCall(PetscFunctionListView(*fl, NULL));
  // should not do anything
  PetscCall(PetscFunctionListClear(*fl));
  PetscCall(PetscFunctionListView(*fl, viewer));
  PetscCall(PetscFunctionListDuplicate(*fl, fl_dup));
  PetscCheck(*fl_dup, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Duplicating empty function list has not allocated a new one");
  // still empty
  PetscCall(PetscFunctionListView(*fl, viewer));
  // also empty
  PetscCall(PetscFunctionListView(*fl_dup, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestPetscFunctionListFind(PetscViewer viewer, PetscFunctionList fl, PetscFunctionList fl_dup, size_t *fl_size, size_t *fl_dup_size)
{
  PetscFunctionBegin;
  // add a bunch of functions, and ensure they are all there
  for (size_t i = 0; i < num_funcs; ++i) {
    PetscVoidFunction func;

    PetscCall(PetscFunctionListAdd(&fl, all_names[i], all_funcs[i]));
    PetscCall(PetscFunctionListFind(fl, all_names[i], &func));
    PetscCheck(func == all_funcs[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListFind() failed to find %s() immediately after inserting it! returned %p != expected %p", all_names[i], (void *)(PETSC_UINTPTR_T)func, (void *)(PETSC_UINTPTR_T)(all_funcs[i]));
    // make sure the pointer is good
    func();
  }

  // ensure that none of them are missing
  for (size_t i = 0; i < num_funcs; ++i) {
    PetscVoidFunction func;

    PetscCall(PetscFunctionListFind(fl, all_names[i], &func));
    PetscCheck(func == all_funcs[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListFind() failed to find %s() after inserting all functions! returned %p != expected %p", all_names[i], (void *)(PETSC_UINTPTR_T)func, (void *)(PETSC_UINTPTR_T)(all_funcs[i]));
    // make sure the pointer is good
    func();
  }

  // appends to fl_dup
  PetscCall(PetscFunctionListDuplicate(fl, &fl_dup));

  // ensure that none of them are missing
  for (size_t i = 0; i < num_funcs; ++i) {
    PetscVoidFunction fl_func, fl_dup_func;

    PetscCall(PetscFunctionListFind(fl, all_names[i], &fl_func));
    PetscCheck(fl_func == all_funcs[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListFind() failed to find %s() after inserting all functions! returned %p != expected %p", all_names[i], (void *)(PETSC_UINTPTR_T)fl_func, (void *)(PETSC_UINTPTR_T)(all_funcs[i]));
    // make sure the pointer is good
    fl_func();
    PetscCall(PetscFunctionListFind(fl_dup, all_names[i], &fl_dup_func));
    PetscCheck(fl_dup_func == fl_func, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListFind() returned different results for %s() for duplicated function list. returned %p != expected %p", all_names[i], (void *)(PETSC_UINTPTR_T)fl_dup_func, (void *)(PETSC_UINTPTR_T)fl_func);
    fl_dup_func();
  }

  // same as in fl
  PetscCall(PetscFunctionListView(fl_dup, viewer));
  // clearing fl should have no effect on fl_dup
  PetscCall(PetscFunctionListClear(fl));
  // ensure that none of them are missing
  for (size_t i = 0; i < num_funcs; ++i) {
    PetscVoidFunction fl_dup_func;

    PetscCall(PetscFunctionListFind(fl_dup, all_names[i], &fl_dup_func));
    PetscCheck(fl_dup_func == all_funcs[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListFind() failed to find %s() in duplicated function list after clearing original list! returned %p != expected %p", all_names[i], (void *)(PETSC_UINTPTR_T)fl_dup_func, (void *)(PETSC_UINTPTR_T)(all_funcs[i]));
    fl_dup_func();
  }
  PetscCall(PetscFunctionListView(fl_dup, viewer));
  *fl_size     = 0;
  *fl_dup_size = num_funcs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestPetscFunctionListGet(PetscViewer viewer, PetscFunctionList fl, PetscFunctionList fl_dup, size_t expected_fl_size, size_t expected_fl_dup_size)
{
  const char **array;
  int          n;
#define PetscCheckArrayPointer(expected_non_null, array) \
  PetscCheck((expected_non_null) ? (array) != NULL : (array) == NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListGet() returned invalid array (%p) for cleared function list, expected %s", (void *)(array), (expected_non_null) ? "non-null" : "null")

  PetscFunctionBegin;
  PetscCall(PetscFunctionListGet(fl, &array, &n));
  PetscCheck((size_t)n == expected_fl_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListGet() returned unexpected size %d, expected %zu", n, expected_fl_size);
  PetscCheckArrayPointer(expected_fl_size, array);

  PetscCall(PetscFunctionListGet(fl_dup, &array, &n));
  PetscCheck((size_t)n == expected_fl_dup_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscFunctionListGet() returned unexpected size %d, expected %zu", n, expected_fl_dup_size);
  PetscCheckArrayPointer(expected_fl_dup_size, array);
  for (int i = 0; i < n; ++i) PetscCall(PetscViewerASCIIPrintf(viewer, "%d: %s\n", i + 1, array[i]));
  PetscCall(PetscFree(array));
  // ensure that free-ing the array is OK
  PetscCall(PetscFunctionListView(fl_dup, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
#undef PetscCheckArrayPointer
}

int main(int argc, char *argv[])
{
  PetscViewer       viewer;
  PetscFunctionList fl = NULL, fl_dup = NULL;
  size_t            fl_size, fl_dup_size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCheck(num_names == num_funcs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of functions %zu != number of function names %zu", num_funcs, num_names);
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer));

  PetscCall(TestPetscFunctionListCreate(viewer, &fl, &fl_dup));
  PetscCall(TestPetscFunctionListFind(viewer, fl, fl_dup, &fl_size, &fl_dup_size));
  PetscCall(TestPetscFunctionListGet(viewer, fl, fl_dup, fl_size, fl_dup_size));
  PetscCall(PetscFunctionListPrintTypes(PETSC_COMM_WORLD, PETSC_STDOUT, "my_prefix_", "-petsc_function_type", "Description", "PetscFunctionList", fl_dup, "foo", "bar"));

  PetscCall(PetscFunctionListDestroy(&fl));
  PetscCheck(fl == NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to destroy PetscFunctionList, pointer (%p) is non-null", (void *)fl);
  PetscCall(PetscFunctionListDestroy(&fl_dup));
  PetscCheck(fl_dup == NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to destroy PetscFunctionList, pointer (%p) is non-null", (void *)fl_dup);
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:

TEST*/
