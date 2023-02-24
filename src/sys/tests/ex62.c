
static char help[] = "Tests `PetscGarbageKeySortedIntersect()`\n\n";

#include <petscsys.h>
#include <petsc/private/garbagecollector.h>

/* This program tests `PetscGarbageKeySortedIntersect(), which is the
   public (MPI) interface to
   `PetscErrorCode GarbageKeySortedIntersect_Private()`.
   Sets are sent packed in arrays, with the first entry as the number of
   set elements and the sets the remaining elements. This is because the
   MPI reduction operation must have the call signature:
   void PetscGarbageKeySortedIntersect(void *inset, void *inoutset, PetscMPIInt *length, MPI_Datatype *dtype)
   This is a thin wrapper for the private routine:
   PetscErrorCode GarbageKeySortedIntersect_Private(PetscInt64 seta[], PetscInt *lena, PetscInt64 setb[], PetscInt lenb)
   Where
   seta = (PetscInt64 *)inoutset;
   setb = (PetscInt64 *)inset;
   And the arguments are passed as:
   &seta[1], (PetscInt *)&seta[0], &setb[1], (PetscInt)setb[0]
*/

/* Populate a set with upto the first 49 unique Fibonnaci numbers */
PetscErrorCode Fibonnaci(PetscInt64 **set, PetscInt n)
{
  PetscInt   ii;
  PetscInt64 fib[] = {1,        2,        3,        5,        8,         13,        21,        34,        55,        89,         144,        233,        377,        610,        987,        1597,    2584,
                      4181,     6765,     10946,    17711,    28657,     46368,     75025,     121393,    196418,    317811,     514229,     832040,     1346269,    2178309,    3524578,    5702887, 9227465,
                      14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025};

  PetscFunctionBeginUser;
  PetscAssert((n < 50), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "n must be less than 50\n");
  PetscCall(PetscMalloc1(n + 1, set));
  (*set)[0] = (PetscInt64)n;
  for (ii = 0; ii < n; ii++) { (*set)[ii + 1] = fib[ii]; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Populate a set with Square numbers */
PetscErrorCode Square(PetscInt64 **set, PetscInt n)
{
  PetscInt64 ii;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(n + 1, set));
  (*set)[0] = (PetscInt64)n;
  for (ii = 1; ii < n + 1; ii++) { (*set)[ii] = ii * ii; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Populate a set with Cube numbers */
PetscErrorCode Cube(PetscInt64 **set, PetscInt n)
{
  PetscInt64 ii;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(n + 1, set));
  (*set)[0] = (PetscInt64)n;
  for (ii = 1; ii < n + 1; ii++) { (*set)[ii] = ii * ii * ii; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Populate a set with numbers to sixth power */
PetscErrorCode Sixth(PetscInt64 **set, PetscInt n)
{
  PetscInt64 ii;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(n + 1, set));
  (*set)[0] = (PetscInt64)n;
  for (ii = 1; ii < n + 1; ii++) { (*set)[ii] = ii * ii * ii * ii * ii * ii; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print out the contents of a set */
PetscErrorCode PrintSet(PetscInt64 *set)
{
  char     text[64];
  PetscInt ii;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "["));
  for (ii = 1; ii <= (PetscInt)set[0]; ii++) {
    PetscCall(PetscFormatConvert(" %" PetscInt64_FMT ",", text));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, text, set[ii]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check set equality */
PetscErrorCode AssertSetsEqual(PetscInt64 *set, PetscInt64 *true_set)
{
  PetscInt ii;

  PetscFunctionBeginUser;
  PetscAssert((set[0] == true_set[0]), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Sets of different sizes");
  for (ii = 1; ii < set[0] + 1; ii++) PetscAssert((set[ii] == true_set[ii]), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Sets are different");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when two enpty sets are passed */
PetscErrorCode test_empty_empty()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(1, &set_a));
  PetscCall(PetscMalloc1(1, &set_b));

  set_a[0] = 0;

  set_b[0] = 0;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta is empty */
PetscErrorCode test_a_empty()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(1, &set_a));
  PetscCall(PetscMalloc1(2, &set_b));

  set_a[0] = 0;

  set_b[0] = 1;
  set_b[1] = 1;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when setb is empty */
PetscErrorCode test_b_empty()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(2, &set_a));
  PetscCall(PetscMalloc1(1, &set_b));

  set_a[0] = 1;
  set_a[1] = 1;

  set_b[0] = 0;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when both sets are identical */
PetscErrorCode test_identical()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {3, 1, 4, 9};
  PetscMPIInt length  = 4;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(4, &set_a));
  PetscCall(PetscMalloc1(4, &set_b));

  set_a[0] = 3;
  set_a[1] = 1;
  set_a[2] = 4;
  set_a[3] = 9;

  set_b[0] = 3;
  set_b[1] = 1;
  set_b[2] = 4;
  set_b[3] = 9;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when sets have no elements in common */
PetscErrorCode test_disjoint()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(4, &set_a));
  PetscCall(PetscMalloc1(4, &set_b));

  set_a[0] = 3;
  set_a[1] = 1;
  set_a[2] = 4;
  set_a[3] = 9;

  set_b[0] = 3;
  set_b[1] = 2;
  set_b[2] = 6;
  set_b[3] = 8;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when sets only have one element in common */
PetscErrorCode test_single_common()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {1, 4};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(4, &set_a));
  PetscCall(PetscMalloc1(5, &set_b));

  set_a[0] = 3;
  set_a[1] = 1;
  set_a[2] = 4;
  set_a[3] = 9;

  set_b[0] = 3;
  set_b[1] = 2;
  set_b[2] = 4;
  set_b[3] = 6;
  set_b[4] = 8;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Specific test case flagged by PETSc issue #1247 */
PetscErrorCode test_issue_1247()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(3, &set_a));
  PetscCall(PetscMalloc1(2, &set_b));

  set_a[0] = 2;
  set_a[1] = 2;
  set_a[2] = 3;

  set_b[0] = 1;
  set_b[1] = 1;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta is empty and setb is large */
PetscErrorCode test_empty_big()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(1, &set_a));
  PetscCall(Square(&set_b, 999));

  set_a[0] = 0;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta is small and setb is large */
PetscErrorCode test_small_big()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {3, 1, 4, 9};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(5, &set_a));
  PetscCall(Square(&set_b, 999));

  set_a[0] = 4;
  set_a[1] = 1;
  set_a[2] = 4;
  set_a[3] = 8;
  set_a[4] = 9;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta is medium sized and setb is large */
PetscErrorCode test_moderate_big()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {2, 1, 144};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(Fibonnaci(&set_a, 49));
  PetscCall(Square(&set_b, 999));

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta and setb are large */
PetscErrorCode test_big_big()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64 *truth;
  PetscMPIInt length = 1;

  PetscFunctionBeginUser;

  PetscCall(Cube(&set_a, 999));
  PetscCall(Square(&set_b, 999));

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));

  PetscCall(Sixth(&truth, 9));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));
  PetscCall(PetscFree(truth));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when setb is empty and setb is large */
PetscErrorCode test_big_empty()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {0};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(Cube(&set_a, 999));
  PetscCall(PetscMalloc1(1, &set_b));

  set_b[0] = 0;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when setb is small and setb is large */
PetscErrorCode test_big_small()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {2, 1, 8};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(Cube(&set_a, 999));
  PetscCall(PetscMalloc1(5, &set_b));

  set_b[0] = 4;
  set_b[1] = 1;
  set_b[2] = 4;
  set_b[3] = 8;
  set_b[4] = 9;

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when setb is medium sized and setb is large */
PetscErrorCode test_big_moderate()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64  truth[] = {2, 1, 8};
  PetscMPIInt length  = 1;

  PetscFunctionBeginUser;

  PetscCall(Cube(&set_a, 999));
  PetscCall(Fibonnaci(&set_b, 49));

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tests functionality when seta and setb are large, in the opposite
 order to test_big_big() */
PetscErrorCode test_big_big_reversed()
{
  PetscInt64 *set_a, *set_b;
  PetscInt64 *truth;
  PetscMPIInt length = 1;

  PetscFunctionBeginUser;

  PetscCall(Cube(&set_a, 999));
  PetscCall(Square(&set_b, 999));

  PetscGarbageKeySortedIntersect((void *)set_b, (void *)set_a, &length, NULL);
  PetscCall(PrintSet(set_a));

  PetscCall(Sixth(&truth, 9));
  PetscCall(AssertSetsEqual(set_a, truth));

  PetscCall(PetscFree(set_a));
  PetscCall(PetscFree(set_b));
  PetscCall(PetscFree(truth));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Main executes the individual tests in a predefined order */
int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Small tests */
  /* Test different edge cases with small sets */
  PetscCall(test_empty_empty());
  PetscCall(test_a_empty());
  PetscCall(test_b_empty());
  PetscCall(test_identical());
  PetscCall(test_disjoint());
  PetscCall(test_single_common());
  PetscCall(test_issue_1247());

  /* Big tests */
  /* Test different edge cases with big sets */
  PetscCall(test_empty_big());
  PetscCall(test_small_big());
  PetscCall(test_moderate_big());
  PetscCall(test_big_big());
  PetscCall(test_big_empty());
  PetscCall(test_big_small());
  PetscCall(test_big_moderate());
  PetscCall(test_big_big_reversed());

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ALL PASSED\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 0

TEST*/
