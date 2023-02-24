
static char help[] = "Tests `GarbageKeyAllReduceIntersect_Private()` in parallel\n\n";

#include <petscsys.h>
#include <petsc/private/garbagecollector.h>

/* This program tests `GarbageKeyAllReduceIntersect_Private()`.
   To test this routine in parallel, the sieve of Eratosthenes is
   implemented.
*/

/* Populate an array with Prime numbers <= n.
   Primes are generated using trial division
*/
PetscErrorCode Prime(PetscInt64 **set, PetscInt n)
{
  size_t      overestimate;
  PetscBool   is_prime;
  PetscInt64  ii, jj, count = 0;
  PetscInt64 *prime;

  PetscFunctionBeginUser;
  /* There will be fewer than ceil(1.26 * n/log(n)) primes <= n */
  overestimate = (size_t)PetscCeilReal(((PetscReal)n) * 1.26 / PetscLogReal((PetscReal)n));
  PetscCall(PetscMalloc1(overestimate, &prime));
  for (ii = 2; ii < n + 1; ii++) {
    is_prime = PETSC_TRUE;
    for (jj = 2; jj <= PetscFloorReal(PetscSqrtReal(ii)); jj++) {
      if (ii % jj == 0) {
        is_prime = PETSC_FALSE;
        break;
      }
    }
    if (is_prime) {
      prime[count] = ii;
      count++;
    }
  }

  PetscCall(PetscMalloc1((size_t)count + 1, set));
  (*set)[0] = count;
  for (ii = 1; ii < count + 1; ii++) { (*set)[ii] = prime[ii - 1]; }
  PetscCall(PetscFree(prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print out the contents of a set */
PetscErrorCode PrintSet(MPI_Comm comm, PetscInt64 *set)
{
  char     text[64];
  PetscInt ii;

  PetscFunctionBeginUser;
  PetscCall(PetscSynchronizedPrintf(comm, "["));
  for (ii = 1; ii <= (PetscInt)set[0]; ii++) {
    PetscCall(PetscFormatConvert(" %" PetscInt64_FMT ",", text));
    PetscCall(PetscSynchronizedPrintf(comm, text, set[ii]));
  }
  PetscCall(PetscSynchronizedPrintf(comm, "]\n"));
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
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

/* Parallel implementation of the sieve of Eratosthenes */
PetscErrorCode test_sieve(MPI_Comm comm)
{
  PetscInt64  ii, local_p, maximum, n;
  PetscInt64 *local_set, *cursor, *bootstrap_primes, *truth;
  PetscMPIInt size, rank;
  PetscReal   x;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* There should be at least `size + 1` primes smaller than
     (size + 1)*(log(size + 1) + log(log(size + 1))
    once `size` >=6
    This is sufficient for each rank to create its own sieve based off
    a different prime and calculate the size of the sieve.
  */
  x = (PetscReal)(size > 6) ? size + 1 : 7;
  x = x * (PetscLogReal(x) + PetscLogReal(PetscLogReal(x)));
  PetscCall(Prime(&bootstrap_primes, PetscCeilReal(x)));

  /* Calculate the maximum possible prime, select a prime number for
     each rank and allocate memorty for the sieve
  */
  maximum = bootstrap_primes[size + 1] * bootstrap_primes[size + 1] - 1;
  local_p = bootstrap_primes[rank + 1];
  n       = maximum - local_p - (maximum / local_p) + 1 + rank + 1;
  PetscCall(PetscMalloc1(n + 1, &local_set));

  /* Populate the sieve first with all primes <= `local_p`, followed by
     all integers that are not a multiple of `local_p`
  */
  local_set[0] = n;
  cursor       = &local_set[1];
  for (ii = 0; ii < rank + 1; ii++) {
    *cursor = bootstrap_primes[ii + 1];
    cursor++;
  }
  for (ii = local_p + 1; ii <= maximum; ii++) {
    if (ii % local_p != 0) {
      *cursor = ii;
      cursor++;
    }
  }
  PetscCall(PetscPrintf(comm, "SIEVES:\n"));
  PetscCall(PrintSet(comm, local_set));

  PetscCall(PetscFree(bootstrap_primes));

  /* Perform the intersection, testing parallel intersection routine */
  PetscCall(GarbageKeyAllReduceIntersect_Private(PETSC_COMM_WORLD, &local_set[1], (PetscInt *)&local_set[0]));

  PetscCall(PetscPrintf(comm, "INTERSECTION:\n"));
  PetscCall(PrintSet(comm, local_set));

  PetscCall(Prime(&truth, maximum));
  PetscCall(PetscPrintf(comm, "TRUTH:\n"));
  PetscCall(PrintSet(comm, truth));

  /* Assert the intersection corresponds to primes calculated using
     trial division
  */
  PetscCall(AssertSetsEqual(local_set, truth));

  PetscCall(PetscFree(local_set));
  PetscCall(PetscFree(truth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Main executes the individual tests in a predefined order */
int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(test_sieve(PETSC_COMM_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ALL PASSED\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     test:
       nsize: 2
       suffix: 2
       output_file: output/ex63_2.out
     test:
       nsize: 3
       suffix: 3
       output_file: output/ex63_3.out
     test:
       nsize: 4
       suffix: 4
       output_file: output/ex63_4.out

TEST*/
