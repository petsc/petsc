
static char help[] = "Run Birthday Spacing Tests for PetscRandom.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

/* L'Ecuyer & Simard, 2001.
 * "On the performance of birthday spacings tests with certain families of random number generators"
 * https://doi.org/10.1016/S0378-4754(00)00253-6
 */

static int size_t_compar(const void *a, const void *b)
{
  size_t A = *((const size_t *)a);
  size_t B = *((const size_t *)b);
  if (A < B)  return -1;
  if (A == B) return 0;
  return 1;
}

static PetscErrorCode PoissonTailProbability(PetscReal lambda, PetscInt Y, PetscReal *prob)
{
  PetscReal p = 1.;
  PetscReal logLambda;
  PetscInt  i;
  PetscReal logFact = 0.;

  PetscFunctionBegin;
  logLambda = PetscLogReal(lambda);
  for (i = 0; i < Y; i++) {
    PetscReal exponent = -lambda + i * logLambda - logFact;

    logFact += PetscLogReal(i+1);

    p -= PetscExpReal(exponent);
  }
  *prob = p;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscMPIInt    size;
  PetscInt       log2d, log2n, t, N;
  size_t         d, k, n, i, Y;
  PetscReal      lambda;
  size_t         *X;
  PetscRandom    random;
  PetscReal      p;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  t     = 8;
  log2d = 7;
  log2n = 20;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Birthday spacing test parameters", "PetscRandom");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-t",     "t, the dimension of the sample space", "ex3.c", t, &t, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-log2d", "The log of d, the number of bins per direction", "ex3.c", log2d, &log2d, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-log2n", "The log of n, the number of samples per process", "ex3.c", log2n, &log2n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if ((size_t)log2d * t > sizeof(size_t) * 8 - 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE, "The number of bins (2^%D) is too big for size_t.", log2d * t);
  d = ((size_t) 1) << log2d;
  k = ((size_t) 1) << (log2d * t);
  if ((size_t)log2n > sizeof(size_t) * 8 - 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE, "The number of samples per process (2^%D) is too big for size_t.", log2n);
  n = ((size_t) 1) << log2n;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  N    = size;
  lambda = PetscPowRealInt(2.,(3 * log2n - (2 + log2d * t)));

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Generating %zu samples (%g GB) per process in a %D dimensional space with %zu bins per dimension.\n", n, (n*1.e-9)*sizeof(size_t), t, d);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Expected spacing collisions per process %f (%f total).\n", (double) lambda, N * lambda);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&random);CHKERRQ(ierr);
  ierr = PetscRandomSetType(random,PETSCRANDER48);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(random);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(random,0.0,1.0);CHKERRQ(ierr);
  ierr = PetscRandomView(random,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&X);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    PetscInt j;
    size_t   bin  = 0;
    size_t   mult = 1;

    for (j = 0; j < t; j++, mult *= d) {
      PetscReal x;
      size_t slot;

      ierr = PetscRandomGetValueReal(random,&x);CHKERRQ(ierr);
      /* worried about precision here */
      slot = (size_t) (x * d);
      bin += mult * slot;
    }
    if (bin >= k) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Generated point in bin %zu, but only %zu bins\n",bin,k);
    X[i] = bin;
  }

  qsort(X, n, sizeof(size_t), size_t_compar);
  for (i = 0; i < n - 1; i++) {
    X[i] = X[i + 1] - X[i];
  }
  qsort(X, n - 1, sizeof(size_t), size_t_compar);
  for (i = 0, Y = 0; i < n - 2; i++) {
    Y += (X[i + 1] == X[i]);
  }

  ierr = MPI_Allreduce(MPI_IN_PLACE, &Y, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = PoissonTailProbability(N*lambda,Y,&p);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%zu total collisions counted: that many or more should occur with probabilty %g.\n",Y,(double)p);CHKERRQ(ierr);

  ierr = PetscFree(X);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&random);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: rander48_failure
  test:
    suffix: sprng_failure
    requires: sprng
    args: -random_type sprng
  test:
    suffix: random123
    nsize: 8
    requires: random123
    args: -random_type random123

TEST*/
