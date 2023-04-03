static const char help[] = "Benchmarking PetscSF Ping-pong latency (similar to osu_latency)\n\n";

/*
  This is a simple test to measure the latency of MPI communication.
  The test is run with two processes.  The first process sends a message
  to the second process, and after having received the message, the second
  process sends a message back to the first process once.  The is repeated
  a number of times.  The latency is defined as half time of the round-trip.

  It mimics osu_latency from the OSU microbenchmarks (https://mvapich.cse.ohio-state.edu/benchmarks/).

  Usage: mpirun -n 2 ./ex1k -mtype <type>
  Other arguments have a default value that is also used in osu_latency.

  Examples:

  On Summit at OLCF:
    jsrun --smpiargs "-gpu" -n 2 -a 1 -c 7 -g 1 -r 2 -l GPU-GPU -d packed -b packed:7 ./ex1k  -mtype kokkos

  On Crusher at OLCF:
    srun -n2 -c32 --cpu-bind=map_cpu:0,1 --gpus-per-node=8 --gpu-bind=map_gpu:0,1 ./ex1k -mtype kokkos
*/
#include <petscsf.h>
#include <Kokkos_Core.hpp>

/* Same values as OSU microbenchmarks */
#define LAT_LOOP_SMALL     10000
#define LAT_SKIP_SMALL     100
#define LAT_LOOP_LARGE     1000
#define LAT_SKIP_LARGE     10
#define LARGE_MESSAGE_SIZE 8192

int main(int argc, char **argv)
{
  PetscSF        sf[64];
  PetscLogDouble t_start = 0, t_end = 0, time[64];
  PetscInt       i, j, n, nroots, nleaves, niter = 100, nskip = 10;
  PetscInt       maxn = 512 * 1024; /* max 4M bytes messages */
  PetscSFNode   *iremote;
  PetscMPIInt    rank, size;
  PetscScalar   *rootdata = NULL, *leafdata = NULL, *pbuf, *ebuf;
  size_t         msgsize;
  PetscMemType   mtype       = PETSC_MEMTYPE_HOST;
  char           mstring[16] = {0};
  PetscBool      set;
  PetscInt       skipSmall = -1, loopSmall = -1;
  MPI_Op         op = MPI_REPLACE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscKokkosInitializeCheck());

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 processes");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxn", &maxn, NULL)); /* maxn PetscScalars */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-skipSmall", &skipSmall, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-loopSmall", &loopSmall, NULL));

  PetscCall(PetscMalloc1(maxn, &iremote));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mtype", mstring, 16, &set));
  if (set) {
    PetscBool isHost, isKokkos;
    PetscCall(PetscStrcasecmp(mstring, "host", &isHost));
    PetscCall(PetscStrcasecmp(mstring, "kokkos", &isKokkos));
    if (isHost) mtype = PETSC_MEMTYPE_HOST;
    else if (isKokkos) mtype = PETSC_MEMTYPE_KOKKOS;
    else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unknown memory type: %s", mstring);
  }

  if (mtype == PETSC_MEMTYPE_HOST) {
    PetscCall(PetscMalloc2(maxn, &rootdata, maxn, &leafdata));
  } else {
    PetscCallCXX(rootdata = (PetscScalar *)Kokkos::kokkos_malloc(sizeof(PetscScalar) * maxn));
    PetscCallCXX(leafdata = (PetscScalar *)Kokkos::kokkos_malloc(sizeof(PetscScalar) * maxn));
  }
  PetscCall(PetscMalloc2(maxn, &pbuf, maxn, &ebuf));
  for (i = 0; i < maxn; i++) {
    pbuf[i] = 123.0;
    ebuf[i] = 456.0;
  }

  for (n = 1, i = 0; n <= maxn; n *= 2, i++) {
    PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf[i]));
    PetscCall(PetscSFSetFromOptions(sf[i]));
    if (rank == 0) {
      nroots  = n;
      nleaves = 0;
    } else {
      nroots  = 0;
      nleaves = n;
      for (j = 0; j < nleaves; j++) {
        iremote[j].rank  = 0;
        iremote[j].index = j;
      }
    }
    PetscCall(PetscSFSetGraph(sf[i], nroots, nleaves, NULL, PETSC_COPY_VALUES, iremote, PETSC_COPY_VALUES));
    PetscCall(PetscSFSetUp(sf[i]));
  }

  if (loopSmall > 0) {
    nskip = skipSmall;
    niter = loopSmall;
  } else {
    nskip = LAT_SKIP_SMALL;
    niter = LAT_LOOP_SMALL;
  }

  for (n = 1, j = 0; n <= maxn; n *= 2, j++) {
    msgsize = sizeof(PetscScalar) * n;
    if (mtype == PETSC_MEMTYPE_HOST) {
      PetscCall(PetscArraycpy(rootdata, pbuf, n));
      PetscCall(PetscArraycpy(leafdata, ebuf, n));
    } else {
      Kokkos::View<PetscScalar *>                          dst1((PetscScalar *)rootdata, n);
      Kokkos::View<PetscScalar *>                          dst2((PetscScalar *)leafdata, n);
      Kokkos::View<const PetscScalar *, Kokkos::HostSpace> src1((const PetscScalar *)pbuf, n);
      Kokkos::View<const PetscScalar *, Kokkos::HostSpace> src2((const PetscScalar *)ebuf, n);
      PetscCallCXX(Kokkos::deep_copy(dst1, src1));
      PetscCallCXX(Kokkos::deep_copy(dst2, src2));
    }

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    }
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

    for (i = 0; i < niter + nskip; i++) {
      if (i == nskip) {
        PetscCallCXX(Kokkos::fence());
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
        t_start = MPI_Wtime();
      }
      PetscCall(PetscSFBcastWithMemTypeBegin(sf[j], MPIU_SCALAR, mtype, rootdata, mtype, leafdata, op));
      PetscCall(PetscSFBcastEnd(sf[j], MPIU_SCALAR, rootdata, leafdata, op));
      PetscCall(PetscSFReduceWithMemTypeBegin(sf[j], MPIU_SCALAR, mtype, leafdata, mtype, rootdata, op));
      PetscCall(PetscSFReduceEnd(sf[j], MPIU_SCALAR, leafdata, rootdata, op));
    }
    PetscCallCXX(Kokkos::fence());
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start) * 1e6 / (niter * 2);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t##  PetscSF Ping-pong test on %s ##\n  Message(Bytes) \t\tLatency(us)\n", mtype == PETSC_MEMTYPE_HOST ? "Host" : "Device"));
  for (n = 1, j = 0; n <= maxn; n *= 2, j++) {
    PetscCall(PetscSFDestroy(&sf[j]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%16" PetscInt_FMT " \t %16.4f\n", ((PetscInt)sizeof(PetscScalar)) * n, time[j]));
  }
  PetscCall(PetscFree2(pbuf, ebuf));
  if (mtype == PETSC_MEMTYPE_HOST) {
    PetscCall(PetscFree2(rootdata, leafdata));
  } else {
    PetscCallCXX(Kokkos::kokkos_free(rootdata));
    PetscCallCXX(Kokkos::kokkos_free(leafdata));
  }
  PetscCall(PetscFree(iremote));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    requires: kokkos
    # use small numbers to make the test cheap
    args: -maxn 4 -skipSmall 1 -loopSmall 1
    filter: grep "DOES_NOT_EXIST"
    output_file: output/empty.out
    nsize: 2

    test:
      args: -mtype {{host kokkos}}

    test:
      requires: mpix_stream
      args: -mtype kokkos -sf_use_stream_aware_mpi 1

TEST*/
