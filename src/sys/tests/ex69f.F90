program ex69F90

!   Demonstrates two issues
!
!   A) How using mpiexec to start up a program can dramatically change
!      the OpenMP thread binding/mapping resulting in poor performance
!
!      Set the environmental variable with, for example,
!        export OMP_NUM_THREADS=4
!      Run this example on one MPI process three ways
!        ./ex69f
!        mpiexec -n 1 ./ex69f
!        mpiexec --bind-to numa -n 1 ./ex69f
!
!      You may get very different wall clock times
!      It seems some mpiexec implementations change the thread binding/mapping that results with
!      OpenMP so all the threads are run on a single core
!
!      The same differences occur without the PetscInitialize() call indicating
!      the binding change is done by the mpiexec, not the MPI_Init()
!
!   B) How cpu_time() may give unexpected results, much larger than expected,
!      even for code portions with no OpenMP
!
!      Note the CPU time for output of the second loop, it should equal the wallclock time
!      since the loop is not run in parallel (with OpenMP) but instead it may be listed as
!      many times higher
!
!     $ OMP_NUM_THREADS=8 ./ex69f (ifort compiler)
!       CPU time reported by cpu_time()              1.66649300000000
!       Wall clock time reported by system_clock()   0.273980000000000
!       Wall clock time reported by omp_get_wtime()  0.273979902267456
!
#include <petsc/finclude/petscsys.h>
  use petsc
  implicit none

  PetscErrorCode ierr
  double precision cputime_start, cputime_end, wtime_start, wtime_end, omp_get_wtime
  integer(kind=8) systime_start, systime_end, systime_rate
  double precision x(100)
  integer i, maxthreads, omp_get_max_threads

  PetscCallA(PetscInitialize(ierr))
  call system_clock(systime_start, systime_rate)
  wtime_start = omp_get_wtime()
  call cpu_time(cputime_start)
!$OMP PARALLEL DO
  do i = 1, 100
    x(i) = exp(3.0d0*i)
  end do
  call cpu_time(cputime_end)
  call system_clock(systime_end, systime_rate)
  wtime_end = omp_get_wtime()
  print *, 'CPU time reported by cpu_time()            ', cputime_end - cputime_start
  print *, 'Wall clock time reported by system_clock() ', real(systime_end - systime_start, kind=8)/real(systime_rate, kind=8)
  print *, 'Wall clock time reported by omp_get_wtime()', wtime_end - wtime_start
  print *, 'Value of x(22)', x(22)
!$ maxthreads = omp_get_max_threads()
  print *, 'Number of threads set', maxthreads

  call system_clock(systime_start, systime_rate)
  wtime_start = omp_get_wtime()
  call cpu_time(cputime_start)
  do i = 1, 100
    x(i) = exp(3.0d0*i)
  end do
  call cpu_time(cputime_end)
  call system_clock(systime_end, systime_rate)
  wtime_end = omp_get_wtime()
  print *, 'CPU time reported by cpu_time()            ', cputime_end - cputime_start
  print *, 'Wall clock time reported by system_clock() ', real(systime_end - systime_start, kind=8)/real(systime_rate, kind=8)
  print *, 'Wall clock time reported by omp_get_wtime()', wtime_end - wtime_start
  print *, 'Value of x(22)', x(22)
  PetscCallA(PetscFinalize(ierr))
end program ex69F90

!/*TEST
!
!   build:
!     requires: openmp
!
!   test:
!     filter: grep -v "Number of threads"
!
!TEST*/
