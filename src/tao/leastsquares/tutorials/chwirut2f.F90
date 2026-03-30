!  Program usage: mpiexec -n 1 chwirut1f [-help] [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve a
!  nonlinear least-squares problem on a single processor.  We minimize the
!  Chwirut function:
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2)
!
!  The C version of this code is chwirut1.c
!
#include <petsc/finclude/petsctao.h>
module chwirut2fmodule
  use petsctao
  implicit none
  PetscReal t(0:213)
  PetscReal y(0:213)
  PetscInt, parameter :: m = 214, n = 3
  PetscMPIInt, parameter :: nn = n
  PetscMPIInt rank
  PetscMPIInt size
  PetscMPIInt, parameter :: idle_tag = 2000, die_tag = 3000
  PetscMPIInt, parameter :: zero = 0, one = 1

contains
  subroutine InitializeData()

    PetscInt i
    i = 0
    y(i) = 92.9000_PETSC_REAL_KIND; t(i) = 0.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 78.7000_PETSC_REAL_KIND; t(i) = 0.6250_PETSC_REAL_KIND; i = i + 1
    y(i) = 64.2000_PETSC_REAL_KIND; t(i) = 0.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 64.9000_PETSC_REAL_KIND; t(i) = 0.8750_PETSC_REAL_KIND; i = i + 1
    y(i) = 57.1000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 43.3000_PETSC_REAL_KIND; t(i) = 1.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 31.1000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 23.6000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 31.0500_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 23.7750_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 17.7375_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.8000_PETSC_REAL_KIND; t(i) = 3.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 11.5875_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 9.4125_PETSC_REAL_KIND; t(i) = 4.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.7250_PETSC_REAL_KIND; t(i) = 4.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.3500_PETSC_REAL_KIND; t(i) = 5.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.0250_PETSC_REAL_KIND; t(i) = 5.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 90.6000_PETSC_REAL_KIND; t(i) = 0.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 76.9000_PETSC_REAL_KIND; t(i) = 0.6250_PETSC_REAL_KIND; i = i + 1
    y(i) = 71.6000_PETSC_REAL_KIND; t(i) = 0.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 63.6000_PETSC_REAL_KIND; t(i) = 0.8750_PETSC_REAL_KIND; i = i + 1
    y(i) = 54.0000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 39.2000_PETSC_REAL_KIND; t(i) = 1.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.3000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.4000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.1750_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 22.1250_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 17.5125_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 14.2500_PETSC_REAL_KIND; t(i) = 3.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 9.4500_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 9.1500_PETSC_REAL_KIND; t(i) = 4.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.9125_PETSC_REAL_KIND; t(i) = 4.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.4750_PETSC_REAL_KIND; t(i) = 5.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 6.1125_PETSC_REAL_KIND; t(i) = 5.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 80.0000_PETSC_REAL_KIND; t(i) = 0.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 79.0000_PETSC_REAL_KIND; t(i) = 0.6250_PETSC_REAL_KIND; i = i + 1
    y(i) = 63.8000_PETSC_REAL_KIND; t(i) = 0.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 57.2000_PETSC_REAL_KIND; t(i) = 0.8750_PETSC_REAL_KIND; i = i + 1
    y(i) = 53.2000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 42.5000_PETSC_REAL_KIND; t(i) = 1.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 26.8000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 20.4000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 26.8500_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.0000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.4625_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.5250_PETSC_REAL_KIND; t(i) = 3.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.5375_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.5875_PETSC_REAL_KIND; t(i) = 4.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.1250_PETSC_REAL_KIND; t(i) = 4.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 6.1125_PETSC_REAL_KIND; t(i) = 5.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.9625_PETSC_REAL_KIND; t(i) = 5.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 74.1000_PETSC_REAL_KIND; t(i) = 0.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 67.3000_PETSC_REAL_KIND; t(i) = 0.6250_PETSC_REAL_KIND; i = i + 1
    y(i) = 60.8000_PETSC_REAL_KIND; t(i) = 0.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 55.5000_PETSC_REAL_KIND; t(i) = 0.8750_PETSC_REAL_KIND; i = i + 1
    y(i) = 50.3000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 41.0000_PETSC_REAL_KIND; t(i) = 1.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.4000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 20.4000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.3625_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.1500_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.7625_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.2000_PETSC_REAL_KIND; t(i) = 3.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.8750_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.1750_PETSC_REAL_KIND; t(i) = 4.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.3500_PETSC_REAL_KIND; t(i) = 4.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.9625_PETSC_REAL_KIND; t(i) = 5.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.6250_PETSC_REAL_KIND; t(i) = 5.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 81.5000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 62.4000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 32.5000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.4100_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.1200_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 15.5600_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.6300_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 78.0000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 59.9000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 33.2000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.8400_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.7500_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 14.6200_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 3.9400_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 76.8000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 61.0000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 32.9000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.8700_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 11.8100_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.3100_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.4400_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 78.0000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 63.5000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 33.8000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.5600_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.6300_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.7500_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.1200_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.4400_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 76.8000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 60.0000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 47.8000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 32.0000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 22.2000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 22.5700_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 18.8200_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.9500_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 11.2500_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 9.0000_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 6.6700_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 75.8000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 62.0000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 48.8000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 35.2000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 20.0000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 20.3200_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 19.3100_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.7500_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.4200_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.3100_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.4200_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 70.5000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 59.5000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 48.5000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 35.8000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.0000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.6700_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.0000_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 15.6400_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.1700_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.5500_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.1200_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 78.0000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 66.0000_PETSC_REAL_KIND; t(i) = .6250_PETSC_REAL_KIND; i = i + 1
    y(i) = 62.0000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 58.0000_PETSC_REAL_KIND; t(i) = .8750_PETSC_REAL_KIND; i = i + 1
    y(i) = 47.7000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 37.8000_PETSC_REAL_KIND; t(i) = 1.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 20.2000_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 21.0700_PETSC_REAL_KIND; t(i) = 2.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.8700_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 9.6700_PETSC_REAL_KIND; t(i) = 3.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.7600_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.4400_PETSC_REAL_KIND; t(i) = 4.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 4.8700_PETSC_REAL_KIND; t(i) = 4.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 4.0100_PETSC_REAL_KIND; t(i) = 5.2500_PETSC_REAL_KIND; i = i + 1
    y(i) = 3.7500_PETSC_REAL_KIND; t(i) = 5.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 24.1900_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 25.7600_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 18.0700_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 11.8100_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.0700_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.1200_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 70.8000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 54.7000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 48.0000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 39.8000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.8000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 23.7000_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.6200_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 23.8100_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 17.7000_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 11.5500_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.0700_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.7400_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 80.7000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 61.3000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 47.5000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.0000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 24.0000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 17.7000_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 24.5600_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 18.6700_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.2400_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.7400_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.8700_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.5100_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 66.7000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 59.2000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 40.8000_PETSC_REAL_KIND; t(i) = 1.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 30.7000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 25.7000_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.3000_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 25.9900_PETSC_REAL_KIND; t(i) = 2.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.9500_PETSC_REAL_KIND; t(i) = 2.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.3500_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 8.6200_PETSC_REAL_KIND; t(i) = 4.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 7.2000_PETSC_REAL_KIND; t(i) = 5.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 6.6400_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.6900_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 81.0000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 64.5000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 35.5000_PETSC_REAL_KIND; t(i) = 1.5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 13.3100_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 4.8700_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 12.9400_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 5.0600_PETSC_REAL_KIND; t(i) = 6.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 15.1900_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 14.6200_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 15.6400_PETSC_REAL_KIND; t(i) = 3.0000_PETSC_REAL_KIND; i = i + 1
    y(i) = 25.5000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 25.9500_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 81.7000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 61.6000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.8000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 29.8100_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 17.1700_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.3900_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 28.4000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 28.6900_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 81.3000_PETSC_REAL_KIND; t(i) = .5000_PETSC_REAL_KIND; i = i + 1
    y(i) = 60.9000_PETSC_REAL_KIND; t(i) = .7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 16.6500_PETSC_REAL_KIND; t(i) = 2.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 10.0500_PETSC_REAL_KIND; t(i) = 3.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 28.9000_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1
    y(i) = 28.9500_PETSC_REAL_KIND; t(i) = 1.7500_PETSC_REAL_KIND; i = i + 1

  end

  subroutine TaskWorker(ierr)

    PetscErrorCode ierr
    PetscReal x(n), f(1)
    PetscMPIInt tag
    PetscInt index
#if defined(PETSC_USE_MPI_F08)
    MPIU_Status status
#else
    MPIU_Status status(MPI_STATUS_SIZE)
#endif
    tag = IDLE_TAG
    f = 0.0
    ! Send check-in message to rank-0
    PetscCallMPI(MPI_Send(f, one, MPIU_SCALAR, zero, IDLE_TAG, PETSC_COMM_WORLD, ierr))
    do while (tag /= DIE_TAG)
      PetscCallMPI(MPI_Recv(x, nn, MPIU_SCALAR, zero, MPI_ANY_TAG, PETSC_COMM_WORLD, status, ierr))
#if defined(PETSC_USE_MPI_F08)
      tag = status%MPI_TAG
#else
      tag = status(MPI_TAG)
#endif
      if (tag == IDLE_TAG) then
        PetscCallMPI(MPI_Send(f, one, MPIU_SCALAR, zero, IDLE_TAG, PETSC_COMM_WORLD, ierr))
      else if (tag /= DIE_TAG) then
        index = tag
        ! Compute local part of residual
        PetscCall(RunSimulation(x, index, f(1), ierr))

        ! Return residual to rank-0
        PetscCallMPI(MPI_Send(f, one, MPIU_SCALAR, zero, tag, PETSC_COMM_WORLD, ierr))
      end if
    end do
    ierr = 0
  end

  subroutine RunSimulation(x, i, f, ierr)

    PetscReal x(n), f
    PetscInt, intent(in) :: i
    PetscErrorCode, intent(out) :: ierr
    f = y(i) - exp(-x(1)*t(i))/(x(2) + x(3)*t(i))
    ierr = 0
  end

  subroutine StopWorkers(ierr)

    integer checkedin
#if defined(PETSC_USE_MPI_F08)
    MPIU_Status status
#else
    MPIU_Status status(MPI_STATUS_SIZE)
#endif
    PetscMPIInt source
    PetscReal f(1), x(n)
    PetscErrorCode, intent(out) :: ierr

    checkedin = 0
    do while (checkedin < size - 1)
      PetscCallMPI(MPI_Recv(f, one, MPIU_SCALAR, MPI_ANY_SOURCE, MPI_ANY_TAG, PETSC_COMM_WORLD, status, ierr))
      checkedin = checkedin + 1
#if defined(PETSC_USE_MPI_F08)
      source = status%MPI_SOURCE
#else
      source = status(MPI_SOURCE)
#endif
      x(1:n) = 0.0
      PetscCallMPI(MPI_Send(x, nn, MPIU_SCALAR, source, DIE_TAG, PETSC_COMM_WORLD, ierr))
    end do
    ierr = 0
  end

! --------------------------------------------------------------------
!  FormFunction - Evaluates the function f(X) and gradient G(X)
!
!  Input Parameters:
!  tao - the Tao context
!  X   - input vector
!  dummy - not used
!
!  Output Parameters:
!  f - function vector

  subroutine FormFunction(ta, x, f, dummy, ierr)

    Tao ta
    Vec x, f
    PetscErrorCode ierr

    PetscInt i, checkedin
    PetscInt finished_tasks
    PetscMPIInt next_task
    PetscMPIInt tag, source
#if defined(PETSC_USE_MPI_F08)
    MPIU_Status status
#else
    MPIU_Status status(MPI_STATUS_SIZE)
#endif
    PetscInt dummy

    PetscReal, pointer :: f_v(:), x_v(:)
    PetscReal fval(1)

    ierr = 0

!     Get pointers to vector data
    PetscCall(VecGetArrayRead(x, x_v, ierr))
    PetscCall(VecGetArray(f, f_v, ierr))

!     Compute F(X)
    if (size == 1) then
      ! Single processor
      do i = 1, m
        PetscCall(RunSimulation(x_v, i, f_v(i), ierr))
      end do
    else
      ! Multiprocessor main
      next_task = zero
      finished_tasks = 0
      checkedin = 0

      do while (finished_tasks < m .or. checkedin < size - 1)
        PetscCallMPI(MPI_Recv(fval, one, MPIU_SCALAR, MPI_ANY_SOURCE, MPI_ANY_TAG, PETSC_COMM_WORLD, status, ierr))
#if defined(PETSC_USE_MPI_F08)
        tag = status%MPI_TAG
        source = status%MPI_SOURCE
#else
        tag = status(MPI_TAG)
        source = status(MPI_SOURCE)
#endif
        if (tag == IDLE_TAG) then
          checkedin = checkedin + 1
        else
          f_v(tag + 1) = fval(1)
          finished_tasks = finished_tasks + 1
        end if
        if (next_task < m) then
          ! Send task to worker
          PetscCallMPI(MPI_Send(x_v, nn, MPIU_SCALAR, source, next_task, PETSC_COMM_WORLD, ierr))
          next_task = next_task + one
        else
          ! Send idle message to worker
          PetscCallMPI(MPI_Send(x_v, nn, MPIU_SCALAR, source, IDLE_TAG, PETSC_COMM_WORLD, ierr))
        end if
      end do
    end if

!     Restore vectors
    PetscCall(VecRestoreArrayRead(x, x_v, ierr))
    PetscCall(VecRestoreArray(F, f_v, ierr))
  end

  subroutine FormStartingPoint(x)

    Vec x
    PetscReal, pointer :: x_v(:)
    PetscErrorCode ierr

    PetscCall(VecGetArray(x, x_v, ierr))
    x_v(1) = 0.15_PETSC_REAL_KIND
    x_v(2) = 0.008_PETSC_REAL_KIND
    x_v(3) = 0.01_PETSC_REAL_KIND
    PetscCall(VecRestoreArray(x, x_v, ierr))
  end
end module chwirut2fmodule

program main
  use chwirut2fmodule
  implicit none
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See additional variable declarations in the file chwirut2f.h

  PetscErrorCode ierr    ! used to check for functions returning nonzeros
  Vec x       ! solution vector
  Vec f       ! vector of functions
  Tao ta     ! Tao context

!  Initialize TAO and PETSc
  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))

!  Initialize problem parameters
  call InitializeData()

  if (rank == 0) then
!  Allocate vectors for the solution and gradient
    PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, x, ierr))
    PetscCallA(VecCreateSeq(PETSC_COMM_SELF, m, f, ierr))

!   The TAO code begins here

!   Create TAO solver
    PetscCallA(TaoCreate(PETSC_COMM_SELF, ta, ierr))
    PetscCallA(TaoSetType(ta, TAOPOUNDERS, ierr))

!   Set routines for function, gradient, and hessian evaluation
    PetscCallA(TaoSetResidualRoutine(ta, f, FormFunction, 0, ierr))

!   Optional: Set initial guess
    call FormStartingPoint(x)
    PetscCallA(TaoSetSolution(ta, x, ierr))

!   Check for TAO command line options
    PetscCallA(TaoSetFromOptions(ta, ierr))
!   SOLVE THE APPLICATION
    PetscCallA(TaoSolve(ta, ierr))

!   Free TAO data structures
    PetscCallA(TaoDestroy(ta, ierr))

!   Free PETSc data structures
    PetscCallA(VecDestroy(x, ierr))
    PetscCallA(VecDestroy(f, ierr))
    PetscCallA(StopWorkers(ierr))

  else
    PetscCallA(TaskWorker(ierr))
  end if

  PetscCallA(PetscFinalize(ierr))
end
!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      nsize: 3
!      args: -tao_monitor_short -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5
!      requires: !single
!
!
!TEST*/
