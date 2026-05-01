#pragma once

/*
  Distance-based localization kernels shared by the CPU and Kokkos backends of LETKF.
  All functions are pure (no PETSc state), so the same definitions are used host-only
  from `dalocalizationletkf.c` and host+device from `dalocalizationletkf.kokkos.cxx`.

  When compiled in a TU that has already included Kokkos, `KOKKOS_INLINE_FUNCTION` is
  defined and the functions are annotated for both host and device. Otherwise they fall
  back to plain `static inline`.

  Include `<Kokkos_Core.hpp>` (or any header that pulls it in) BEFORE this header to get the
  device-callable variants; otherwise the host-only fallbacks are silently selected.
*/

#include <petsc.h>
#include <petscda.h>

#if defined(KOKKOS_INLINE_FUNCTION)
  #define LETKF_KERNEL_FN          KOKKOS_INLINE_FUNCTION
  #define LETKF_KERNEL_UNREACHABLE Kokkos::abort
  #define LETKF_KERNEL_EXP(x)      Kokkos::exp(x) /* device-callable; PetscExpReal would resolve to host libm */
#else
  #define LETKF_KERNEL_FN               static inline
  #define LETKF_KERNEL_UNREACHABLE(msg) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, msg)
  #define LETKF_KERNEL_EXP(x)           PetscExpReal(x)
#endif

/* Gaspari-Cohn 5th-order piecewise polynomial. */
LETKF_KERNEL_FN PetscReal LETKFGaspariCohn(PetscReal distance, PetscReal radius)
{
  PetscReal r, r2, r3, r4, r5, val;

  if (radius <= 0.0) return 0.0;
  r = distance / radius;
  if (r >= 2.0) return 0.0;

  r2 = r * r;
  r3 = r2 * r;
  r4 = r3 * r;
  r5 = r4 * r;

  if (r <= 1.0) val = -0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0;
  else val = (1.0 / 12.0) * r5 - 0.5 * r4 + 0.625 * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
  return val;
}

/* Gaussian kernel exp(-d^2 / (2 r^2)), truncated at d = 2 r (value ~ exp(-2) ~ 0.135). */
LETKF_KERNEL_FN PetscReal LETKFGaussian(PetscReal distance, PetscReal radius)
{
  PetscReal r;

  if (radius <= 0.0) return 0.0;
  r = distance / radius;
  if (r >= 2.0) return 0.0;
  return LETKF_KERNEL_EXP(-0.5 * r * r);
}

/* Boxcar kernel: 1 inside the radius, 0 outside. */
LETKF_KERNEL_FN PetscReal LETKFBoxcar(PetscReal distance, PetscReal radius)
{
  if (radius <= 0.0) return 0.0;
  return distance < radius ? 1.0 : 0.0;
}

/* Squared cutoff distance beyond which the kernel is guaranteed to return zero. */
LETKF_KERNEL_FN PetscReal LETKFCutoffSquared(PetscDALETKFLocalizationType type, PetscReal radius)
{
  switch (type) {
  case PETSCDA_LETKF_LOC_GASPARI_COHN:
  case PETSCDA_LETKF_LOC_GAUSSIAN:
    return 4.0 * radius * radius; /* (2*radius)^2 */
  case PETSCDA_LETKF_LOC_BOXCAR:
    return radius * radius;
  default:
    /* Unreachable: callers PetscCheck() the type before reaching here, and LOC_NONE skips Q construction entirely. */
    LETKF_KERNEL_UNREACHABLE("LETKFCutoffSquared: invalid localization type");
    return 0.0;
  }
}

LETKF_KERNEL_FN PetscReal LETKFKernelEval(PetscDALETKFLocalizationType type, PetscReal distance, PetscReal radius)
{
  switch (type) {
  case PETSCDA_LETKF_LOC_GASPARI_COHN:
    return LETKFGaspariCohn(distance, radius);
  case PETSCDA_LETKF_LOC_GAUSSIAN:
    return LETKFGaussian(distance, radius);
  case PETSCDA_LETKF_LOC_BOXCAR:
    return LETKFBoxcar(distance, radius);
  default:
    /* Unreachable: see LETKFCutoffSquared(). */
    LETKF_KERNEL_UNREACHABLE("LETKFKernelEval: invalid localization type");
    return 0.0;
  }
}
