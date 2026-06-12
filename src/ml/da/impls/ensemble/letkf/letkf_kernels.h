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

  Radius semantics differ by kernel; the user-supplied `radius` always controls the
  effective support but the corresponding cutoff distance varies:
    - Gaspari-Cohn: compactly supported with cutoff at distance = 2*radius
    - Gaussian:     truncated at distance = 2*radius (value ~ exp(-2) ~ 0.135)
    - Boxcar:       cutoff at distance = radius (1 inside, 0 outside)
  Set `radius` per kernel rather than expecting identical falloff across choices.
*/

#include <petsc.h>
#include <petscda.h>

#if defined(KOKKOS_INLINE_FUNCTION)
  #define LETKF_KERNEL_FN               KOKKOS_INLINE_FUNCTION
  #define LETKF_KERNEL_UNREACHABLE(...) Kokkos::abort(__VA_ARGS__)
  #define LETKF_KERNEL_EXP(x)           Kokkos::exp(x)  /* device-callable; PetscExpReal would resolve to host libm */
  #define LETKF_KERNEL_SQRT(x)          Kokkos::sqrt(x) /* device-callable; PetscSqrtReal would resolve to host libm */
#else
  #define LETKF_KERNEL_FN               static inline
  #define LETKF_KERNEL_UNREACHABLE(...) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, __VA_ARGS__)
  #define LETKF_KERNEL_EXP(x)           PetscExpReal(x)
  #define LETKF_KERNEL_SQRT(x)          PetscSqrtReal(x)
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

/* Cutoff distance beyond which the kernel is guaranteed to return zero. Callers square
   the result inline rather than going through a sqrt(cutoff^2) round-trip so the bbox
   prune in PetscDALETKFGatherObsBbox() does not pick up a 1-ulp shrink that could drop
   a boundary observation under the BOXCAR kernel's strict (distance < radius) test. */
LETKF_KERNEL_FN PetscReal LETKFCutoff(PetscDALETKFLocalizationType type, PetscReal radius)
{
  switch (type) {
  case PETSCDA_LETKF_LOC_GASPARI_COHN:
  case PETSCDA_LETKF_LOC_GAUSSIAN:
    return 2.0 * radius;
  case PETSCDA_LETKF_LOC_BOXCAR:
    return radius;
  case PETSCDA_LETKF_LOC_NONE:
  case PETSCDA_LETKF_LOC_NUM_TYPES:
    break;
  }
  /* Unreachable: callers PetscCheck() the type before reaching here, and LOC_NONE skips Q construction entirely. */
  LETKF_KERNEL_UNREACHABLE("LETKFCutoff: invalid localization type");
  return 0.0;
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
  case PETSCDA_LETKF_LOC_NONE:
  case PETSCDA_LETKF_LOC_NUM_TYPES:
    break;
  }
  /* Unreachable: see LETKFCutoff(). */
  LETKF_KERNEL_UNREACHABLE("LETKFKernelEval: invalid localization type");
  return 0.0;
}

/* Squared distance between two coordinate tuples with per-dimension minimum-image periodicity.
   For each d with bd[d] > 0 the difference is folded into [-bd[d]/2, bd[d]/2]; non-periodic dims
   pass through. The shape is identical across both passes of both backends, so factor it out. */
LETKF_KERNEL_FN PetscReal LETKFPeriodicDist2(PetscInt dim, const PetscReal *v, const PetscReal *o, const PetscReal *bd)
{
  PetscReal dist2 = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    PetscReal diff = v[d] - o[d];
    if (bd[d] > 0.0) {
      PetscReal L = bd[d];
      if (diff > 0.5 * L) diff -= L;
      else if (diff < -0.5 * L) diff += L;
    }
    dist2 += diff * diff;
  }
  return dist2;
}

/* Localization weight for a single (vertex, observation) pair: zero outside the cutoff or when the
   kernel itself returns zero, else the kernel value. Encapsulates the dist2 + cutoff2 + KernelEval
   triple shared by both passes of the AIJ and Kokkos Q-construction loops. */
LETKF_KERNEL_FN PetscReal LETKFRowWeight(PetscDALETKFLocalizationType type, PetscReal radius, PetscReal cutoff2, PetscInt dim, const PetscReal *v, const PetscReal *o, const PetscReal *bd)
{
  PetscReal dist2 = LETKFPeriodicDist2(dim, v, o, bd);
  if (dist2 >= cutoff2) return 0.0;
  return LETKFKernelEval(type, LETKF_KERNEL_SQRT(dist2), radius);
}
