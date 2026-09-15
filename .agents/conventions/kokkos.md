# Kokkos / Device Code

For an unreachable guard (a `default:` arm or "can't happen" branch) inside a `KOKKOS_INLINE_FUNCTION`, use `Kokkos::abort("message")`. `SETERRQ`/`SETERRABORT` are not device-callable.

When a persistent workspace view is processed in chunks, only build a `Kokkos::subview` for the active range when a consumer actually reads the view extent (e.g. `KokkosBatched::TeamVectorGMRES` infers batch size from `view.extent(0)`). If every kernel is bounded by an explicit count parameter (`RangePolicy(0, n_active)`, or a function arg like `n_batch`), pass the full-capacity view directly — the subview adds no safety and obscures intent.
