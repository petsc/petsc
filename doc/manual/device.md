(ch_device)=

# Devices, contexts, and GPU streams

`Vec` and `Mat` let applications select CPU or GPU implementations without changing
their numerical code. The GPU design is described in {cite}`mills2021,mills2024`.
Faibussowitsch et al. {cite}`faibussowitsch2023` discuss the stream model and
propose further support for asynchronous execution.

## The same computation on CPU or GPU

The vector type determines where the following computation runs:

```c
#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec       x, y;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 100000));
  PetscCall(VecSetType(x, VECSTANDARD));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(y, 1.0));
  PetscCall(VecAXPY(y, 2.0, x));
  PetscCall(VecNorm(y, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm: %g\n", (double)norm));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}
```

The example uses CPU vectors by default. With a CUDA-enabled PETSc build, run
it with `-vec_type cuda` to use the GPU. `VecSetFromOptions()` applies this
choice, and `VecDuplicate()` gives `y` the same type as `x`. The computation
is unchanged; `-vec_type standard` selects the CPU implementation explicitly.

With CUDA vectors, `VecSet()` fills the device arrays, `VecAXPY()` calls cuBLAS
to compute $y \leftarrow y + 2x$ on the GPU, and `VecNorm()` computes the local
norm there. Only the scalar result is returned to the CPU; the vector entries
stay on the GPU. For multiple MPI processes, PETSc also combines the local results
to obtain the global norm.

## Host and device memory

A `VECCUDA` vector can have both a host array, accessible to the CPU, and a device
array, accessible to GPU kernels. PETSc tracks which copy has the latest values
and transfers data when an operation needs it on the other side. Writing on one
side makes the other copy stale. With `-vec_type cuda` in the example above,
both initialization and computation use the device arrays, so no host array
access is needed.

Use the array access routine that matches what the application will do:

| CPU access | Routine | Effect |
|------------|---------|--------|
| Read entries | `VecGetArrayRead()` | Copies values from the GPU if needed; preserves their validity on the GPU. |
| Read and modify entries | `VecGetArray()` | Copies values from the GPU if needed; after `VecRestoreArray()`, PETSc copies changes to the GPU when a GPU operation needs them. |
| Write without preserving old values | `VecGetArrayWrite()` | Skips copying old values from the GPU. Entries not explicitly written have unspecified values; they are not guaranteed to be zero or retain their previous values. |

Pair each getter with its corresponding restore routine and do not use the
pointer after restoring it. For example, given a CUDA vector `x` whose earlier
work has completed, the following fills its local entries on the CPU:

```c
PetscScalar *a;
PetscInt     n;

PetscCall(VecGetLocalSize(x, &n));
PetscCall(VecGetArrayWrite(x, &a));
for (PetscInt i = 0; i < n; ++i) a[i] = (PetscScalar)(i + 1);
PetscCall(VecRestoreArrayWrite(x, &a));
```

The next GPU operation that reads `x` transfers these values from host to device
memory. Repeating this pattern between GPU operations can make transfers
dominate the computation. Use `-log_view` to inspect their counts and sizes;
see {any}`sec_profgpu`.

For application GPU kernels, use the `VecCUDAGetArray()` family to obtain a
device pointer. `VecGetArrayAndMemType()`
provides a pointer and its memory type for code supporting several backends.
The access mode and matching restore routine matter on the device too.

## The current context and the root stream

A stream is a queue of operations that execute in submission order.
`PetscDevice` identifies a backend and device. `PetscDeviceContext` associates
that device with an execution stream and the library handles used to submit
work. PETSc's built-in CUDA and HIP vector and matrix implementations use PETSc's
*current context*; external packages such as H2OPUS may use their own streams.
Initially this is the *root context*, owned by PETSc.
`PetscDeviceContextSetCurrentContext()` selects another context for subsequent
operations.

In the first GPU example, `VecSet()`, `VecAXPY()`, and `VecNorm()` use the same
stream, so the norm sees the completed update. `VecAXPY()` can return while the
GPU is still working. Here, `VecNorm()` waits for its scalar result before
returning, so no explicit synchronization is needed between the two calls.
This follows the cuBLAS behavior for
[scalar results returned to host memory](https://docs.nvidia.com/cuda/cublas/index.html#scalar-parameters).

The root uses `PETSC_STREAM_DEFAULT` unless configured otherwise. For CUDA this
uses the default stream, also called the null stream or stream 0. Its usual
legacy behavior orders work with other blocking streams, but does not order
work with nonblocking streams. CUDA also supports a per-thread default-stream
mode; see its [stream synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).
The null stream does not mean that every call waits on the CPU.

To use a nonblocking root stream, run the application with
`-root_device_context_stream_type nonblocking`. The choices are
`-root_device_context_stream_type (default|nonblocking|default_with_barrier|nonblocking_with_barrier)`.
The `WITH_BARRIER` types request a stream synchronization at the end of PETSc
asynchronous operations.

Use the option above for an application-wide choice. For concurrency within a
routine, create child contexts as shown below, leaving the root configuration
under the application's control. The public `PetscDeviceContextSetStreamType()`
API is available when configuring a context explicitly; call
`PetscDeviceContextSetUp()` afterwards.

Public APIs that take a context argument require a concrete context or, where
supported, `PetscDeviceContextDefault`.

`PetscDeviceContextDefault` selects the default stream on the current context's
device, even when the current context uses another stream. This placeholder is
resolved when an operation needs a context. Use
`PetscDeviceContextGetDefaultContext()` to obtain a concrete handle that remains
bound to that device if the current context later changes.

## Independent updates on two streams

Suppose `x1`, `y1`, `x2`, and `y2` are separate CUDA vectors on the same GPU,
created and initialized on the current context. On one stream,

```c
PetscCall(VecAXPY(y1, 2.0, x1));
PetscCall(VecAXPY(y2, 3.0, x2));
```

the two updates execute in order, even though their data are independent.
To allow overlap, create two nonblocking child contexts and submit one update
to each. The following replaces the two calls above; `norm1` and `norm2` are
host results. Include `<petscdevice.h>` for the context routines:

```c
PetscDeviceContext  parent;
PetscDeviceContext *work;
PetscReal           norm1, norm2;

PetscCall(PetscDeviceContextGetCurrentContext(&parent));
PetscCall(PetscDeviceContextForkWithStreamType(parent, PETSC_STREAM_NONBLOCKING, 2, &work));
PetscCall(PetscDeviceContextSetCurrentContext(work[0]));
PetscCall(VecAXPY(y1, 2.0, x1));
PetscCall(PetscDeviceContextSetCurrentContext(work[1]));
PetscCall(VecAXPY(y2, 3.0, x2));
PetscCall(PetscDeviceContextSetCurrentContext(parent));
PetscCall(PetscDeviceContextJoin(parent, 2, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &work));
PetscCall(VecNorm(y1, NORM_2, &norm1));
PetscCall(VecNorm(y2, NORM_2, &norm2));
```

The fork orders both children after earlier work on `parent`, including vector
initialization. The join orders subsequent work on `parent` after both updates
and releases the child contexts and their array. We return to `parent` before
releasing the children because a current context must remain alive. Fork and
join establish stream dependencies; they do not require the CPU to wait for the
updates. The norms return completed scalar results. Actual overlap depends on
the hardware and workload; it need not improve performance.

If work on a context named `producer` writes values that work on another
context, `consumer`, will read, call
`PetscDeviceContextWaitForContext(consumer, producer)` after submitting the
writes and before submitting the reads. This orders the GPU work without
requiring the CPU to wait. `PetscDeviceContextSynchronize(dctx)` waits on the
CPU; use it before CPU access to memory still in use by that context, including
unfinished data transfers.

Application kernels can use `PetscDeviceContextGetStreamHandle()` to obtain the
native stream and submit work in the same order as PETSc operations.

```{eval-rst}
.. bibliography:: /petsc.bib
   :filter: docname in docnames
```
