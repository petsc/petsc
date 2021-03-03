=================
Threads and PETSc
=================

With the advent of multicore machines as standard practice from laptops to supercomputers,
the issue of a hybrid MPI-thread (for example, MPI-OpenMP) "version" of PETSc is of
interest. In this model, one MPI process per CPU (node) and several threads on that CPU
(node) work together inside that MPI process.

The core PETSc team has come to the consensus that pure MPI using neighborhood collectives
and the judicious using of MPI shared memory (for data structures that you may not wish to
have duplicated on each MPI process due to memory constraints) will provide the best
performance for HPC simulation needs on current generation systems, next generation
systems and exascale systems. It is also a much simpler programming model then MPI +
threads (leading to simpler code).

Note that the PETSc team has no problems with proposals to replace the pure MPI
programming model with a different programming model but only with an alternative that is
demonstrably __better__, not with something more complicated that has not been
demonstrated to be better nor that has any technical reason to be believed to be any
better. Ever since the IBM SP2 in 1996 we've been told that "pure MPI won't scale to the
next generation of machine", this has yet to be true and there is no reason to believe
that it will be true.

Though the current and planned programming model for PETSc is pure MPI we do provide some
limited support for use with the hybrid MPI-thread model. These are discussed below. Many
people throw around the term "hybrid MPI-thread" as if it is a trivial change in the
programming model. It is not -- major ramifications must be understood if such a hybrid
approach is to be used successfully. Hybrid approaches can be developed in many ways that
affect usability and performance.

The simple model of PETSc with threads
======================================

One may contain all the thread operations inside the Mat and Vec classes, leaving the
user's programming model identical to what it is today. This model can be done in two ways
by having Vec and Mat class implementations that use

#. OpenMP compiler directives to parallelize some of the methods

#. POSIX threads (pthread) calls to parallelize some of the methods.

We tried this approach (with support for both OpenMP and pthreads) and found the code was
never faster than pure MPI and cumbersome to use hence we have removed it.

An alternative simple model of PETSc with threads
=================================================

Alternatively, on my have individual threads (OpenMP or others) to each manage their own
(sequential) PETSc objects (and each thread can interact only with its own objects). This
is useful when one has many small systems (or sets of ODEs) that must be integrated in an
"embarrassingly parallel" fashion.

To use this feature one must ``configure`` PETSc with the option
``--with-threadsafety --with-log=0 [--with-openmp or
--download-concurrencykit]``. ``$PETSC_DIR/src/ksp/ksp/tutorials/ex61f.F90`` demonstrates
how this may be used with OpenMP. The code uses a small number of ``#pragma omp critical``
in non-time-critical locations in the code and thus only works with OpenMP and not with
pthreads.

A more complicated model of PETSc with threads
==============================================

This would allow users to write threaded code that made PETSc calls, is not supported
because PETSc is not currently thread-safe. Because the issues involved with toolkits and
thread safety are complex, this short answer is almost meaningless. Thus, this page
attempts to explain how threads and thread safety relate to PETSc. Note that we are
discussing here only "software threads" as opposed to "hardware threads."

Threads are used in 2 main ways in HPC:

#. **Loop-level compiler control**. The C/C++/FORTRAN compiler manages the dispatching of
   threads at the beginning of a loop, where each thread is assigned non-overlapping
   selections from the loop. OpenMP, for example, defines standards (compiler directives)
   for indicating to compilers how they should "thread parallelize" loops.

#. **User control**. The programmer manages the dispatching of threads directly by
   assigning threads to tasks (e.g., a subroutine call). For example, consider POSIX
   threads (pthreads) or the user thread management in OpenMP.

Threads are merely streams of control and do not have any global data associated with
them. Any global variables (e.g., common blocks in FORTRAN) are "shared" by all the
threads; that is, any thread can access and change that data. In addition, any space
allocated (e.g., in C with malloc or C++ with new) to which a thread has a reference can
be read/changed by that thread. The only private data a thread has are the local variables
in the subroutines that it has called (i.e., the stack for that thread) or local variables
that one explicitly indicates are to be not shared in compiler directives.

In its simplest form, thread safety means that any memory (global or allocated) to which
more than one thread has access, has some mechanism to ensure that the memory remains
consistent when the various threads act upon it. This can be managed by simply associating
a lock with each "memory" and making sure that each thread locks the memory before
accessing it and unlocks when it has completed accessing the memory. In an object oriented
library, rather than associating locks with individual data items, one can think about
associating locks with objects; so that only a single thread can operate on an object at a
time.

.. note::

   PETSc is not *generically* thread-safe!

   All the PETSc objects created during a simulation do not have locks associated with
   them. Again, the reason is performance; ensuring atomic operations will almost
   certainly have a large impact on performance. Even with very inexpensive locks, there
   will still likely be a few "hot-spots". For example, threads may share a commmon vector
   or matrix, so any "setter" calls such as ``MatSetValues()`` would likely need to be
   serialized. ``VecGetArrayRead()``/``VecGetArrayWrite()`` would similarly face such
   bottlenecks.

Some concerns about a thread model for parallelism
==================================================

A thread model for parallelism of numerical methods appears to be powerful for problems
that can store their data in very simple (well controlled) data structures. For example,
if field data is stored in a two-dimensional array, then each thread can be assigned a
nonoverlapping slice of data to operate on. OpenMP makes managing much of this sort of
thing reasonably straightforward.

When data must be stored in a more complicated opaque data structure (for example an
unstructured grid or sparse matrix), it is more difficult to partition the data among the
threads to prevent conflict and still achieve good performance. More difficult, but
certainly not impossible. For these situations, perhaps it is more natural for each thread
to maintain its own private data structure that is later merged into a common data
structure. But to do this, one has to introduce a great deal of private state associated
with the thread, i.e., it becomes more like a "light-weight process".

In conclusion, at least for the PETSc package, the concept of being thread-safe is not
simple. It has major ramifications about its performance and how it would be used; it is
not a simple matter of throwing a few locks around and then everything is honky-dory.

If you have any comments/brickbats on this summary, please direct them to
petsc-maint@mcs.anl.gov; we are interested in alternative viewpoints.

.. seealso::

   The Problem with Threads, Edward A. Lee, Technical Report No. UCB/EECS-2006-1 January
   10, 2006
