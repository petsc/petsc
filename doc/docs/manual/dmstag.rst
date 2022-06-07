.. _chapter_stag:

DMStag: Staggered, Structured Grids in PETSc
--------------------------------------------

For structured (aka "regular") grids with staggered data (living on elements, faces, edges,
and/or vertices), the ``DMSTAG`` object is available. This can
be useful for problems in many domains, including fluid flow, MHD, and seismology.

It is possible, though cumbersome, to implement a staggered-grid code using multiple ``DMDA`` objects, or a single multi-component ``DMDA`` object where some degrees of freedom are unused. DMStag was developed for two main purposes:

1. To help manage some of the burden of choosing and adhering to indexing conventions (in parallel)
2. To provide a uniform abstraction for which scalable solvers and preconditioners may be developed (in particular, using ``PCFIELDSPLIT`` and ``PCMG``).

DMStag is design to behave much
like :ref:`DMDA <sec_struct>`, with a couple of important distinctions, and borrows some terminology
from :doc:`DMPlex <dmplex>`.

Terminology
~~~~~~~~~~~

Like a DMPlex object, a DMStag represents a `cell complex <https://en.wikipedia.org/wiki/CW_complex>`__,
distributed in parallel over the ranks of an ``MPI_Comm``. It is, however,
a very regular complex, consisting of a structured grid of :math:`d`-dimensional cells, with :math:`d \in \{1,2,3\}`,
which are referred to as *elements*, :math:`d-1` dimensional cells defining boundaries between these elements,
and the boundaries of the domain, and in 2 or more dimensions, boundaries of *these* cells,
all the way down to 0 dimensional cells referred to as *vertices*. In 2 dimensions, the 1-dimensional
element boundaries are referred to as *faces*. In 3 dimensions, the 2-dimensional element boundaries
are referred to as *faces* and the 1-dimensional boundaries between faces are referred to as *edges*
The set of cells of a given dimension is referred to as a *stratum*; a DMStag object of dimension :math:`d`
represents a complete cell complex with :math:`d+1` strata.

Each stratum has a constant number of unknowns (which may be zero) associated with each cell.
The distinct unknowns associated with each cell are referred to as *components*.

The structured grid, is like with DMDA, decomposed via a Cartesian product of decompositions in each dimension,
giving a rectangular local subdomain on each rank. This is extended by an element-wise stencil width
of *ghost* elements to create an atlas of overlapping patches.

Working with vectors and operators (matrices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMStag allows the user to reason almost entirely about a global indexing of elements.
Element indices are simply 1-3 ``PetscInt`` values, starting at :math:`0`, in the
back, bottom, left corner of the domain. For instance, element :math:`(1,2,3)`, in 3D,
is the element second from the left, third from the bottom, and fourth from the back
(regardless of how many MPI ranks are used).

To refer to cells (elements, faces, edges, and vertices), a value of
``DMStagStencilLocation`` is used, relative to the element index. The element
itself is referred to with ``DMSTAG_ELEMENT``, the top right vertex (in 2D)
or the top right edge (in 3D) with ``DMSTAG_UP_RIGHT``, the back bottom left
corner in 3D with ``DMSTAG_BACK_DOWN_LEFT``, and so on.

:numref:`figure_dmstag_indexing` gives a few examples in 2D.

.. figure:: /images/docs/manual/dmstag_indexing.svg
  :name: figure_dmstag_indexing

  Locations in DMStag are indexed according to global element indices (here, two in 2D) and a location name. Elements have unique names but other locations can be referred to in more than one way. Element colors correspond to a parallel decomposition, but locations on the grid have names which are invariant to this. Note that the face on the top right can be referred to as being to the left of a "dummy" element :math:`(3,3)` outside the physical domain.

Crucially, this global indexing scheme does not include any "ghost" or "padding" unknowns outside the physical domain.
This is useful for higher-level operations such as computing norms or developing physics-based solvers. However
(unlike ``DMDA``), this implies that the global ``Vec``\s do not have a natural block structure, as different
strata have different numbers of points (e.g. in 1D there is an "extra" vertex on the right). This regular block
structure is, however, very useful for the *local* representation of the data, so in that case *dummy* DOF
are included, drawn as grey in  :numref:`figure_dmstag_local_global`.

.. figure:: /images/docs/manual/dmstag_local_global.svg
  :name: figure_dmstag_local_global

  Local and global representations for a 2D DMStag object, 3 by 4 elements, with one degree of freedom on each of the the three strata: element (squares), faces (triangles), and vertices (circles). The cell complex is parallelized across 4 MPI ranks. In the global representation, the colors correspond to which rank holds the native representation of the unknown. The 4 local representations are shown, with an (elementwise) stencil "box" stencil width of 1. Unknownd are color by their native rank. Dummy unknowns, which correspond to no global degree of freedom, are colored grey. Note that the local representations have have a natural block size of 4, and the global representation has no natural block size.


For working with ``Vec`` data, this approach is used to allow direct access to a multi-dimensional, regular-blocked
array. To avoid the user having to know about the :ref:`internal numbering conventions used <sec_dmstag_numbering>`,
helper functions are used to produce the proper final integer index for a given location and component, referred to as a "slot".
Similarly to ``DMDAVecGetArrayDOF()``, this uses a :math:`d+1` dimensional array in :math:`d` dimensions.
The following snippet give an example of this usage.

.. literalinclude:: /../src/dm/impls/stag/tests/ex51.c
  :start-at: /* Set
  :end-at: PetscCall(DMStagVecRestoreArray

DMStag provides a stencil-based method for getting and setting entries of ``Mat`` and ``Vec`` objects.
The follow excerpt from `DMStag Tutorial ex1 <../../src/dm/tutorials/ex1.c.html>`__ demonstrates
the idea. For more, see the manual page for ``DMStagMatSetValuesStencil()``.

.. literalinclude:: /../src/dm/impls/stag/tutorials/ex1.c
  :start-at: /* Velocity is either a BC
  :end-at: PetscCall(DMStagMatSetValuesStencil

The array-based approach is likely to be more efficient when working with ``Vec``\s.

Coordinates
~~~~~~~~~~~

DMStag, unlike DMDA, supports two approaches to defining coordinates. This is captured by which type of ``DM``
is used to represent the coordinates. No default is imposed, so the user must directly or indirectly call
``DMStagSetCoordinateDMType()``.

If a second ``DMSTAG`` object is used to represent coordinates in "explicit" form, behavior is much like DMDA - the coordinate DM
has :math:`d` DOF on each stratum corresponding to coordinates associated with each cell.

If ``DMPRODUCT`` is used instead,  coordinates are represented by a ``DMPRODUCT`` object referring to a
Cartesian product of 1D ``DMStag`` objects, each of which features explicit coordinates as just mentioned.

Navigating these nested ``DM``\s can be tedious, but note the existence of helper functions like
``DMStagSetUniformCoordinatesProduct()`` and ``DMStagGetProductCoordinateArrays()``.

.. _sec_dmstag_numbering:

Numberings and internal data layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While DMStag aims to hide the details of its internal data layout, for debugging, optimization, and
customization purposes, it can be important to know how DMStag internally numbers unknowns.

Internally, each cell is canonically associated with an element (top-level cell). For purposes of local,
regular-blocked storage, an element is grouped with lower-dimensional cells left of, below ("down"), and behind ("back") it.
This means that "canonical" values of ``DMStagStencilLocation`` are ``DMSTAG_ELEMENT``, plus all entries consisting only of "LEFT", "DOWN", and "BACK". In general, these are the most efficient values to use, unless convenience dictates otherwise, as they are the ones used internally.

When creating the decomposition of the domain to local ranks, and extending these local domains to handle overlapping halo regions and boundary ghost unknowns, this same per-element association is used. This has the advantage of maintaining a regular blocking, but may not be optimal in some situations in terms of data movement.

Numberings are, like ``DMDA``, based on a local "x-fastest, z-slowest" or "PETSc" ordering of elements (see :ref:`sec_ao`), with ordering of locations canonically associated with each element decided by considering unknowns on each cell
to be located at the center of their cell, and using a nested ordering of the same style. Thus, in 3-D, the ordering of the 8 canonical ``DMStagStencilLocation`` values associated
with an element is

.. code-block::

   DMSTAG_BACK_DOWN_LEFT
   DMSTAG_BACK_DOWN
   DMSTAG_BACK_LEFT
   DMSTAG_BACK
   DMSTAG_DOWN_LEFT
   DMSTAG_DOWN
   DMSTAG_LEFT
   DMSTAG_ELEMENT

Multiple DOF associated with a given point are stored sequentially (as with DMDA).

For local ``Vec``\s, this gives a regular-blocked numbering, with the same number of unknowns associated with each element, including some "dummy" unknowns which to not correspond to any (local or global) unknown in the global representation. See :numref:`figure_dmstag_numbering_local` for an example.

In the global representation, only physical unknowns are numbered (using the same "Z" ordering for unknowns which are present), giving irregular numbers of unknowns, depending on whether a domain boundary is present. See :numref:`figure_dmstag_numbering_global` for an example.

.. figure:: /images/docs/manual/dmstag_numbering_global.svg
  :name: figure_dmstag_numbering_global

  Global numbering scheme for a 2D DMStag object with one DOF per stratum. Note that the numbering depends on the parallel decomposition (over 4 ranks, here).


.. figure:: /images/docs/manual/dmstag_numbering_local.svg
  :name: figure_dmstag_numbering_local

  Local numbering scheme on rank 1 (Cf. :numref:`figure_dmstag_local_global`) for a 2D DMStag object with one DOF per stratum. Note that dummy locations (grey) are used to give a regular block size (here, 4).

It should be noted that this is an *interlaced* (AoS) representation. If a segregated (SoA) representation is required,
one should  use ``DMCOMPOSITE`` collecting several DMStag objects, perhaps using ``DMStagCreateCompatibleDMStag()`` to
quickly create additional DMStag objects from an initial one.

