
.. _chapter_network:

Networks
--------

The DMNetwork class provides
abstractions for representing general unstructured networks such as
communication networks, power grid, computer networks, transportation
networks, electrical circuits, graphs, and others.

Application flow
~~~~~~~~~~~~~~~~

The general flow of an application code using DMNetwork is as
follows:

#. Create a network object.

   .. code-block::

      DMNetworkCreate(MPI_Comm comm, DM *dm);

#. Create components and register them with the network. A “component”
   is specific application data at a vertex/edge of the network required
   for its residual evaluation. For example, components could be
   resistor/inductor data for circuit applications, edge weights for
   graph problems, or generator/transmission line data for power grids.
   Components are registered by calling

   .. code-block::

      DMNetworkRegisterComponent(DM dm, const char *name, size_t size, PetscInt *compkey);

   Here, ``name`` is the component name, ``size`` is the size of
   component data type, and ``compkey`` is an integer key that can be
   used for setting/getting the component at a vertex or an edge.

#. A DMNetwork can consist of one or more physical subnetworks. When
   multiple physical subnetworks are used one can (optionally) provide
   coupling information between subnetworks which consist only of shared vertices of the physical subnetworks. The
   topological sizes of the network are set by calling

   .. code-block::

      DMNetworkSetNumSubNetworks(DM dm, PetscInt nsubnet, PetscInt Nsubnet);

   Here, ``nsubnet`` and ``Nsubnet`` are the local and global number of subnetworks.

#. A subnetwork is added to the network by calling

   .. code-block::

      DMNetworkAddSubnetwork(DM dm, const char* name, PetscInt ne, PetscInt edgelist[], PetscInt *netnum);

   Here ``name`` is the subnetwork name, ``ne`` is the number of local edges on the subnetwork, and ``edgelist`` is the connectivity for the subnetwork.
   The output ``netnum`` is the global numbering of the subnetwork in the network.
   Each element of ``edgelist`` is an integer array of size ``2*ne``
   containing the edge connectivity for the subnetwork.

   | As an example, consider a network comprised of 2 subnetworks that
     are coupled. The topological information for the network is as
     follows:
   | subnetwork 0: v0 — v1 — v2 — v3
   | subnetwork 1: v1 — v2 — v0
   | The two subnetworks are coupled by merging vertex 0 from subnetwork 0 with vertex 2 from subnetwork 1.
   | The ``edgelist`` of this network is
   | edgelist[0] = {0,1,1,2,2,3}
   | edgelist[1] = {1,2,2,0}

   The coupling is done by calling

   .. code-block::

      DMNetworkAddSharedVertices(DM dm, PetscInt anet, PetscInt bnet, PetscInt nsv, PetscInt asv[], PetscInt bsv[]);

   Here ``anet`` and ``bnet`` are the first and second subnetwork global numberings returned by ``DMNetworkAddSubnetwork()``,
   ``nsv`` is the number of vertices shared by the two subnetworks, ``asv`` and ``bsv`` are the vertex indices in the subnetwork ``anet`` and ``bnet`` .

#. The next step is to have DMNetwork create a bare layout (graph) of
   the network by calling

   .. code-block::

      DMNetworkLayoutSetUp(DM dm);

#. After completing the previous steps, the network graph is set up, but
   no physics is associated yet. This is done by adding the components
   and setting the number of variables to the vertices and edges.

   A component and number of variables are added to a vertex/edge by calling

   .. code-block::

      DMNetworkAddComponent(DM dm, PetscInt p, PetscInt compkey, void* compdata, PetscInt nvar)

   where ``p`` is the network vertex/edge point in the range obtained by
   either ``DMNetworkGetVertexRange()``/``DMNetworkGetEdgeRange()``, ``DMNetworkGetSubnetwork()``, or ``DMNetworkGetSharedVertices()``;
   ``compkey`` is the component key returned when registering the component
   (``DMNetworkRegisterComponent()``); ``compdata`` holds the data for the
   component; and ``nvar`` is the number of variables associated to the added component at this network point. DMNetwork supports setting multiple components
   at a vertex/edge. At a shared vertex, DMNetwork currently requires the owner process of the vertex adds all the components and number of variables.

   DMNetwork currently assumes the component data to be stored in a
   contiguous chunk of memory. As such, it does not do any
   packing/unpacking before/after the component data gets distributed.
   Any such serialization (packing/unpacking) should be done by the
   application.

#. Set up network internal data structures.

   .. code-block::

      DMSetUp(DM dm);

#. Distribute the network (also moves components attached with
   vertices/edges) to multiple processors.

   .. code-block::

      DMNetworkDistribute(DM dm, const char partitioner[], PetscInt overlap, DM *distDM);

#. Associate the ``DM`` with a PETSc solver:

   .. code-block::

      KSPSetDM(KSP ksp, DM dm) or SNESSetDM(SNES snes, DM dm) or TSSetDM(TS ts, DM dm).

Utility functions
~~~~~~~~~~~~~~~~~

``DMNetwork`` provides several utility functions for operations on the
network. The most commonly used functions are: obtaining iterators for
vertices/edges,

.. code-block::

   DMNetworkGetEdgeRange(DM dm, PetscInt *eStart, PetscInt *eEnd);

.. code-block::

   DMNetworkGetVertexRange(DM dm, PetscInt *vStart, PetscInt *vEnd);

.. code-block::

   DMNetworkGetSubnetwork(DM dm, PetscInt netnum, PetscInt *nv, PetscInt *ne, const PetscInt **vtx, const PetscInt **edge);

checking the status of a vertex,

.. code-block::

   DMNetworkIsGhostVertex(DM dm, PetscInt p, PetscBool *isghost);

.. code-block::

   DMNetworkIsSharedVertex(DM dm, PetscInt p, PetscBool *isshared);

and retrieving local/global indices of vertex/edge component variables for
inserting elements in vectors/matrices,

.. code-block::

   DMNetworkGetLocalVecOffset(DM dm, PetscInt p, PetscInt compnum, PetscInt *offset);

.. code-block::

   DMNetworkGetGlobalVecOffset(DM dm, PetscInt p, PetscInt compnum, PetscInt *offsetg).

In network applications, one frequently needs to find the supporting
edges for a vertex or the connecting vertices covering an edge. These
can be obtained by the following two routines.

.. code-block::

   DMNetworkGetConnectedVertices(DM dm, PetscInt edge, const PetscInt *vertices[]);

.. code-block::

   DMNetworkGetSupportingEdges(DM dm, PetscInt vertex, PetscInt *nedges, const PetscInt *edges[]).

Retrieving components and number of variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The components and the corresponding number of variables set at a vertex/edge can be accessed by

.. code-block::

   DMNetworkGetComponent(DM dm, PetscInt p, PetscInt compnum, PetscInt *compkey, void **component, PetscInt *nvar)

input ``compnum`` is the component number, output ``compkey`` is the key set by ``DMNetworkRegisterComponent``. An example
of accessing and retrieving the components and number of variables at vertices is:

.. code-block::

   PetscInt Start,End,numcomps,key,v,compnum;
   void *component;

   DMNetworkGetVertexRange(dm, &Start, &End);
   for (v = Start; v < End; v++) {
     DMNetworkGetNumComponents(dm, v, &numcomps);
     for (compnum=0; compnum < numcomps; compnum++) {
       DMNetworkGetComponent(dm, v, compnum, &key, &component, &nvar);
       compdata = (UserCompDataType)(component);
     }
   }

The above example does not explicitly use the component key. It is
used when different component types are set at different vertices. In
this case, ``compkey`` is used to differentiate the component type.

.. bibliography:: /petsc.bib
    :filter: docname in docnames


