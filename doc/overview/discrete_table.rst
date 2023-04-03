.. _dm_table:

============================================
Summary of Discretization Management Systems
============================================

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * -
     - ``DMType``
     - Constructor
     - External Packages
     - Details
   * - Structured grids
     - ``DMDA``
     - ``DMDACreate3d()``
     -
     -
   * - Staggered structured grids
     - ``DMSTAG``
     - ``DMStagCreate3d()``
     -
     -
   * - Octrees
     - ``DMFOREST``
     - ``DMForestSetBaseDM()``
     - ``p4est``
     -
   * - Networks
     - ``DMNETWORK``
     - ``DMNetworkCreate()``
     -
     -
   * - Particles
     - ``DMSWARM``
     - ``DMSwarmSetCellDM()``
     -
     -
   * - Unstructured grids
     - ``DMPLEX``
     - ``DMPlexCreate()``
     -
     - Support for finite elements and volumes

