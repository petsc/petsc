.. _doc_creepycrawly:

*************************
Reporting Bugs And Errors
*************************

The PETSc maintenance e-mail petsc-maint@mcs.anl.gov is intended for users to:

- Report bugs.
- Ask for clarification.
- Ask for help in tracking down bugs.
- Request new features within PETSc.
- Recommmend changes or additions to the development team.

.. Note::

   We respond to almost all email the same day and many within the hour. Please do not
   send maintenance requests to the individual PETSc authors; all e-mail to
   petsc-maint@mcs.anl.gov is automatically distributed to all of the PETSc authors, so
   our response time using this address will be fastest.

Before sending a bug report, please consult the :ref:`FAQ <doc_config_faq>` to determine
whether a fix or work-around to the problem already exists. Also, see the chapter on
:ref:`performance tuning <ch_performance>` in the PETSc users manual for guidelines on
achieving good efficiency within PETSc codes.

Guidelines For Bug Reports
==========================

The more information that you convey about a bug, the easier it will be for us to target
the problem. We suggest providing the following information:

.. admonition:: Don'ts
   :class: yellow

   - Please do NOT send winmail.dat Microsoft email attachments; we cannot read them.
   - Please do NOT stick huge files like ``configure.log`` DIRECTLY into the email
     message. Instead, include them as attachments.
   - Please do NOT paste your **entire** codes DIRECTLY into the email message. Instead,
     include them as attachments. Small snippets of code are acceptable however.

.. admonition:: Do's

   - Detailed steps to recreate the problem if possible.
   - Copy of the **complete** error message if pheasible, otherwise include the full error
     message as an attachment.
   - If the problem involves installation, send the entire ``configure.log`` and
     ``make.log`` as attachments.

     - ``configure.log`` can be found either at ``$PETSC_DIR/configure.log``,
       ``$PETSC_DIR/configure.log.bkp`` (which holds the second-most recent
       ``configure.log``), or in
       ``$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/configure.log[.bkp]``.

     - ``make.log`` can be found in the same places as listed above, however note that
       there is no ``make.log.bkp`` so be sure to not overwrite your ``make.log`` with
       additional build attempts.
   - Machine type: (e.g. HPC, laptop, etc.)
   - OS Version and Type: (run uname -a to get the version number)
   - PETSc Version: (run PETSc program with -version, or look in
     ``$PETSC_DIR/include/petscversion.h``)
   - MPI implementation: (e.g. MPICH, LAM, IBM, SGI)
   - Compiler and version: (e.g. Gnu C, Gnu g++, native C)
   - Probable PETSc component: (e.g. ``Mat``, ``Vec``, ``DM``, ``KSP``, etc.)

