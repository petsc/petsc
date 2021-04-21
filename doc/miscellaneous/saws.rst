=======================================
SAWs - Scientific Application Webserver
=======================================

SAWs is a software library that uses threads, sockets, and locks to
allow a client program to read (and change when desired) variables in
the server (PETSc) application).

Requirements and installation of the SAWs
-----------------------------------------
One can simply add ``--download-saws`` to the arguments for ``./configure`` when configuring PETSc.

Alternately, a current version of SAWs is available from ``https://bitbucket.org/saws/saws``
and one can add  the flag  ``--with-saws-dir=/directorywhereamsisinstalled``.

Usage from PETSc
----------------

To examine options in a browser, run PETSc applications with
``-saws_view saws -ts_view_pre saws`` and point your brower to
``http://localhost:8080`` or, if you are running your application on a
different machine, pass the name of that machine to the browser.
