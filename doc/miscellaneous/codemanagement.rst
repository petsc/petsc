===============
Code Management
===============

We list some of the techniques that may be used to increase one's
efficiency when developing PETSc application codes. We have learned to
use these techniques ourselves, and they have improved our efficiency
tremendously.

Editing and Compiling
---------------------

The biggest time sink in code development is generally the cycle of
EDIT-COMPILE-LINK-RUN. We often see users working in a single window
with a cycle such as:

-  Edit a file with ``emacs`` or ``vim``.
-  Exit ``emacs`` or ``vim``.
-  Run ``make`` and see some error messages.
-  Start ``emacs`` or ``vim`` and try to fix the errors; often starting
   the editor hides the error messages so that users cannot remember all
   of them and thus do not fix all the compiler errors.
-  Run ``make`` generating a bunch of object (.o) files.
-  Link the executable (which also removes the .o files). Users may
   delete the .o files because they anticipate compiling the next time
   on a different machine that uses a different compiler.
-  Run the executable.
-  Detect some error condition and restart the cycle.

In addition, during this process the user often moves manually among
different directories for editing, compiling, and running.

Several easy ways to improve the cycle
--------------------------------------

-  ``emacs`` and ``vim`` have a feature to allow the user to compile
   using make and have the editor automatically jump to the line of
   source code where the compiler detects an error, even when not
   currently editing the erroneous file.
-  The etags feature of ``emacs`` or tags feature of ``vim`` enables one
   to search quickly through a group of user-defined source files
   (and/or PETSc source files) regardless of the directory in which they
   are located. `GNU Global <http://www.gnu.org/s/global>`__ is a richer
   alternative to etags.
-  Also, ``emacs`` and ``vim`` easily enable:

   -  editing files that reside in any directory and retaining one's
      place within each of them
   -  searching for files in any directory as one attempts to load them
      into the editor

You might consider using ``Microsoft Visual Studio``, ``Eclipse`` or
other advanced software development systems. See the :ref:`Users Manual<sec-developer-environments>`.

Debugging
---------

Most code development for PETSc codes should be done on one processor;
hence, using a standard debugger such as dbx, gdb, xdbx, etc. is fine.
For debugging parallel runs we suggest **Totalview** if it is available
on your machine. Otherwise, you can run each process in a separate
debugger; this is not the same as using a parallel debugger, but in most
cases it is not so bad. The PETSc run-time options
``-start_in_debugger`` [-display xdisplay:0] will open separate windows
and debuggers for each process. You should debug using the debugging
versions of the libraries (run ./configure with the additional option
--with-debugging (the default)).

It really pays to learn how to use a debugger; you will end up writing
more interesting and far more ambitious codes once it is easy for you to
track down problems in the codes.

Other suggestions
-----------------

-  Choose consistent and obvious names for variables and functions.
   (Short variable names may be faster to type, but by using longer
   names you don't have to remember what they represent since it is
   clear from the name.)
-  Use informative comments.
-  Leave space in the code to make it readable.
-  Line things up in the code for readability. Remember that any code
   written for an application will be visited **over and over** again,
   so spending an extra 20 percent of effort on it the first time will
   make each of those visits faster and more efficient.
-  Realize that you **will** have to debug your code. **No one** writes
   perfect code, so always write code that may be debugged and learn how
   to use a debugger. In most cases using the debugger to track down
   problems is much faster than using print statements.
-  **Never** hardwire a large problem size into your code. Instead,
   allow a command line option to run a small problem. We've seen people
   developing codes who have to wait 15 minutes after starting a run to
   reach the crashing point; this is an inefficient way of developing
   code.
-  Develop your code on the simplest machine to which you have access.
   We have accounts on a variety of large parallel machines, but we
   write and initially test all our code on laptops or workstations
   because the user interface is friendlier, and the turn-around time
   for compiling and running is much faster than for the parallel
   machines. We use the parallel machines only for large jobs. Since
   PETSc code is completely portable, switching to a parallel machine
   from our laptop/workstation development environment simply means
   logging into another machine -- there are no code or makefile
   changes.
-  Never develop code directly on a multi-node computing system; your
   productivity will be much lower than if you developed on a
   well-organized workstation.
-  Keep your machines' operating systems and compilers up-to-date (or
   force your systems people to do this :-). You should always work with
   whatever tools are currently the best available. It may seem that you
   are saving time by not spending the time upgrading your system, but,
   in fact, your loss in efficiency by sticking with an outdated system
   is probably larger than then the time required to keep it up-to-date.

Fortran notes
-------------

PETSc provides interfaces and modules for Fortran 90; see
:doc:`/docs/manual/fortran`.

When passing floating point numbers into Fortran subroutines, always
make sure you have them marked as double precision (e.g., pass in ``10.d0``
instead of ``10.0`` or declare them as PETSc variables, e.g.
``PetscScalar one = 1.0``). Otherwise, the compiler interprets the input as a single
precision number, which can cause crashes or other mysterious problems.
Make sure to declare all variables (do not use the implicit feature of
Fortran). In fact, we **highly** recommend using the **implicit none**
option at the beginning of each Fortran subroutine you write.
