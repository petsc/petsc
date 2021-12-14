PETSc Testing System
====================

The PETSc test system consists of

* A language contained within the example source files that describes the tests to be run
* The *test generator* (``config/gmakegentest.py``) that at the ``make`` step parses the example source files and generates the makefiles and shell scripts
* The *PETSc test harness* that consists of makefile and shell scripts that runs the executables with several logging and reporting features

Details on using the harness may be found in the :ref:`user's manual <sec_runningtests>`.


In the examples below, we make use of the following alias for a commonly-used command:

.. code-block:: console

   $ alias ptmake='make -f gmakefile.test'

where ``ptmake`` stands for "PETSc test make".

Getting help
------------

To find help for the test harness options and available targets, use

.. code-block:: console

   $ ptmake help

Determining the failed jobs of a given run
------------------------------------------

The running of the test harness will show which tests fail, but you may not have
logged the output or run without showing the full error.  The best way of
examining the errors is with this command:

.. code-block:: console

   $ $EDITOR $PETSC_DIR/$PETSC_ARCH/tests/test*err.log

This method can also be used for pipeline jobs. Failed jobs can have all of the
log files downloaded from the artifacts download tab on the right side:

.. figure:: /images/developers/test-artifacts.png
   :alt: Test Artifacts at Gitlab

   Test artifacts can be downloaded from gitlab.

To see the list of all tests that failed from the last run, you can also run this command:

.. code-block:: console

    $ ptmake print-test test-fail=1

To print it out in a column format:

.. code-block:: console

    $ ptmake print-test test-fail=1 | tr ' ' '\n' | sort

Once you know which tests failed, the question is how to debug them.

Introduction to debugging workflows
-----------------------------------

Here, two different workflows on developing with the test harness are presented,
and then the language for adding a new test is described.  Before describing the
workflow, we first discuss the output of the test harness and how it maps onto
makefile targets and shell scripts.

Consider this line from running the PETSc test system:

::

    TEST arch-ci-linux-uni-pkgs/tests/counts/vec_is_sf_tests-ex1_basic_1.counts

The string ``vec_is_sf_tests-ex1_basic_1`` gives the following information:

* The file generating the tests is found in ``$PETSC_DIR/src/vec/is/sf/tests/ex1.c``
* The makefile target for the *test* is ``vec_is_sf_tests-ex1_basic_1``
* The makefile target for the *executable* is ``$PETSC_ARCH/tests/vec/is/sf/tests/ex1``
* The shell script running the test is located at: ``$PETSC_DIR/$PETSC_ARCH/tests/vec/is/sf/tests/runex1_basic_1.sh``

Let's say that you want to debug a single test as part of development.  There
are two basic methods of doing this:  1)  use shell script directly in test
directory, or 2) use makefile from the top level directory.  We present both
workflows.   There are many permutations of this and a developer should always
find the method that makes them the most productive.

Debugging a PETSc test using shell scripts
------------------------------------------

First, suggest looking at the working directory and look at the options to the
scripts:

.. code-block:: console

      $ cd $PETSC_ARCH/tests/vec/is/sf/tests
      $ ./runex1_basic_1.sh -h
      Usage: ./runex1_basic_1.sh [options]

      OPTIONS
        -a <args> ......... Override default arguments
        -c ................ Cleanup (remove generated files)
        -C ................ Compile
        -d ................ Launch in debugger
        -e <args> ......... Add extra arguments to default
        -f ................ force attempt to run test that would otherwise be skipped
        -h ................ help: print this message
        -n <integer> ...... Override the number of processors to use
        -j ................ Pass -j to petscdiff (just use diff)
        -J <arg> .......... Pass -J to petscdiff (just use diff with arg)
        -m ................ Update results using petscdiff
        -M ................ Update alt files using petscdiff
        -o <arg> .......... Output format: 'interactive', 'err_only'
        -p ................ Print command:  Print first command and exit
        -t ................ Override the default timeout (default=60 sec)
        -U ................ run cUda-memcheck
        -V ................ run Valgrind
        -v ................ Verbose: Print commands


We will be using the ``-C``, ``-V``, and ``-p`` flags.

A basic workflow is something similar to:

.. code-block:: console

     $ <edit>
     $ runex1_basic_1.sh -C
     $ <edit>
     $ ...
     $ runex1_basic_1.sh -m  # If need to update results
     $ ...
     $ runex1_basic_1.sh -V  # Make sure valgrind clean
     $ cd $PETSC_DIR
     $ git commit -a

For loops sometimes can become onerous to run the whole test.
In this case, you can use the ``-p`` flag to print just the first
command.  It will print a command suitable for running from
``$PETSC_DIR``, but it is easy to modify for execution in the test
directory:

.. code-block:: console

     $ runex1_basic_1.sh -p

Debugging a single PETSc test using makefile
---------------------------------------------

First recall how to find help for the options:

.. code-block:: console

   $ ptmake help-test


To compile the test and run it:

.. code-block:: console

   $ ptmake test search=vec_is_sf_tests-ex1_basic_1

This can consist of your basic workflow.  However,
for the normal compile and edit, running the entire harness with search can be
cumbersome.  So first get the command:

.. code-block:: console

     $ ptmake vec_is_sf_tests-ex1_basic_1 PRINTONLY=1
     <copy command>
     <edit>
     $ ptmake $PETSC_ARCH/tests/vec/is/sf/tests/ex1
     $ /scratch/kruger/contrib/petsc-mpich-cxx/bin/mpiexec -n 1 arch-mpich-cxx-py3/tests/vec/is/sf/tests/ex1
     ...
     $ cd $PETSC_DIR
     $ git commit -a


Advanced searching
------------------

For forming a search, it is recommended to always use ``print-test`` instead of
``test`` to make sure it is returning the values that you want.

The three basic and recommended arguments are:

+ ``search`` (or ``s``)

  -  Searches based on name of test target (see above)
  -  Use the familiar glob syntax (like the Unix ``ls`` command). Example:

     .. code-block:: console

        $ ptmake print-test search='vec_is*ex1*basic*1'

     Equivalently:

     .. code-block:: console

        $ ptmake print-test s='vec_is*ex1*basic*1'

  -  It also takes full paths. Examples:

     .. code-block:: console

        $ ptmake print-test s='src/vec/is/tests/ex1.c'

     .. code-block:: console

        $ ptmake print-test s='src/dm/impls/plex/tests/'

     .. code-block:: console

        $ ptmake print-test s='src/dm/impls/plex/tests/ex1.c'


+ ``query`` and ``queryval`` (or ``q`` and ``qv``)

  -  ``query`` corresponds to test harness keyword, ``queryval`` to the value. Example:

     .. code-block:: console

        $ ptmake print-test query='suffix' queryval='basic_1'

  -  Invokes ``config/query_tests.py`` to query the tests (see
     ``config/query_tests.py --help`` for more information).
  -  See below for how to use as it has many features

+ ``searchin`` (or ``i``)

  -  Filters results of above searches. Example:

     .. code-block:: console

        $ ptmake print-test s='src/dm/impls/plex/tests/ex1.c' i='*refine_overlap_2d*'

Searching using gmake's native regexp functionality is kept for people who like it, but most developers will likely prefer the above methods:

+ ``gmakesearch``

  -  Use gmake's own filter capability.
  -  Fast, but requires knowing gmake regex syntax which uses ``%`` instead of ``*``
  -  Also very limited (cannot use two ``%``'s for example)
  -  Example:

     .. code-block:: console

        $ ptmake test gmakesearch='vec_is%ex1_basic_1'

+ ``gmakesearchin``

  -  Use gmake's own filter capability to search in previous results. Example:

     .. code-block:: console

        $ ptmake test gmakesearch='vec_is%1' gmakesearchin='basic'

+ ``argsearch``

  -  search on arguments using gmake.  This is deprecated in favor of the query/queryval method as described below. Example:

     .. code-block:: console

        $ ptmake test argsearch='sf_type'

  - Not very powerful

Query-based searching
~~~~~~~~~~~~~~~~~~~~~

Basic examples.  Note the the use of glob style matching is also accepted in the value field:

.. code-block:: console

   $ ptmake print-test query='suffix' queryval='basic_1'

.. code-block:: console

   $ ptmake print-test query='requires' queryval='cuda'

.. code-block:: console

   $ ptmake print-test query='requires' queryval='defined(PETSC_HAVE_MPI_GPU_AWARE)'

.. code-block:: console

   $ ptmake print-test query='requires' queryval='*GPU_AWARE*'

Using the ``name`` field is equivalent to the search above:

-  Example:

   .. code-block:: console

      $ ptmake print-test query='name' queryval='vec_is*ex1*basic*1'

-  Useful because this can be combined with union/intersect queries as discussed below

Arguments are tricky to search for.  Consider

.. code-block:: none

  args:  -ksp_monitor_short -pc_type ml -ksp_max_it 3

Search terms are

.. code-block:: none

    ksp_monitor, pc_type ml, ksp_max_it

Certain items are ignored:

+ Numbers (see ``ksp_max_it`` above), but floats are ignored as well.
+ Loops: ``args: -pc_fieldsplit_diag_use_amat {{0 1}}`` gives ``pc_fieldsplit_diag_use_amat`` as the search term
+ Input files: ``-f *``

Examples of argument searching:

.. code-block:: console

   $ ptmake print-test query='args' queryval='ksp_monitor'

.. code-block:: console

   $ ptmake print-test query='args' queryval='*monitor*'

.. code-block:: console

   $ ptmake print-test query='args' queryval='pc_type ml'


Multiple simultaneous queries can be performed with union (``,``), and intesection (``|``)  operators in the ``query`` field.  Examples:

-  All examples using ``cuda`` and all examples using ``hip``:

   .. code-block:: console

      $ ptmake print-test query='requires,requires' queryval='cuda,hip'

-  Examples that require both triangle and ctetgen (intersection of tests)

   .. code-block:: console

      $ ptmake print-test query='requires|requires' queryval='ctetgen,triangle'

-  Tests that require either ``ctetgen`` or ``triangle``

   .. code-block:: console

      $ ptmake print-test query='requires,requires' queryval='ctetgen,triangle'

-  Find ``cuda`` examples in the ``dm`` package.

   .. code-block:: console

      $ ptmake print-test query='requires|name' queryval='cuda,dm*'


Here is a way of getting a feel for how the union and intersect operators work:

.. code-block:: console

      $ ptmake print-test query='requires' queryval='ctetgen' | tr ' ' '\n' | wc -l
      170
      $ ptmake print-test query='requires' queryval='triangle' | tr ' ' '\n' | wc -l
      330
      $ ptmake print-test query='requires,requires' queryval='ctetgen,triangle' | tr ' ' '\n' | wc -l
      478
      $ ptmake print-test query='requires|requires' queryval='ctetgen,triangle' | tr ' ' '\n' | wc -l
      22

The total number of tests for running only ctetgen or triangle is 500.  They have 22 tests in common, and 478 that
run independently of each other.

The union and intersection have fixed grouping.  So this string argument

.. code-block:: none

    query='requires,requires|args' queryval='cuda,hip,*log*'

will can be read as

.. code-block:: none

   requires:cuda && (requires:hip || args:*log*)

which is probably not what is intended.


``query/queryval`` also support negation (``!``), but is limited.
The negation only applies to tests that have a related field in it.  So for
example, the arguments of

.. code-block:: none

  query=requires queryval='!cuda'

will only match if they explicitly have::

     requires: !cuda

It does not match all cases that do not require cuda.


Debugging for loops
--------------------

One of the more difficult issues is how to debug for loops when a subset of the
arguments are the ones that cause a code crash.  The default naming scheme is
not always helpful for figuring out the argument combination.

For example:

.. code-block:: console

      $ ptmake test s='src/ksp/ksp/tests/ex9.c' i='*1'
      Using MAKEFLAGS: i=*1 s=src/ksp/ksp/tests/ex9.c
              TEST arch-osx-pkgs-opt-new/tests/counts/ksp_ksp_tests-ex9_1.counts
       ok ksp_ksp_tests-ex9_1+pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_type-additive
       not ok diff-ksp_ksp_tests-ex9_1+pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_type-additive
       ok ksp_ksp_tests-ex9_1+pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_type-multiplicative
       ...


In this case, the trick is to use the verbose option, ``V=1`` (or for the shell script workflows, ``-v``) to have it show the commands:

.. code-block:: console

      $ ptmake test s='src/ksp/ksp/tests/ex9.c' i='*1' V=1
      Using MAKEFLAGS: V=1 i=*1 s=src/ksp/ksp/tests/ex9.c
      arch-osx-pkgs-opt-new/tests/ksp/ksp/tests/runex9_1.sh  -v
       ok ksp_ksp_tests-ex9_1+pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_diag_use_amat-0_pc_fieldsplit_type-additive # mpiexec  -n 1 ../ex9 -ksp_converged_reason -ksp_error_if_not_converged  -pc_fieldsplit_diag_use_amat 0 -pc_fieldsplit_diag_use_amat 0 -pc_fieldsplit_type additive > ex9_1.tmp 2> runex9_1.err
      ...

This can still be hard to read and pick out what you want.  So use the fact that you want ``not ok``
combined with the fact that ``#`` is the delimiter:

.. code-block:: console

      $ ptmake test s='src/ksp/ksp/tests/ex9.c' i='*1' v=1 | grep 'not ok' | cut -d# -f2
      mpiexec  -n 1 ../ex9 -ksp_converged_reason -ksp_error_if_not_converged  -pc_fieldsplit_diag_use_amat 0 -pc_fieldsplit_diag_use_amat 0 -pc_fieldsplit_type multiplicative > ex9_1.tmp 2> runex9_1.err



PETSc Test Description Language
-------------------------------

PETSc tests and tutorials contain within their file a simple language to
describe tests and subtests required to run executables associated with
compilation of that file. The general skeleton of the file is

.. code-block::

    static char help[] = "A simple MOAB example\n\

    ...
    <source code>
    ...

    /*TEST
       build:
         requires: moab
       testset:
         suffix: 1
         requires: !complex
       testset:
         suffix: 2
         args: -debug -fields v1,v2,v3
         test:
         test:
           args: -foo bar
    TEST*/

For our language, a *test* is associated with the following

* A single shell script
* A single makefile
* A single output file that represents the *expected results*

Two or more command tests, usually, one or more mpiexec tests that run
the executable, and one or more diff tests to compare output with the
expected result.

Our language also supports a *testset* that specifies either a new test
entirely or multiple executable/diff tests within a single test. At the
core, the executable/diff test combination will look something like
this:

.. code-block:: sh

    mpiexec -n 1 ../ex1 1> ex1.tmp 2> ex1.err
    diff ex1.tmp output/ex1.out 1> diff-ex1.tmp 2> diff-ex1.err

In practice, we want to do various logging and counting by the test
harness; as are explained further below. The input language supports
simple yet flexible test control, and we begin by describing this
language.

Runtime Language Options
~~~~~~~~~~~~~~~~~~~~~~~~

At the end of each test file, a marked comment block is
inserted to describe the test(s) to be run. The elements of the test are
done with a set of supported key words that sets up the test.

The goals of the language are to be

* as minimal as possible with the simplest test requiring only one keyword,
* independent of the filename such that a file can be renamed without rewriting the tests, and
* intuitive.

In order to enable the second goal, the *basestring* of the filename is
defined as the filename without the extension; for example, if the
filename is ``ex1.c``, then ``basestring=ex1``.

With this background, these keywords are as follows.

-  **testset** or **test**: (*Required*)

   -  At the top level either a single test or a test set must be
      specified. All other keywords are sub-entries of this keyword.

-  **suffix**: (*Optional*; *Default:* ``suffix=""``)

   -  The test name is given by ``testname = basestring`` if the suffix
      is set to an empty string, and by
      ``testname = basestring + "_" + suffix`` otherwise.

   -  This can be specified only for top level test nodes.

-  **output_file**: (*Optional*; *Default:*
   ``output_file = "output/" + testname + ".out"``)

   -  The output of the test is to be compared with an *expected result*
      whose name is given by ``output_file``.

   -  This file is described relative to the source directory of the
      source file and should be in the output subdirectory (for example,
      ``output/ex1.out``)

-  **nsize**: (*Optional*; *Default:* ``nsize=1``)

   -  This integer is passed to mpiexec; i.e., ``mpiexec -n nsize``

-  **args**: (*Optional*; *Default:* ``""``)

   -  These arguments are passed to the executable.

-  **diff_args**: (*Optional*; *Default:* ``""``)

   -  These arguments are passed to the ``lib/petsc/bin/petscdiff`` script that
      is used in the diff part of the test.  For example, ``-j`` enables testing
      the floating point numbers.

-  **TODO**: (*Optional*; *Default:* ``False``)

   -  Setting this Boolean to True will tell the test to appear in the
      test harness but report only TODO per the TAP standard.

   -  A runscript will be generated and can easily be modified by hand
      to run.

-  **filter**: (*Optional*; *Default:* ``""``)

   -  Sometimes only a subset of the output is meant to be tested
      against the expected result. If this keyword is used, it processes
      the executable output and puts it into the file to be actually
      compared with ``output_file``.

   -  The value of this is the command to be run, for example,
      ``grep foo`` or ``sort -nr``.

   -  If the filter begins with ``Error:``, then the test is assumed to
      be testing the ``stderr`` output, and the error code and output
      are set up to be tested.

-  **filter_output**: (*Optional*; *Default:* ``""``)

   -  Sometimes filtering the output file is useful for standardizing
      tests. For example, in order to handle the issues related to
      parallel output, both the output from the test example and the
      output file need to be sorted (since sort does not produce the
      same output on all machines). This works the same as filter to
      implement this feature

-  **localrunfiles**: (*Optional*; *Default:* ``""``)

   -  The tests are run under ``$PETSC_ARCH/tests``, but some tests
      require runtime files that are maintained in the source tree.
      Files in this (space-delimited) list will be copied over. If you
      list a directory instead of files, it will copy the entire
      directory (this is limited currently to a single directory)

   -  The copying is done by the test generator and not by creating
      makefile dependencies.

-  **requires**: (*Optional*; *Default:* ``""``)

   -  This is a space-delimited list of run requirements (not build
      requirements; see Build requirements below).

   -  In general, the language supports ``and`` and ``not`` constructs
      using ``! => not`` and ``, => and``.

   -  MPIUNI should work for all -n 1 examples so this need not be in
      the requirements list.

   -  Inputs sometimes require external matrices that are found in the
      DATAFILES path. For these tests ``requires: datafilespath`` can be
      specifed.

   -  Packages are indicated with lower-case specification, for example,
      ``requires: superlu_dist``.

   -  Any defined variable in petscconf.h can be specified with the
      ``defined(...)`` syntax, for example, ``defined(PETSC_USE_INFO)``.

   -  Any definition of the form ``PETSC_HAVE_FOO`` can just use
      ``requires: foo`` similar to how third-party packages are handled.

-  **timeoutfactor**: (*Optional*; *Default:* ``"1"``)

   -  This parameter allows you to extend the default timeout for an
      individual test such that the new timeout time is
      ``timeout=(default timeout) x (timeoutfactor)``.

   -  Tests are limited to a set time that is found at the top of
      ``"config/petsc_harness.sh"`` and can be overwritten by passing in
      the ``TIMEOUT`` argument to ``gmakefile`` (see
      ``ptmake help``.

Additional Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the above keywords, other language features are
supported.

-  **for loops**: Specifying ``{{list of values}}`` will generate a loop over
   an enclosed space-delimited list of values.
   It is supported within ``nsize`` and ``args``. For example,
   ::

       nsize: {{1 2 4}}
       args: -matload_block_size {{2 3}}

   Here the output for each ``-matload_block_size`` value is assumed to give
   the same output so that only one output file is needed.

   If the loop causes a different output, then separate output needs to be used:
   ::

       args: -matload_block_size {{2 3}separate output}

   In this case, each loop value generates a separate script,
   and a separate output file is needed.

   Note that ``{{...}shared output}`` is equivalent to ``{{...}}``.

   See examples below for how it works in practice.

Test Block Examples
~~~~~~~~~~~~~~~~~~~

The following is the simplest test block:

.. code-block:: yaml

    /*TEST
      test:
    TEST*/

If this block is in ``src/a/b/examples/tutorials/ex1.c``, then it will
create ``a_b_tutorials-ex1`` test that requires only one
processor/thread, with no arguments, and diff the resultant output with
``src/a/b/examples/tutorials/output/ex1.out``.

For Fortran, the equivalent is

.. code-block:: fortran

    !/*TEST
    !  test:
    !TEST*/

A more complete example showing just the part within the `/*TEST`:

.. code-block:: yaml

      test:
      test:
        suffix: 1
        nsize: 2
        args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres
        args: -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs
        args: -s2_pc_type jacobi -s2_ksp_monitor_short
        requires: x

This creates two tests. Assuming that this is
``src/a/b/examples/tutorials/ex1.c``, the tests would be
``a_b_tutorials-ex1`` and ``a_b_tutorials-ex1_1``.

Following is an example of how to test a permutuation of arguments
against the same output file:

.. code-block:: yaml

      testset:
        suffix: 19
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/poisson1
        args: -ksp_type cg -pc_type icc -pc_factor_levels 2
        test:
        test:
          args: -mat_type seqsbaij

Assuming that this is ``ex10.c``, there would be two mpiexec/diff
invocations in ``runex10_19.sh``.

Here is a similar example, but the permutation of arguments creates
different output:


.. code-block:: yaml

      testset:
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/medium
        args: -ksp_type bicg
        test:
          suffix: 4
          args: -pc_type lu
        test:
          suffix: 5

Assuming that this is ``ex10.c``, two shell scripts will be created:
``runex10_4.sh`` and ``runex10_5.sh``.

An example using a for loop is:

.. code-block:: yaml

      testset:
        suffix: 1
        args:   -f ${DATAFILESPATH}/matrices/small -mat_type aij
        requires: datafilespath
      testset:
        suffix: 2
        output_file: output/ex138_1.out
        args: -f ${DATAFILESPATH}/matrices/small
        args: -mat_type baij -matload_block_size {{2 3}shared output}
        requires: datafilespath

In this example, ``ex138_2`` will invoke ``runex138_2.sh`` twice with
two different arguments, but both are diffed with the same file.

Following is an example showing the hierarchical nature of the test
specification.

.. code-block:: yaml

      testset:
        suffix:2
        output_file: output/ex138_1.out
        args: -f ${DATAFILESPATH}/matrices/small -mat_type baij
        test:
          args: -matload_block_size 2
        test:
          args: -matload_block_size 3

This is functionally equivalent to the for loop shown above.

Here is a more complex example using for loops:

.. code-block:: yaml

      testset:
        suffix: 19
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/poisson1
        args: -ksp_type cg -pc_type icc
        args: -pc_factor_levels {{0 2 4}separate output}
        test:
        test:
          args: -mat_type seqsbaij

If this is in ``ex10.c``, then the shell scripts generated would be

* ``runex10_19_pc_factor_levels-0.sh``
* ``runex10_19_pc_factor_levels-2.sh``
* ``runex10_19_pc_factor_levels-4.sh``

Each shell script would invoke twice.

Build Language Options
~~~~~~~~~~~~~~~~~~~~~~

You can specify issues related to the compilation of the source file
with the ``build:`` block. The language is as follows.

-  **requires:** (*Optional*; *Default:* ``""``)

   -  Same as the runtime requirements (for example, can include
      ``requires: fftw``) but also requirements related to types:

      #. Precision types: ``single``, ``double``, ``quad``, ``int32``

      #. Scalar types: ``complex`` (and ``!complex``)

   -  In addition, ``TODO`` is available to allow you to skip the build
      of this file but still maintain it in the source tree.

-  **depends:** (*Optional*; *Default:* ``""``)

   -  List any dependencies required to compile the file

A typical example for compiling for only real numbers is

.. code-block::

    /*TEST
      build:
        requires: !complex
      test:
    TEST*/

PETSC Test Harness
------------------

The goals of the PETSc test harness are threefold.

1. Provide standard output used by other testing tools
2. Be as lightweight as possible and easily fit within the PETSc build chain
3. Provide information on all tests, even those that are not built or run because they do not meet the configuration requirements

Before understanding the test harness, you should first understand the
desired requirements for reporting and logging.

Testing the Parsing
~~~~~~~~~~~~~~~~~~~

After inserting the language into the file, you can test the parsing by
executing

A dictionary will be pretty-printed. From this dictionary printout, any
problems in the parsing are is usually obvious. This python file is used
by

in generating the test harness.

Test Output Standards: TAP
--------------------------

The PETSc test system is designed to be compliant with the `Test Anything Protocal (TAP) <https://testanything.org/tap-specification.html>`__.

This is a simple standard designed to allow testing tools to work
together easily. There are libraries to enable the output to be used
easily, including sharness, which is used by the git team. However, the
simplicity of the PETSc tests and TAP specification means that we use
our own simple harness given by a single shell script that each file
sources: ``$PETSC_DIR/config/petsc_harness.sh``.

As an example, consider this test input:

.. code-block:: yaml

      test:
        suffix: 2
        output_file: output/ex138.out
        args: -f ${DATAFILESPATH}/matrices/small -mat_type {{aij baij sbaij}} -matload_block_size {{2 3}}
        requires: datafilespath

A sample output from this would be:

::

    ok 1 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"
    ok 2 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"
    ok 3 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 3"
    ok 4 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 3"
    ok 5 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size 2"
    ok 6 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size 2"
    ...

    ok 11 In mat...tests: "./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type saij -matload_block_size 2"
    ok 12 In mat...tests: "Diff of ./ex138 -f ${DATAFILESPATH}/matrices/small -mat_type aij -matload_block_size 2"

Test Harness Implementation
---------------------------

Most of the requirements for being TAP-compliant lie in the shell
scripts, so we focus on that description.

A sample shell script is given the following.

.. code-block:: sh

    #!/bin/sh
    . petsc_harness.sh

    petsc_testrun ./ex1 ex1.tmp ex1.err
    petsc_testrun 'diff ex1.tmp output/ex1.out' diff-ex1.tmp diff-ex1.err

    petsc_testend

``petsc_harness.sh`` is a small shell script that provides the logging and reporting
functions ``petsc_testrun`` and ``petsc_testend``.

A small sample of the output from the test harness is as follows.

.. code-block:: none

    ok 1 ./ex1
    ok 2 diff ex1.tmp output/ex1.out
    not ok 4 ./ex2
    #	ex2: Error: cannot read file
    not ok 5 diff ex2.tmp output/ex2.out
    ok 7 ./ex3 -f /matrices/small -mat_type aij -matload_block_size 2
    ok 8 diff ex3.tmp output/ex3.out
    ok 9 ./ex3 -f /matrices/small -mat_type aij -matload_block_size 3
    ok 10 diff ex3.tmp output/ex3.out
    ok 11 ./ex3 -f /matrices/small -mat_type baij -matload_block_size 2
    ok 12 diff ex3.tmp output/ex3.out
    ok 13 ./ex3 -f /matrices/small -mat_type baij -matload_block_size 3
    ok 14 diff ex3.tmp output/ex3.out
    ok 15 ./ex3 -f /matrices/small -mat_type sbaij -matload_block_size 2
    ok 16 diff ex3.tmp output/ex3.out
    ok 17 ./ex3 -f /matrices/small -mat_type sbaij -matload_block_size 3
    ok 18 diff ex3.tmp output/ex3.out
    # FAILED   4 5
    # failed 2/16 tests; 87.500% ok

For developers, modifying the lines that get written to the file can be
done by modifying ``$PETSC_DIR/config/example_template.py``.

To modify the test harness, you can modify ``$PETSC_DIR/config/petsc_harness.sh``.

Additional Tips
~~~~~~~~~~~~~~~

To rerun just the reporting use

.. code-block:: console

    $ config/report_tests.py

To see the full options use

.. code-block:: console

    $ config/report_tests.py -h

To see the full timing information for the five most expensive tests use

.. code-block:: console

    $ config/report_tests.py -t 5
