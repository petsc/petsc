PETSc Testing System
====================

The PETSc test system consists of

*     A language contained within the example source files that describes the tests to be run
*     The *test generator* (``config/gmakegentest.py``) that at the ``make`` step parses the example source files and generates the makefiles and shell scripts.
*    The *petsc test harness* that consists of makefile and shell scripts that runs the executables with several logging and reporting features.

Details on using the harness may be found in the :doc:`../manual/index`.

PETSc Test Description Language
-------------------------------

PETSc tests and tutorials contain within their file a simple language to
describe tests and subtests required to run executables associated with
compilation of that file. The general skeleton of the file is

::

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
* A single output file that represents the *expected esults*

Two or more command tests, usually, one or more mpiexec tests that run
the executable, and one or more diff tests to compare output with the
expected result.

Our language also supports a *testset* that specifies either a new test
entirely or multiple executable/diff tests within a single test. At the
core, the executable/diff test combination will look something like
this:

::

    mpiexec -n 1 ../ex1 1> ex1.tmp 2> ex1.err
    diff ex1.tmp output/ex1.out 1> diff-ex1.tmp 2> diff-ex1.err

In practice, we want to do various logging and counting by the test
harness; as are explained further below. The input language supports
simple yet flexible test control, and we begin by describing this
language.

Runtime Language Options
~~~~~~~~~~~~~~~~~~~~~~~~

At the end of each test file, a marked comment block, using YAML, is
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

   -  The tests are run under ``PETSC_ARCH/tests``, but some tests
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
      ``make -f gmakefile help``.

Additional Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the above keywords, other language features are
supported.

    -  for loops: Specifying ``!``\  ... shared output! or ``!``\  ...
       separate output! will create for loops over an enclosed
       space-delmited list. If the loop causes a different output, then
       separate output would be used. If the loop does not cause
       separate output, then the shared (in shorthand notation, ``!``\
       ... !) syntax must be used.

       For loops are supported within nsize and args. An example is

       ::

           args: -matload_block_size {{2 3}}

       In this case, two execution lines would be addded with two
       different arguments. Associated ``diff`` lines would be added as
       well automatically.

       Here the output for each ``matloadblocksize`` is assumed to give
       the same output so that only one diff file is needed. If the
       variables produced different output, then the ``separate output``
       option would be added. In this case, each loop variable and value
       become a separate script.

       See examples below for how it works in practice.

Test Block Examples
~~~~~~~~~~~~~~~~~~~

The following is the simplest test block:

::

    /*TEST test: TEST*/

which is equivalent to

::

    /*TEST testset: test: TEST*/

which is equivalent to

::

    /*TEST testset: TEST*/

If this block is in ``src/a/b/examples/tutorials/ex1.c``, then it will
create ``a_b_tutorials-ex1`` test that requires only one
processor/thread, with no arguments, and diff the resultant output with
``src/a/b/examples/tutorials/output/ex1.out``.

For Fortran, the equivalent is

.. code-block:: fortran

    !/*TEST ! test: !TEST*/

A more complete example is

::

    /*TEST
      test:
      test:
        suffix: 1
        nsize: 2
        args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres
        args: -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs
        args: -s2_pc_type jacobi -s2_ksp_monitor_short
        requires: x
    TEST*/

This creates two tests. Assuming that this is
``src/a/b/examples/tutorials/ex1.c``, the tests would be
``a_b_tutorials-ex1`` and ``a_b_tutorials-ex1_1``.

Following is an example of how to test a permutuation of arguments
against the same output file:

::

    /*TEST
      testset:
        suffix: 19
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/poisson1
        args: -ksp_type cg -pc_type icc -pc_factor_levels 2
        test:
        test:
          args: -mat_type seqsbaij
    TEST*/

Assuming that this is ``ex10.c``, there would be two mpiexec/diff
invocations in ``runex10_19.sh``.

Here is a similar example, but the permutation of arguments creates
different output:

::

    /*TEST
      testset:
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/medium
        args: -ksp_type bicg
        test:
          suffix: 4
          args: -pc_type lu
        test:
          suffix: 5
    TEST*/

Assuming that this is ``ex10.c``, two shell scripts will be created:
``runex10_4.sh`` and ``runex10_5.sh``.

An example using a for loop is:

::

    /*TEST
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
    TEST*/

In this example, ``ex138_2`` will invoke ``runex138_2.sh`` twice with
two different arguments, but both are diffed with the same file.

Following is an example showing the hierarchical nature of the test
specification.

::

    /*TEST
      testset:
        suffix:2
        output_file: output/ex138_1.out
        args: -f ${DATAFILESPATH}/matrices/small -mat_type baij
        test:
          args: -matload_block_size 2
        test:
          args: -matload_block_size 3
    TEST*/

This is functionally equivalent to the for loop shown above.

Here is a more complex example using for loops:

::

    /*TEST
      testset:
        suffix: 19
        requires: datafilespath
        args: -f0 ${DATAFILESPATH}/matrices/poisson1
        args: -ksp_type cg -pc_type icc
        args: -pc_factor_levels {{0 2 4}separate output}
        test:
        test:
          args: -mat_type seqsbaij
    TEST*/

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

::

    /*T build: requires: !complex T*/

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

The PETSc test system is designed to be compliant with the Test Anything
Protocal (TAP); see https://testanything.org/tap-specification.html

This is a simple standard designed to allow testing tools to work
together easily. There are libraries to enable the output to be used
easily, including sharness, which is used by the git team. However, the
simplicity of the PETSc tests and TAP specification means that we use
our own simple harness given by a single shell script that each file
sources: ``petsc_harness.sh``.

As an example, consider this test input:

::

    /*TEST
      test:
        suffix: 2
        output_file: output/ex138.out
        args: -f ${DATAFILESPATH}/matrices/small -mat_type {{aij baij sbaij}} -matload_block_size {{2 3}}
        requires: datafilespath
    */TEST

A sample output follows.

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

.. code-block:: bash

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
done by modifying ``${PETSC_DIR}/config/example_template.py``.

To modify the test harness, you can modify ``${PETSC_DIR}/config/petsc_harness.sh``.

Additional Tips
~~~~~~~~~~~~~~~

To rerun just the reporting use

::

    config/report_tests.py

To see the full options use

::

    config/report_tests.py -h

To see the full timing information for the five most expensive tests use

::

    config/report_tests.py -t 5
