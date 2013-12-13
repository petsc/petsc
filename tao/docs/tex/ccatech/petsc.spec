Summary: no summary
Name: petsc
Version: 2.1.5
Release: 2
URL: www.mcs.anl.gov/petsc
Packager: Jason Sarich <sarich@mcs.anl.gov>
Source0: %{name}-%{version}.tar.gz
Patch: petsc-2.1.5-redhat.patch
License: blank
Group: blank
BuildRoot: %{_tmppath}/%{name}-root
Prefix: %{_prefix}
Requires: lapack blas

%description
 PETSc is a suite of data structures and routines for the scalable 
(parallel) solution of scientific applications modeled by partial 
differential equations.  It employs the MPI standard for all 
message-passing communication.

%prep
%setup -q
%patch -p 1

%build
PETSC_ARCH=linux PETSC_DIR=$RPM_BUILD_DIR/petsc-2.1.5 BOPT=g_c++ make

%install
rm -rf $RPM_BUILD_ROOT
for i in `find -name "*.h" -printf "%h\n"`;
do 
   mkdir -p $RPM_BUILD_ROOT/usr/local/cca/petsc/$i; 
done
for i in `find -name "*.h"`;
do 
   cp $i $RPM_BUILD_ROOT/usr/local/cca/petsc/$i; 
done

mkdir -p $RPM_BUILD_ROOT/usr/local/cca/petsc/lib/libg_c++/linux
cp $RPM_BUILD_DIR/petsc-2.1.5/lib/libg_c++/linux/* $RPM_BUILD_ROOT/usr/local/cca/petsc/lib/libg_c++/linux
mkdir -p $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
mkdir -p $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/linux
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/common/base $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/common/test $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/common/rules $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/common/variables $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/common/bopt_g_c++ $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/common
cp $RPM_BUILD_DIR/petsc-2.1.5/bmake/linux/* $RPM_BUILD_ROOT/usr/local/cca/petsc/bmake/linux
cp -r $RPM_BUILD_DIR/petsc-2.1.5/docs $RPM_BUILD_ROOT/usr/local/cca/petsc

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
...snip
A complete list of all .h, .a, and .so files has been cut out here
...snip
/usr/local/cca/petsc/bmake/common/base
/usr/local/cca/petsc/bmake/common/bopt_g_c++
/usr/local/cca/petsc/bmake/common/rules
/usr/local/cca/petsc/bmake/common/variables
/usr/local/cca/petsc/bmake/common/test
/usr/local/cca/petsc/bmake/linux/packages
/usr/local/cca/petsc/bmake/linux/rules
/usr/local/cca/petsc/bmake/linux/variables
/usr/local/cca/petsc/docs/*

%changelog
* Thu Jan 30 2003 sarich <sarich@localhost.localdomain>
- Initial build.


