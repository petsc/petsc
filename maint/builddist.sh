#!/bin/tcsh
set BUILDDIR = $HOME/working/taobuild
set VERSION=2.0-beta
set REVISION=default
set HERE=$PWD
if (! -d $BUILDDIR) then
    mkdir -p $BUILDDIR
endif
cd $BUILDDIR

if (-d tao-$VERSION) then
    rm -rf tao-$VERSION
endif

echo "Cloning ssh://login.mcs.anl.gov//home/sarich/hg/tao_c -r $REVISION"
hg clone ssh://login.mcs.anl.gov//home/sarich/hg/tao_c -r $REVISION tao-$VERSION

cd tao-$VERSION
export TAO_DIR=$BUILDDIR/tao-2.0-beta

echo "Generating fortran stubs..."
make tao_allfortranstubs BFORT=bfort

echo "Generating etags..."
make tao_alletags

echo "Building tarball $BUILDDIR/tao-$VERSION.tar.gz"
cd $BUILDDIR
tar czf tao-$VERSION.tar.gz --exclude-vcs --exclude="TODO" --exclude="oldtao" --exclude="nlstests" tao-$VERSION 
mv tao-$VERSION.tar.gz $HERE



