#!/bin/bash
export BUILDDIR=$HOME/working/taobuild
export VERSION=2.0-p2
export REVISION=default
export HERE=$PWD
if [ ! -d "$BUILDDIR" ]
then
    mkdir -p $BUILDDIR
fi
cd $BUILDDIR

if [ -d "tao-$VERSION" ]
then
    rm -rf tao-$VERSION
fi

echo "Cloning /home/aotools/hg/tao-2.0 -r $REVISION"
hg clone /home/aotools/hg/tao-2.0 -r $REVISION tao-$VERSION

cd tao-$VERSION
export TAO_DIR=$BUILDDIR/tao-$VERSION

echo "Generating fortran stubs..."
make tao_allfortranstubs BFORT=bfort

echo "Generating etags..."
make tao_alletags

echo "Building manual..."
make tao_manual

echo "Building manpages..."
make tao_allmanpages

echo "Creating html from source files..."
make tao_htmlpages

echo "Building tarball $BUILDDIR/tao-$VERSION.tar.gz"
cd $BUILDDIR
tar czf tao-$VERSION.tar.gz --exclude-vcs --exclude="TODO" --exclude="oldtao" --exclude="nlstests" tao-$VERSION --exclude="sqpcon" --exclude="lm" --exclude="pounder" --exclude="taodm" --exclude="tex"

echo "mv tao-$VERSION.tar.gz $HERE"
mv tao-$VERSION.tar.gz $HERE


