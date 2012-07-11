#!/bin/bash
export BUILDDIR=$HOME/working/taobuild
export VERSION=2.1-p0
export REVISION=default
export HERE=$PWD
export C2HTML=c2html
if [ ! -d "$BUILDDIR" ]
then
    mkdir -p $BUILDDIR
fi
cd $BUILDDIR

if [ -d "tao-$VERSION" ]
then
    rm -rf tao-$VERSION
fi

echo "Cloning /home/aotools/hg/tao-2.1 -r $REVISION"
hg clone /home/aotools/hg/tao-2.1 -r $REVISION tao-$VERSION

cd tao-$VERSION
export TAO_DIR=$BUILDDIR/tao-$VERSION

echo "Generating fortran stubs..."
make tao_allfortranstubs BFORT=bfort

echo "Generating etags..."
make tao_alletags

echo "Building manual..."
make tao_manual
echo "Building manpages..."
make tao_allmanpages DOCTEXT=doctext MANPAGES=manpages C2HTML=c2html MAPNAMES=mapnames

#echo "Creating html from source files..."
#make tao_htmlpages DOCTEXT=doctext MANPAGES=manpages

echo "Building tarball $BUILDDIR/tao-$VERSION.tar.gz"
cd $BUILDDIR
tar czf tao-$VERSION.tar.gz --exclude-vcs --exclude="TODO" --exclude="oldtao" --exclude="nlstests" tao-$VERSION --exclude="sqpcon" --exclude="lm" --exclude="pounder" --exclude="taodm" --exclude="tex" --exclude="poundersb"

echo "mv tao-$VERSION.tar.gz $HERE"
mv tao-$VERSION.tar.gz $HERE


