#!/bin/bash

dotfilename=include-graph.dot


if [ $# -eq 0 ] ; then
    echo 'Error: Output filename not provided.'
    exit 0
fi




#
# Section 1: Create dot file
#

echo "Creating graph file $dotfilename..."

echo "digraph G {" > $dotfilename
#echo " size=\"5,10\"" >> $dotfilename
#echo " ratio=\"0.3\"" >> $dotfilename
echo " rankdir=LR" >> $dotfilename
echo " node [color=red]" >> $dotfilename
#echo " size=\"8,6\"; ratio=fill; node[fontsize=24];" >> $dotfilename


#
### Part 1: Extract files from include/petsc/private/ and put them in subgraph ###
#

echo "  subgraph cluster_private {" >> $dotfilename
echo "    label = \"petsc/private\"; rank=\"10\"" >> $dotfilename

# Set labels
for f in `ls include/petsc/private/*.h`
do
  f2=${f#include/petsc/private/} 
  echo "    ${f2%.h} [label=\"$f2\",color=black];" >> $dotfilename
done

# Set connections
echo "    " >> $dotfilename
echo "    //Connections:" >> $dotfilename
for f in `ls include/petsc/private/*.h`
do
  f2=${f#include/petsc/private/} 
  lines=`grep "^#include *[<\"]petsc/private/" $f | sed "s,.*include *[<\"]\([^>\"]*\).*,\1," | sed "s,petsc/private/,,"`  #first sed command extracts anything between '<' and '>' or '"' and '"' after #include
  for line in `echo $lines`
  do
    line2=${line%.h}
    echo "    ${f2%.h} -> ${line2%.h\"} ;" >> $dotfilename
  done
done

echo "  }" >> $dotfilename




#
### Part 2: build dependencies for include/*.h: ###
#

# Set labels
for f in `ls include/*.h`
do
  f2=${f#include/}
  f3=${f2/%.hh/2}
  echo "  ${f3%.h} [label=\"$f2\",color=black];" >> $dotfilename
done

# Set connections
echo "  " >> $dotfilename
echo "  //Connections to petsc/private:" >> $dotfilename
for f in `ls include/*.h`
do
  f2=${f#include/}
  f3=${f2/%.hh/2}
  lines=`grep "^#include *[<\"]petsc/private/" $f | sed "s,.*include *[<\"]\([^>\"]*\).*,\1,"` | sed "s,petsc/private/,,"  #first sed command extracts anything between '<' and '>' or '"' and '"' after #include
  for line in `echo $lines`
  do
    line2=${line%.h}
    line3=${line2/%.hh>/2}
    echo "  ${f3%.h} -> ${line3%.h\"} ;" >> $dotfilename
  done
done

echo "  " >> $dotfilename
echo "  //Connections from petsc/private: (these might be problematic) " >> $dotfilename
for f in `ls include/petsc/private/*.h`
do
  f2=${f#include/petsc/private/}
  lines=`grep "^#include *[<\"]petsc" $f | grep -v "petsc/private" | sed "s,.*include *[<\"]\([^>\"]*\).*,\1,"`  #first sed command extracts anything between '<' and '>' or '"' and '"' after #include
  for line in `echo $lines`
  do
    line2=${line%.h}
    echo "  ${f2%.h} -> ${line2%.h\"} ;" >> $dotfilename
  done
done


echo "  " >> $dotfilename
echo "  //Connections within include/:" >> $dotfilename
for f in `ls include/*.h`
do
  f2=${f#include/}
  f3=${f2/%.hh/2}
  lines=`grep "^#include *[<\"]petsc" $f | grep -v "petsc/private" | sed "s,.*include *[<\"]\([^>\"]*\).*,\1,"`
  for line in `echo $lines`
  do
    line2=${line%.h}
    line3=${line2/%.hh>/2}
    echo "  ${f3%.h} -> ${line3%.h\"} ;" >> $dotfilename
  done
done



echo "}" >> $dotfilename



#
# Section 2: Create the graph
#

echo "Rendering output graph using GraphViz to file $1..."
dot -T${1##*.} $dotfilename -o $1
echo "Deleting temporary file $dotfilename"
rm $dotfilename
echo "DONE"

