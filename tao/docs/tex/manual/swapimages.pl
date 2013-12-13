#!/usr/local/bin/perl

use File::Copy;

$htmlfile = $ARGV[0];       #html with "file" references
$destdir  = $ARGV[1];       #destination directory for image files
if(defined($destdir)) {
    if(-d "$destdir") {  #check that it is a directory
	if(!(-w "$destdir")) #check that the user can write to it
	{
	    die "usage: swapimages.pl htmlfile [destination image dir] ($destdir not writable)\n";
	}
    } else {
	die "usage: swapimages.pl htmlfile [destination image dir] ($destdir must be created first)\n";
    }
}

%paths = ();             #make sure that the absolute paths are the same

open(HF, "$htmlfile") or die "usage: swapimages.pl htmlfile [destination image dir] ($htmlfile not valid)\n";

while(defined($line = <HF>)) {

    while($line =~ /SRC=\"file:(\S+)\"/) {
	$file    = "";
	$absfile = $1;
	$absfile =~ s/^\s+//g;    #don't allow spaces in front of names
	$absfile =~ s/\s+$//g;    #don't allow spaces after names

	@path = split /\/|\\/, "$absfile";
	$file = $path[$#path]; #take the last member as the file
	if(exists $paths{$file}) {
	    if($absfile ne $paths{$file}) {
		for($i = 0; $i < 10; $i++) {
		    if(exists $paths{"$i".$file}) {
			if($absfile eq $paths{"$i".$file}) {
			    $file = "$i".$file;
			    last;
			}
		    } else {
			$file = "$i".$file;
			$paths{$file} = $absfile;
			last;
		    }
		}
		if($i==10) {
		    die "Found 10 different paths to a file called $file\n";
		}
	    }
	} else {
	    $paths{$file} = $absfile;
	}
	$line =~ s/SRC=\"file:(\S+)\"/SRC=\"$file\"/;
    }
    print $line;
}

close HF;

@tomove = (keys %paths);
foreach $file (@tomove) {
    copy("$paths{$file}", "$destdir/$file") or die "Could not copy $paths{$file} to $destdir/$file\n";
}			      
	
	


