#!/usr/local/bin/perl
#
#linkmans.pl adds hyperlinks to manual pages within an html document.
#
#usage: linkmans.pl <citation file> <html file>
#

$citefile = $ARGV[0];
$htmlfile = $ARGV[1];

if(!defined($citefile) or !defined($htmlfile)) {
    print "\nUsage: linkmans.pl <citation file> <html file>\n\n";
}

%functions = ();
open(CIT, "$citefile") or die "Unable to open $citefile\n";
while(<CIT>) {
    ($name, $place) = split/\+\+\+\+man\+/; 
    if(defined($name) and defined($place)) {
	$name =~ s/^man:\+//;
	$name =~ s/\+\+(.*)//;
	$place =~ s/\s//g;
	$functions{$name} = $place;
    }
}
close CIT;
open(HTML, "$htmlfile") or die "Unable to open $htmlfile\n";
open(TMP, ">$htmlfile.linkmans") or die "Unable to open $htmlfile.linkmans for writing.\n";
$skip = 0;
while(<HTML>) {
    $line = $_;
    while(/(\w+)/) {
	$word = $1;
	if(defined($ref = $functions{$word})) {
	    if(/$word(.*)\/A/) {
		unless($1 =~ /<A/) {
		    $skip = 1;
		}
	    }
	    $line =~ s/$word/<a href="$ref">$word<\/a>/g unless($skip);
	}
	s/$word//g;
	$skip = 0;
    }
    print TMP $line;
}
close TMP;
close HTML;


