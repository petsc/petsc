#!/usr/local/bin/perl
# -*- Perl -*-
# CONFIG: Change the first line to indicate the Perl binary on your system.
#          Try "whereis perl" or "find / -name perl -print"

###############################################################################
##                                                                           ##
##     AccessWatch v1.32 - Access accounting for World Wide Web sites        ##
##      Copyright (C) 1994-1996 by David G. Maher. All rights reserved.      ##
##      <URL:http://www.eg.bucknell.edu/~dmaher/accesswatch/>                ##
##                                                                           ##
##     Contact Information:                                                  ##
##        Dave Maher <dmaher@bucknell.edu>                                   ##
##        C-0500 Bucknell University, Lewisburg, PA 17837 USA                ##
##        <URL:http://www.eg.bucknell.edu/~dmaher/>                          ##
##                                                                           ##
##     Please read the included license agreement (readme.txt) before        ##
##      using this program. Your use of this software indicates your         ##
##      acceptance of the license agreement and warranty.                    ##
##                                                                           ##
##     If you wish to modify this code, mail <dmaher@bucknell.edu>.          ##
##                                                                           ##
###############################################################################

###############################################################################
########## Credits and thanks for great ideas #################################
###############################################################################
##                                                                           ##
## A huge thank you to Paul Blackman <ictinus@lake.canberra.edu.au> for      ##
##   his work on hourly server statistics, including a table of hourly       ##
##   accesses/hour.                                                          ##
##                                                                           ##
## Thanks to Jeff Boulter <boulter@bucknell.edu> for access error exclusion  ##
##   code, and for his many suggestions and debugging help.                  ##
##                                                                           ##
## Many thanks for suggestions and code tweaks:                              ##
##      Chris Brown <cwb3@ra.msstate.edu>                                    ##
##      Ron Gery    <rong@halcyon.com>                                       ##
##                                                                           ##
###############################################################################

# *** File configuration ***

($base=$0) =~ s/[A-z0-9,\.,\-]*$//;  # if you have trouble, define $base as
                                     #  the full directory pathname that 
                                     #  contains the accesswatch script.

require $base.'accesswatch.cfg' || die "Configuration file not found: $!\n"; 

foreach (@ARGV) {
    $verbose = 1 
	if (/v/);   # turn on -v command line switch
    $verbose = 0    
	if (/q/);   # turn on -q command line switch
}

$summarylink = "index.html";                 # Summary information link
$detailslink = "details.html";               # Access details link

$summaryfile = $base.$summarylink;           # Summary Information file
$detailsfile = $base.$detailslink;           # Access details file

$domaincodes = $base.'lib/domain.desc';      # Location of domaincodes database
$pagedescriptions = $base.'lib/page.desc';   # Text description of urls

# *** Graphics files ***

local(%horizbar) = ( 0, "img/blueblock.gif",
		     1, "img/redblock.gif"
		   );

local(%vertbar) = ( -1,  "img/clearvert.gif",
		     0,  "img/brwnvert.gif",
		     1,  "img/ltgnvert.gif",
		     2,  "img/pinkvert.gif",
		     3,  "img/cyanvert.gif",
		     4,  "img/orgvert.gif",
		     5,  "img/purpvert.gif",
		     6,  "img/yellvert.gif",
		     7,  "img/grnvert.gif",
		     8,  "img/bluevert.gif",
		     9,  "img/redvert.gif"
		  );

# *** Other options ***

$truncate =  1;   # Truncate extra path info for page demand list if too long.
$toolong  = 50;   # If you want to truncate, how *many* characters is too long?

###############################################################################
########     NO MODIFICATIONS MAY BE MADE TO THE FOLLOWING CODE        ########
########            WITHOUT THE CONSENT OF THE AUTHOR                  ########
###############################################################################

# *** Data structures and counters ***

local(@accesses);		# contains entire list of page accesses

local(%domains, %pages, %hosts);# contain list of domain extensions, pages,
                                #  and unique hosts, and associates each
                                #  with the appropriate count.

local(%pageDesc, %domainDesc);  # lists of page urls/domaincodes with 
                                # corresponding descriptions

local(%stat) = (
	 'accesses',       '0', # count of pages served
	 'serverCount',    '0', # server hits
	 'hits',           '0', # total subdirectory hits
	 'localCount',     '0', # number of accesses from local machines
	 'errors',         '0', # number of answers > 400 from server
	 'redirect',       '0', # number of redirects  
	 'size',           '0', # number of bytes transmitted

	 'uniqueHosts',    '0', # computed by UpdateStatArray
	 'hostPageAverage','0', # computed by UpdateStatArray
	 'serverLoad',     '0', # computed by UpdateStatArray
	 'localPercent',   '0', # computed by UpdateStatArray
	 'outsidePercent', '0', # computed by UpdateStatArray
	 'outsideCount',   '0', # computed by UpdateStatArray
	 'accessesPerHour','0', # computed by UpdateStatArray
	 'accessesPerDay', '0', # computed by UpdateStatArray
	 
         'hr00',           '-1', # accesses in hour 0
	 'hr01',           '-1', # accesses in hour 1
	 'hr02',           '-1', # accesses in hour 2
	 'hr03',           '-1', # accesses in hour 3
	 'hr04',           '-1', # accesses in hour 4
	 'hr05',           '-1', # accesses in hour 5
	 'hr06',           '-1', # accesses in hour 6
	 'hr07',           '-1', # accesses in hour 7
	 'hr08',           '-1', # accesses in hour 8
	 'hr09',           '-1', # accesses in hour 9
	 'hr10',	   '-1', # accesses in hour 10
	 'hr11',	   '-1', # accesses in hour 11
	 'hr12',	   '-1', # accesses in hour 12
	 'hr13',	   '-1', # accesses in hour 13
	 'hr14',	   '-1', # accesses in hour 14
	 'hr15',	   '-1', # accesses in hour 15
	 'hr16',	   '-1', # accesses in hour 16
	 'hr17',	   '-1', # accesses in hour 17
	 'hr18',	   '-1', # accesses in hour 18
	 'hr19',	   '-1', # accesses in hour 19
	 'hr20',	   '-1', # accesses in hour 20
	 'hr21',	   '-1', # accesses in hour 21
	 'hr22',	   '-1', # accesses in hour 22
	 'hr23',	   '-1', # accesses in hour 23
	 'maxhouraccess',  '0', # maximum accesses per hour
	 'minhouraccess',  '0', # minimum accesses per hour
	 );

local($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst);
local($startTime);
local(@mnths, @longmonths);

$version = "1.32"; # Do not touch this line, even if you hack at the code!

##############################################################################

&Main;

#-----------------------------------------------------------------------------#
#  AccessWatch function - Main
#    Purpose  : Coordinates everything else...
#-----------------------------------------------------------------------------#
sub Main {
    $startTime = time;
    $currdate = &SetDateInfo; # returns Day/Mon/Year

    print "AccessWatch v$version starting.\n" if $verbose;
    &ProcessLog;
    &PrepareSummaryPage;
    &PrepareHostDetailsPage;
    print "Finished - Output is in $summarylink.\n" if $verbose;
    exit 0;
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - ProcessLog
#    Purpose  : Scans through access log and picks out the appropriate accesses
#-----------------------------------------------------------------------------#
sub ProcessLog {
    
    $| = 1; print "Parsing access log..." if $verbose; $| = 0;
    
    local($remote, $dash1, $dash2, $date, $tz, 
	  $method, $page, $protocol, $protnum1, $size);

    open (LOG, "<$accessLog") || die "Couldn't open $accessLog\n";
    
    &FastSearch;      	# position pointer at start of day using a fast search

    while (<LOG>) {		
      # start at the current date and push all access relevant info into lists
	if ($verbose && $stat{'serverCount'} % 50 == 1) {
	    $| = 1;
	    print ".";
	    $| = 0;
	}
	if (/$currdate/i) {
	    chomp;
	    # Un-Webify plus signs and %-encoding
	    tr/+/ /;
	    tr/ //s;
	    s/%([a-fA-F0-9][a-fA-F0-9])/pack("C", hex($1))/eg;
	    $stat{'serverCount'}++;

	    if (/$includeURL/i && (! /$excludeURL/i || $excludeURL eq "")) {
		($remote, $dash1, $dash2, $date, $tz, 
		 $method, $page, $protocol, $protnum1, $size)
		    = split(/ /);
		$page =~ s/(.*)\?.*/$1/;
		$date =~ s/\[//;
		local($date, $hour, $min, $sec) = split(":", $date);


		$stat{'hits'}++;
		$stat{'size'} += $size/1000;

		if ($protnum1 < 400 && !($protnum1 eq "302")) {
		    &RecordStats($hour, $min, $sec, $remote, $page);
		}
		else { 
		    if ($protnum1 eq "302") { $stat{'redirect'}++; }
		    else { $stat{'errors'}++; }
		}
	    }				
	    elsif (/$includeURL/i) {
		$stat{'hits'}++;
		($remote, $dash1, $dash2, $date, $tz, 
		 $method, $page, $protocol, $protnum1, $size)
		    = split(/ /);
		$stat{'size'} += $size/1000 if ($size =~ /^\d.*/);
	    }
	}
    }				
    close (LOG);
    print " done.\n" if $verbose;
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - SetDateInfo
#    Purpose  : Sets global date variables, returns Day/Month/Year as specified
#                by log standards.
#-----------------------------------------------------------------------------#
sub SetDateInfo {
    #get current date and return as string (day/month/year)...
    @mnths = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
	      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec');
    @longmonths = ('January', 'February', 'March', 'April', 'May', 
		   'June', 'July', 'August', 'September', 'October', 
		   'November', 'December');
    ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
    $year = "19$year";
    $mday = "0$mday" if ($mday < 10);

    return "$mday/$mnths[$mon]/$year";
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintSummaryPage
#    Purpose  : Coordinates construction of summary interface.
#-----------------------------------------------------------------------------#
sub PrintSummaryPage {

    &UpdateStatArray;

    if ($stat{'accesses'} eq "0") { 
	print OUT <<EOM;
<CENTER><H2>No accesses have been made to $siteName.</H2></CENTER>
EOM
	return; 
    }

    &PrintStatsParagraph;

    print OUT <<EOM;
<CENTER>
EOM
    &PrintTableSummaryStats;
    &PrintTableHourlyStats;  # Added with much help from Paul Blackman
    &PrintTablePageStats;
    &PrintTableDomainStats;
    &PrintTableHostStats;

print OUT <<EOM;
</CENTER>
<P>
EOM

}

#-----------------------------------------------------------------------------#
#  AccessWatch function - FastSearch
#    Purpose  : Scans quickly from the bottom of the access log to start
#                   of the current day.
#-----------------------------------------------------------------------------#
sub FastSearch {
    # parses through access log from the bottom up for speed, searching
    #  for the start of the current day...
    local($avgday) = 10000 * $hour + 1000;
    local($logsize) = (-s LOG);
    local($logoffset) = 0;
    local($jumps) = 0;

    $logoffset = $logsize - $avgday if ($logsize > $avgday);

    seek (LOG, $logoffset, 0);
    <LOG>;
    $_=<LOG>;
    while (/$currdate/i && $logoffset > 0 && <LOG>) {
	$jumps++;
#	print "$_\n";
	$logoffset = $logoffset - 5000;
	$logoffset = 0 if ($logoffset <= 0);
	seek (LOG, $logoffset, 0);
	<LOG>;
	$_=<LOG>;
    }
#    print "***Jumps = $jumps***\n";
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrepareSummaryPage
#    Purpose  : Opens summary file, calls functions to write data, closes, and 
#                   sets the appropriate permissions.
#-----------------------------------------------------------------------------#
sub PrepareSummaryPage {
    $| = 1; print "Preparing Summary... " if $verbose; $| = 0;
    open (OUT, ">$summaryfile") || die "Couldn't open $summaryfile\n";

    &PrintHeader;
    &PrintSummaryPage;
    &PrintFooter;
    close (OUT);
    chmod (0644, $summaryfile);
    print "done.\n" if $verbose;
}    

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrepareHostDetailsPage
#    Purpose  : Opens host details file, calls functions to write data, closes,
#                   and sets the appropriate permissions.
#-----------------------------------------------------------------------------#
sub PrepareHostDetailsPage {
    if ($details) {
	open (OUT, ">$detailsfile") || die "No $detailsfile\n";
	&PrintHeader;
	&PrintHostDetailsList;
	&PrintFooter;
	close (OUT);
	chmod 0644, $detailsfile;
    }
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - RecordStats
#    Purpose  : Takes a single access as input, and updates the appropriate
#                   counters and arrays.
#-----------------------------------------------------------------------------#
sub RecordStats {
    #tally server information, such as domain extensions, total accesses,
    # and page information
    
    local($hour, $minute, $second, $remote, $page) = @_;
    if ($remote !~ /\./) { $remote .= ".$orgdomain"; }
      #takes care of those internal accesses that do not get fully 
      # qualified in the log name -> name.orgname.ext
    local($domainExt) = &GetDomainExtension($remote, 1);

    $stat{'accesses'}++;	
    $domains{$domainExt}++;
    $hosts{$remote}++;
    $pages{$page}++;
    $stat{"hr".$hour}++;

    push (@accesses, "$hour $min $sec $remote $page");

}

#-----------------------------------------------------------------------------#
#  AccessWatch function - GetDomainExtension
#    Purpose  : Takes a hostname as input, and returns the domain suffix. If
#                   second argument is set to true, then it counts accesses
#                   from the local domain.
#-----------------------------------------------------------------------------#
sub GetDomainExtension {
    local($domainExt) = $_[0];
    local($shouldCount) = $_[1];
    
    $stat{'localCount'}++ if ($domainExt =~ /.*$orgdomain/ && $shouldCount);
    $domainExt =~ s/.*\.//g;
    $domainExt = "ip address" if ($domainExt =~ /[0-9].*/);
    $domainExt =~ tr/[A-Z]/[a-z]/;

    return $domainExt;

}

#-----------------------------------------------------------------------------#
#  AccessWatch function - DescribeDomain
#    Purpose  : Takes a domain extension as a parameter, and returns a detailed
#                   description as specified in the domain descriptions file 
#                   provided. Caches file into memory if used more than once.
#-----------------------------------------------------------------------------#
sub DescribeDomain {

    local($description) = "Unknown"; # Default domain description
    local($domain) = $_[0];          # Domain extension (passed parameter)

    if ($domain ne "ip address") {
	if (!(keys %domainDesc)) {
	    open (CODES, "<$domaincodes") ||
		die "Couldn't open $domaincodes\n";
	    while (<CODES>) {
		    chomp;
		    chop;
		    ($code, $description) = split('   ');
		    $code =~ tr/[A-Z]/[a-z]/;
		    $domainDesc{$code} = $description;
	    }	
	    close(CODES);
	}
	if ($domainDesc{$domain}) {
	    $description = $domainDesc{$domain}; 
	}
    }
    return $description;
}				

#-----------------------------------------------------------------------------#
#  AccessWatch function - DescribePage
#    Purpose  : Takes a virtual URL as a parameter, and returns a detailed
#                   description as specified in the page descriptions file 
#                   provided. Caches file into memory if used more than once.
#-----------------------------------------------------------------------------#
sub DescribePage {

    local($url) = $_[0];
    local($description) = $url;

    if (!(keys %pageDesc)) {
	open (DESC, "<$pagedescriptions") || 
	    die "Couldn't open $pagedescriptions...";
	while (<DESC>) { 
	    chomp;
	    /(\S+)\s+\"(.*)\"/;
	    $pageDesc{$1} = $2;
	}
	close(DESC);
    }
    if ($pageDesc{$url}) {
	$description = $pageDesc{$url};
    }

    if ($truncate && length($url) > $toolong && $description eq $url) {
        $description = reverse $url;
	while (length($description) > $toolong) { chop $description; }
	$description = "..." . reverse $description;
    }
    return $description;
}				

#-----------------------------------------------------------------------------#
#  AccessWatch sort routine - byDomain
#    Purpose  : Sort subroutine for sorting an array by domain suffix.
#-----------------------------------------------------------------------------#
sub byDomain { 
    local($aHost) = "";
    local($bHost) = "";
    local(@aTemp, @bTemp) = ();
    if ($a =~ /[^0-9].*\.[^0-9].*/) { 
	@aTemp = reverse split(/\./, $a); 
	foreach (@aTemp) { $aHost .= $_ };
    }
    else {
	$aHost = "zzzzzzz".$a;
    }
    if ($b =~ /[^0-9].*\.[^0-9].*/) { 
	@bTemp = reverse split(/\./, $b);
	foreach (@bTemp) { $bHost .= $_ };
    }
    else {
	$bHost = "zzzzzzz".$b;
    }
    return ($aHost cmp $bHost);
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintHostDetailsList
#    Purpose  : Prints list of all accesses on server - date and page, sorted
#                   remote host.
#-----------------------------------------------------------------------------#
sub PrintHostDetailsList {

    local(%hostlist) = ();
    local($bdelim) = "s$;";    
    local($edelim) = "e$;";
    local($pdelim) = "p$;";

    print OUT <<EOM;
<H1>Access Details</H1>
<DL>
EOM

    foreach $entry (@accesses) { 
	local($hour, $minute, $second, $remote, $page) = split(" ", $entry);
	local($timestring) = $hour.":".$minute.":".$second;
	if ($hostlist{$remote}) {
	    $hostlist{$remote} = 
		$hostlist{$remote}.$bdelim.$timestring.$pdelim.$page.$edelim;
	}
	else {
	    $hostlist{$remote} = 
		$bdelim.$timestring.$pdelim.$page.$edelim;
	}
    }
    
    local(@hosts) = sort byDomain (keys %hostlist);

    local($domainExt) = "";    
    local($prevdomainExt) = "###";    
    local($domainDesc) = "";    

    foreach $host (@hosts) {
	$domainExt = &GetDomainExtension($host, 0);
	if ($domainExt ne $prevdomainExt) {
	    $domainDesc = &DescribeDomain($domainExt);
	    $prevdomainExt = $domainExt;
	    print OUT "</DL>\n<H3>$domainDesc<HR></H3>\n<DL>\n"
	}
	local($entry) = $hostlist{$host};
	$entry =~ s/$bdelim/<DD>/g;
	$entry =~ s/$pdelim/ /g;
	$entry =~ s/$edelim/\n/g;
	
	print OUT "<DT><B>$host</B>\n";
	print OUT "$entry";
    }		       
    print OUT <<EOM;
</DL>
<P>
EOM

}	

sub byReverseNumber { $b <=> $a; }

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintStatsParagraph
#    Purpose  : Prints nice summary of crucial information
#-----------------------------------------------------------------------------#
sub PrintStatsParagraph {
    #prints summary of gathered access statistics, with graphical 
    # representation of percentage of hits according to domain

    print OUT "<H2>Daily Access Statistics";
    print OUT "<HR SIZE=3 WIDTH=40% ALIGN=LEFT></H2>\n";

    print OUT <<EOM;
Today, there have been a total of <B>$stat{'accesses'}</B>
accesses by <B>$stat{'uniqueHosts'}</B> unique hosts viewing an average of <B>$stat{'hostPageAverage'}</B> pages related to <B><I>$siteName.</I></B>
EOM

    print OUT "Of these, ";
    printf OUT ("<B>$stat{'localCount'} (%.3g%%)</B>", $stat{'localPercent'});
    print OUT " have been from $orgname, and ";
    printf OUT ("<B>%d (%.3g%%)</B>", $stat{'outsideCount'}, 
		$stat{'outsidePercent'});

print OUT <<EOM;
 have been from outside hosts.<BR>
There have been a total of <B>$stat{'hits'}</B> hits and <B>$stat{'errors'}</B> errors related 
to <B><I>$siteName</I></B>, accounting for 
EOM
    printf OUT ("<B>%.3g%%</B> of total server hits and consisting of ", $stat{'serverLoad'});

    printf OUT ("<B>%d</B> kilobytes of information. ", $stat{'size'});

    printf OUT ("There have been <B>%3.1f</B>", $stat{'accessesPerHour'});
    print OUT " accesses per hour, and at this rate, <B><I>$siteName</I></B> will get ";
    printf OUT ("<B>%d</B>", $stat{'accessesPerDay'});
    print OUT " accesses today.<P>\n";


}

#-----------------------------------------------------------------------------#
#  AccessWatch function - UpdateStatArray
#    Purpose  : Computes statistics based on accumulated counters.
#-----------------------------------------------------------------------------#
sub UpdateStatArray {    

    # variable descriptions and computations
     # $stat{'count'} = absolute number of accesses
     # $stat{'hits'} = number of page related hits on server
     # $stat{'servercount'} =total  number of hits on server
     # $siteName = description of page
     # $serverload - % of server hits related to page
     # $stat{'localcount'} - number of accesses from within organization
     # $outside - number of accesses from outside hosts
     # $pctlocal - % of accesses from within organization

    $stat{'serverLoad'} = 0;
    $stat{'localPercent'} = 0;
    $stat{'outsidePercent'} = 0;
    $stat{'outsideCount'} = $stat{'accesses'} - $stat{'localCount'};
    $stat{'accessesPerHour'} = 0;

    $stat{'serverLoad'} =  $stat{'hits'} / $stat{'serverCount'} * 100 
	if ($stat{'serverCount'} > 0);
    $stat{'localPercent'} = $stat{'localCount'} / $stat{'accesses'} * 100 
	if ($stat{'accesses'} > 0);
    $stat{'outsidePercent'} = $stat{'outsideCount'} / $stat{'accesses'} * 100 
	if ($stat{'accesses'} > 0);
    $stat{'accessesPerHour'} = ($stat{'accesses'} / (($hour * 60) + $min)) * 60
	if ($hour * 60 + $min > 0);

    $stat{'accessesPerDay'} = $stat{'accessesPerHour'} * 24;
    $stat{'uniqueHosts'} = keys %hosts;
    
    $stat{'hostPageAverage'} = 
	sprintf("%3.1f", $stat{'accesses'}/$stat{'uniqueHosts'})
	    if ($stat{'uniqueHosts'} != 0);

    $stat{'accessesPerHour'} = sprintf("%3.1f", $stat{'accessesPerHour'});
    $stat{'accessesPerDay'} = sprintf("%d", $stat{'accessesPerDay'});

    $stat{'maxhouraccess'} = 0;
    $stat{'minhouraccess'} = 0;

    foreach $hournum ('00'..'23') {
	$stat{'maxhouraccess'} = $stat{"hr".$hournum}
	     if ($stat{"hr".$hournum} > $stat{'maxhouraccess'});	    
	$stat{'minhouraccess'} = $stat{"hr".$hournum}
	     if (($stat{'minhouraccess'} == 0 ||
		  $stat{"hr".$hournum} < $stat{'minhouraccess'}) && 
		 $stat{"hr".$hournum} != -1);	    
    }
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTableDomainStats
#    Purpose  : Prints table of domains, sorted by number of accesses.
#-----------------------------------------------------------------------------#
sub PrintTableDomainStats {

    return if ($maxDomainsToList == 0);

    print OUT <<EOM;
<TABLE BORDER=1 WIDTH=100%>
<TR><TH COLSPAN=5><HR SIZE=5>Accesses by Domain
<HR SIZE=5></TH></TR>
<TR><TH>Domain </TH><TH>Description </TH>
<TH>Count </TH><TH>% of total</TH></TR>
EOM

    local(@countOrderList) = sort byReverseNumber values %domains;
    local(@domainOrder) = ();

    local($domainCount) = $maxDomainsToList - 1;
    $domainCount = $#countOrderList 
	if ($maxDomainsToList == -1 || 
	    ($maxDomainsToList - 1) > $#countOrderList);

    while (@countOrderList) {
	while (local($tag, $value) = (each %domains)) {
	    if ($countOrderList[0] == $value) {
		push (@domainOrder, $tag);
		shift @countOrderList;
	    }
	}
    }

    local(@favDomains) = @domainOrder[0..$domainCount];
    local(@othDomains) = @domainOrder[@favDomains..$#domainOrder];

    foreach (@favDomains) {
	local($Mdomain) = $_;
	local($Mcount) = $domains{$_};

	$Mdomain =~ tr/[A-Z]/[a-z]/;


	print OUT "<TR><TD><DT>$Mdomain</A> </TD>";
	local($description) = &DescribeDomain($Mdomain);
	print OUT "<TD>$description </TD>";
	print OUT "<TD ALIGN=RIGHT>$Mcount </TD>";

	$pct = 0.00;
	$pct = $Mcount / $stat{'accesses'} * 100 if ($stat{'accesses'} > 0);

	printf OUT ("<TD ALIGN=RIGHT> %3.2f</TD>\n", $pct);
        print OUT "<TD ALIGN=LEFT>";
	&PrintBarHoriz($pct,0);
        print OUT "</TD></TR>\n";
    }

    print OUT "<TR><TD COLSPAN=5><B>Also, hosts from the following domains visited:</B> " if (@othDomains);

    foreach (@othDomains) {
	tr/[A-Z]/[a-z]/;
	local($description) = &DescribeDomain($_);
	if (@othDomains) {
	    print OUT "$description ($domains{$_}), ";
	}
	else {
	    print OUT "and $description ($domains{$_}).";
	}
    }
    print OUT "</TD></TR>\n</TABLE><P>\n";
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTableHostStats
#    Purpose  : Prints table of hosts, sorted by number of accesses.
#-----------------------------------------------------------------------------#
sub PrintTableHostStats {

    return if ($maxHostsToList == 0);

    print OUT <<EOM;
<TABLE BORDER=1 WIDTH=100%>
<TR><TH COLSPAN=4><HR SIZE=5>Most Frequent Accesses by Host<HR SIZE=5></TH></TR>
<TR><TH>Host </TH><TH>Count </TH><TH>% of total</TH></TR>
EOM
    
    local(@countOrderList) = sort byReverseNumber values %hosts;
    local(@hostOrder) = ();

    local($hostCount) = $maxHostsToList - 1;
    $hostCount = $#countOrderList 
	if ($maxHostsToList == -1 || 
	    ($maxHostsToList - 1) > $#countOrderList);

    while (@countOrderList) {
	while (local($tag, $value) = (each %hosts)) {
	    if ($countOrderList[0] == $value) {
		push (@hostOrder, $tag);
		shift @countOrderList;
	    }
	}
    }

    local(@favHosts) = @hostOrder[0..$hostCount];

    foreach (@favHosts) {
	local($Mhost) = $_;
	local($Mcount) = $hosts{$_};

	print OUT "<TR><TD><DT>$Mhost</A> </TD>";
	print OUT "<TD ALIGN=RIGHT>$Mcount </TD>";

	$pct = 0.00;
	$pct = $Mcount / $stat{'accesses'} * 100 if ($stat{'accesses'} > 0);

	printf OUT ("<TD ALIGN=RIGHT> %3.2f</TD>\n", $pct);
        print OUT "<TD ALIGN=LEFT>";
	&PrintBarHoriz($pct,0);
        print OUT "</TD></TR>\n";
    }
    print OUT <<EOM if ($details && $stat{'accesses'} ne "0");
<TR><TD COLSPAN=4><DT><B>Access Details:</B> <I>A <A HREF="$detailslink">list</A> of individual accesses, sorted by host</I></TD></TR>
EOM
    print OUT "</TABLE><P>\n";
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTablePageStats
#    Purpose  : Prints table of pages, sorted by number of accesses.
#-----------------------------------------------------------------------------#
sub PrintTablePageStats {
    # outputs a table of pages accessed, ranked in order of demand
    
    return if ($maxPagesToList == 0);

    print OUT <<EOM;
<TABLE BORDER=1 WIDTH=100%>
<TR><TH COLSPAN=5><HR SIZE=5>Page Demand<HR SIZE=5></TH></TR>
<TR><TD ALIGN=CENTER COLSPAN=5>Of the <B>$stat{'uniqueHosts'}</B> hosts that visited today, each traversed an average of <B>$stat{'hostPageAverage'}</B> pages.</TD></TR>
<TR><TH COLSPAN=2>Page Location </TH><TH>Accesses </TH><TH>\% of total</TH></TR>
EOM

    local(@countOrderList) = sort byReverseNumber values %pages;
    local(@pageOrder) = ();

    local($pageCount) = $maxPagesToList - 1;
    $pageCount = $#countOrderList if ($maxPagesToList == -1 || 
				     ($maxPagesToList - 1) > $#countOrderList);

    while (@countOrderList) {	
	while (local($tag, $value) = (each %pages)) {
	    if ($countOrderList[0] == $value) {
		push (@pageOrder, $tag);
		shift @countOrderList;
	    }
	}
    }

    local(@favPages) = @pageOrder[0..$pageCount];

    foreach (@favPages) {
	local($Mpage) = $_;
	local($Mcount) = $pages{$_};

	local($pct) = 0.00;
	$pct = $Mcount / $stat{'accesses'} * 100 if ($stat{'accesses'} > 0);

	print OUT "<TR><TD COLSPAN=2><DT><A HREF=\"$Mpage\">";

	local($description) = &DescribePage($Mpage);

	print OUT "$description </A></TD>";
	
	print OUT "<TD ALIGN=RIGHT>$Mcount </TD>\n";
	printf OUT ("<TD ALIGN=RIGHT> %3.2f</TD>\n", $pct);
        print OUT "<TD ALIGN=LEFT>";
	&PrintBarHoriz($pct,0);
        print OUT "</TD></TR>\n";
    }
    print OUT "</TABLE><P>\n";
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTableHourlyStats
#    Purpose  : Prints bar graph of accesses over the course of the current
#                   day. Thanks very much to Paul Blackman for his work on
#                   this function. 
#-----------------------------------------------------------------------------#
sub PrintTableHourlyStats {

local($hourBar) = "img/hourbar.gif";
local($hour, $pct);

    print OUT <<EOM;
<TABLE BORDER=1 WIDTH=100%>
<TR><TH COLSPAN=3><HR SIZE=5>Hourly Statistics<HR SIZE=5></TH></TR>
<TR>
EOM
    print OUT "<TD ROWSPAN=11>";
    foreach $hour ('00'..'23') {
	if ($stat{'hr'.$hour} > 0.9*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 9);
	}
	elsif ($stat{'hr'.$hour} > 0.8*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 8);
	}
	elsif ($stat{'hr'.$hour} > 0.7*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 7);
	}
	elsif ($stat{'hr'.$hour} > 0.6*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 6);
	}
	elsif ($stat{'hr'.$hour} > 0.5*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 5);
	}
	elsif ($stat{'hr'.$hour} > 0.4*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 4);
	}
	elsif ($stat{'hr'.$hour} > 0.3*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 3);
	}
	elsif ($stat{'hr'.$hour} > 0.2*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 2);
	}
	elsif ($stat{'hr'.$hour} > 0.1*$stat{'maxhouraccess'}) {
	    &PrintBarVert($stat{'hr'.$hour}, 1);
	}
	elsif ($stat{'hr'.$hour} > 0) {
	    &PrintBarVert($stat{'hr'.$hour}, 0); 
	}
	else { 
	    &PrintBarVert(1, -1, $hour); 
	}
    }

    print OUT <<EOM;
<BR>
<IMG SRC="$hourBar" WIDTH=288 HEIGHT=22 BORDER=0 HSPACE=0 VSPACE=0 ALT="">
</TD>
<TD COLSPAN=2><TABLE BORDER=1 WIDTH=100%>
<TR><TH ALIGN=RIGHT>Avg Accesses/Hour</TH><TD ALIGN=RIGHT>$stat{'accessesPerHour'}</TD></TR>
<TR><TH ALIGN=RIGHT>Max Accesses/Hour</TH><TD ALIGN=RIGHT>$stat{'maxhouraccess'}</TD></TR>
<TR><TH ALIGN=RIGHT>Min Accesses/Hour</TH><TD ALIGN=RIGHT>$stat{'minhouraccess'}</TD></TR>
<TR><TH ALIGN=RIGHT>Accesses/Day</TH><TD ALIGN=RIGHT>$stat{'accessesPerDay'}</TD></TR>
</TABLE></TD></TR>
EOM

    foreach $pct (0..9) {
	$img = 9 - $pct;
	print OUT "<TR><TD ALIGN=LEFT><IMG SRC=\"$vertbar{$img}\" HEIGHT=8 WIDTH=10 BORDER=1 ALT=\"\"> &gt ";
	printf OUT ("%d%%</TD>", (9 - $pct)*10);
	printf OUT ("<TD ALIGN=RIGHT>%d accesses</TD></TR>\n", (1 - $pct/10) * $stat{'maxhouraccess'});
    }

    print OUT <<EOM;
</TABLE><P>
EOM

}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintBarVert
#    Purpose  : Prints a vertical bar with height as specified by argument.
#-----------------------------------------------------------------------------#
sub PrintBarVert {
    local($pct) = $_[0];
    local($colorbar) = $vertbar{$_[1]};

    local($scale) = 0;
    $scale = $pct/$stat{'maxhouraccess'} * 200 if ($stat{'maxhouraccess'});

    print OUT "<IMG SRC=\"$colorbar\" ";
    printf OUT ("HEIGHT=%d WIDTH=10 BORDER=1 ALT=\"\">", $scale);
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintBarHoriz
#    Purpose  : Prints a horizontal bar with width as specified by argument.
#-----------------------------------------------------------------------------#
sub PrintBarHoriz {
    local($pct) = $_[0];
    local($colorbar) = $horizbar{$_[1]};
    local($scale) = 1;

    $scale = ($pct*8)/log $pct + 1 if ($pct > 0);
    print OUT "<IMG SRC=\"$colorbar\" ALT=\"";
    print OUT "*" x ($pct/3 + 1) . "\" ";
    printf OUT ("HEIGHT=15 WIDTH=%d BORDER=1>", $scale);
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTableSummaryStats
#    Purpose  : Prints a table which contains general statistics.
#-----------------------------------------------------------------------------#
sub PrintTableSummaryStats {

    print OUT <<EOM;
<TABLE BORDER=1 WIDTH=100%>
<TR><TH COLSPAN=4><HR SIZE=5>Summary Statistics<HR SIZE=5></TH></TR>
<TR><TH> </TH><TH>Count</TH><TH>% of total</TH><TH> </TH></TR>
<TR><TH ALIGN=RIGHT><DT>Accesses from $orgname </TH>
<TD ALIGN=RIGHT>$stat{'localCount'} </TD>
EOM

    printf OUT ("<TD ALIGN=RIGHT>%.3g%%</TD>", $stat{'localPercent'});
    print OUT "<TD ALIGN=LEFT>";
    &PrintBarHoriz($stat{'localPercent'}/2,0);
    print OUT "</TD>";
    print OUT <<EOM;
</TR>
<TR><TH ALIGN=RIGHT><DT>Outside Accesses </TH>
<TD ALIGN=RIGHT>$stat{'outsideCount'} </TD>
EOM

    printf OUT ("<TD ALIGN=RIGHT>%.3g%%</TD>", $stat{'outsidePercent'});
    print OUT "<TD ALIGN=LEFT>";
    &PrintBarHoriz($stat{'outsidePercent'}/2,0);
    print OUT "</TD>";
    print OUT <<EOM;
</TR>
<TR><TH ALIGN=RIGHT><DT>Total Page Accesses</TH>
<TD ALIGN=RIGHT>$stat{'accesses'} </TD><TD ALIGN=RIGHT>100%</TD>
EOM
    print OUT "<TD ALIGN=LEFT>";
    &PrintBarHoriz(50,1);
    print OUT "</TD>";

print OUT <<EOM;
</TR>
<TR><TD COLSPAN=4> </TD></TR>
<TR><TH ALIGN=RIGHT><DT>Total hits related to page </TH>
<TD ALIGN=RIGHT>$stat{'hits'} </TD>
EOM

    printf OUT ("<TD ALIGN=RIGHT>%.3g%%</TD>", $stat{'serverLoad'});
    print OUT "<TD ALIGN=LEFT>";
    &PrintBarHoriz($stat{'serverLoad'}/2,0);
    print OUT "</TD>";
    print OUT <<EOM;
</TR>
<TR><TH ALIGN=RIGHT><DT>Total hits on server </TH>
<TD ALIGN=RIGHT>$stat{'serverCount'} </TD>
<TD ALIGN=RIGHT>100%</TD>
EOM
    print OUT "<TD ALIGN=LEFT>";
    &PrintBarHoriz(50,1);
    print OUT "</TD>";
    print OUT <<EOM;
</TR>
</TABLE><P>
EOM

}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintHeader 
#    Purpose  : Creates HTML header for page and sends it to OUT.
#-----------------------------------------------------------------------------#
sub PrintHeader {
    # *** Do not modify or remove the call to this function. ***
    # Purpose: creates an HTML header for the page
    
    local($totalTime) = time - $startTime;

    print OUT <<EOM;		
<HTML>
<HEAD>
<TITLE>$siteName - AccessWatch Summary</TITLE>
<LINK REV=MADE HREF="mailto:dmaher\@bucknell.edu">
<!-- This page was generated by AccessWatch v$version - Copyright $year David G. Maher. All rights reserved. Removal of this line is against applicable copyright laws. -->
</HEAD>
<BODY $bodyArgs>
<TABLE BORDER=0 WIDTH=100%>
<TR>
<TD><A HREF="http://www.eg.bucknell.edu/~dmaher/accesswatch/"><IMG WIDTH=123 HEIGHT=102 SRC="img/accesswatch.gif" BORDER=0 ALT="AccessWatch"></A></TD>
<TD ALIGN=RIGHT><H2>Accesses for $siteName<BR>
$longmonths[$mon] $mday, $year</H2>
<B>Last updated : <KBD>
EOM
    &PrintTimeString($hour, $min, $sec);

print OUT <<EOM;
</B>
<H5><I>AccessWatch took $totalTime seconds to gather current data</I></H5>
</TD></TR>
</TABLE>
<HR>

EOM
}

#------------------------------------------------------------------------------
#  AccessWatch function - PrintFooter 
#    Purpose  : Creates HTML footer for page and sends it to OUT.
#    Note     : *** Do not modify this function, or remove its call. ***
#------------------------------------------------------------------------------
sub PrintFooter {

    local($fSiteName) = $siteName;
    $fSiteName =~ tr/ /+/;
    $fSiteName = "Unknown" if ($fSiteName eq "");

    print OUT <<EOM;
<HR>
$customFooter<BR CLEAR=BOTH>
<TABLE BORDER=1 CELLPADDING=5 WIDTH=100%>
<TR>
<TD><A HREF="http://www.eg.bucknell.edu/~dmaher/accesswatch/"><IMG HSPACE=0 VSPACE=0 WIDTH=51 HEIGHT=52 BORDER=0 SRC="img/aw_icon.gif" ALT="AccessWatch"></A></TD>
<TD>
This page was produced by <I><A HREF="http://www.eg.bucknell.edu/~dmaher/accesswatch/">AccessWatch v$version</A></I>,
a WWW utility written by
<A HREF="http://www.eg.bucknell.edu/~dmaher/">Dave Maher</A>
<A HREF="mailto:dmaher\@bucknell.edu">
&lt;dmaher\@bucknell.edu&gt;</A>
Copyright &copy;$year All Rights Reserved.
</TD>
</TR>
</TABLE>

<IMG SRC="http://www.eg.bucknell.edu/cgi-bin/dmaher/accesscount.gif?$version,$fSiteName" WIDTH=1 HEIGHT=1 ALT="">
</BODY>
</HTML>
EOM
}

#-----------------------------------------------------------------------------#
#  AccessWatch function - PrintTimeString
#    Purpose  : Creates formatted string of text and sends to OUT.
#    Arguments: 3 integers - The hour, minute, and seconds.
#-----------------------------------------------------------------------------#
sub PrintTimeString {
    local($pm);
    local($hour, $min, $sec) = ($_[0], $_[1], $_[2]);

    if ($hour > 12) { 
	$pm = $hour - 12;
	print OUT "$pm:"; 
    }
    elsif ($hour == 0) {
	print OUT "12:";
    }
    elsif ($hour == 12) {
	$pm = 12;
	print OUT "$pm:";
    }
    else { print OUT "$hour:"; }
    if ($min > 9) { print OUT "$min:"; }		       
    else { print OUT "0$min:"; }
    if ($sec > 9) { print OUT "$sec"; }		       
    else { print OUT "0$sec"; }
    if ($pm) { print OUT "</KBD> p.m."; }
    else { print OUT "</KBD> a.m."; }
}

__END__


