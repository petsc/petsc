%% THIS FILE IS ESUB2ACM.BST  V1.1 (11/11/96)
%%
%% THE masterthesis AND phdthesis FUNCTIONS HAVE BEEN FIXED.
%%
%% This is file `esub2acm.bst', BASED ON 'ACMTRANS.BST' which
%% was originally generated on <1995/8/22>
%% with the docstrip utility (2.2i).
%%
%% esub2acm.bst (V1.1) should be used with esub2acm.sty (V1.0)
%%
%% Both of these files (and others) can be downloaded from
%% ACM's FTP site at:  ftp.acm.org in the directory:
%% /pubs/submitting_accepted_articles/latex/
%%
%% The original source files were:
%%
%% acmtrans.dtx
%%
%% IMPORTANT NOTICE:
%% You are not allowed to distribute this file.
%% For distribution of the original source see
%% the copyright notice in the file acmtrans.dtx .
%%
ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    pub-year
    pub-month
    pub-address
    school
    series
    title
    type
    volume
    year
  }
  {}
  { label.year extra.label sort.year sort.label }
INTEGERS {
    output.state
    before.all
    mid.sentence
    after.sentence
    after.block
}
FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}
STRINGS { s t u }
FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
    { add.period$ write$
      newline$
      "\newblock " write$
    }
    { output.state before.all =
        'write$
        { add.period$ " " * write$ }
      if$
    }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
FUNCTION {output.nonnull.colon}
{ 's :=
  output.state mid.sentence =
    { ": " * write$ }
    { output.state after.block =
    { add.period$ write$
      newline$
      "\newblock " write$
    }
    { output.state before.all =
        'write$
        { add.period$ " " * write$ }
      if$
    }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.colon}
{ duplicate$ empty$
    'pop$
    'output.nonnull.colon
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}
FUNCTION {output.check.colon}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull.colon
  if$
}
FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{
    output.state after.block =
        'skip$
        { output.state before.all =
            'skip$
            { after.sentence 'output.state := }
          if$
        }
    if$
}
FUNCTION {output.year.check}
{ year empty$
     { "empty year in " cite$ * warning$ }
     { add.period$ write$
       " \bibyear{" year * extra.label * "}" *
       mid.sentence 'output.state :=
     }
  if$
}
FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}
FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}
FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}
FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {parenthesize}
{ duplicate$ empty$
    { pop$ "" }
    { " (" swap$ * ")" * }
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\bibemph{" swap$ * "}" * }
  if$
}
FUNCTION {emphasize.ic}
{ duplicate$ empty$
    { pop$ "" }
    { "\bibemphic{" swap$ * "}" * }
  if$
}
FUNCTION {emphasize.comma}
{ duplicate$ empty$
    { pop$ "" }
    { emphasize "," * }
  if$
}
FUNCTION {smallcaps}
{ duplicate$ empty$
    { pop$ "" }
    { "\bibsc{" swap$ * "}" * }
  if$
}
INTEGERS { nameptr namesleft numnames }
FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=               % nameptr = 1;
  s num.names$ 'numnames :=    % numnames = num.name$(s);
  numnames 'namesleft :=
    { namesleft #0 > }
    {
        s nameptr "{vv~}{ll}{, f.}{, jj}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
              { ", " * t * }
              { numnames #2 >
                  { "," * }
                  'skip$
                if$
                t "others" =
                    { " et~al." * }
                    { " and " * t * } % from Chicago Manual of Style
                  if$
               }
               if$
             }
            't
        if$
        nameptr #1 + 'nameptr :=          % nameptr += 1;
        namesleft #1 - 'namesleft :=      % namesleft =- 1;
    }
  while$
    smallcaps
}
FUNCTION {my.full.label}
{ 's :=
  #1 'nameptr :=               % nameptr = 1;
  s num.names$ 'numnames :=    % numnames = num.name$(s);
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}" format.name$ 't :=  % get the next name
      nameptr #1 >
        { namesleft #1 >
              { ", " * t * }
              { numnames #2 >
                  { "," * }
                  'skip$
                if$
                t "others" =
                    { " et~al." * }
                    { " and " * t * } % from Chicago Manual of Style
                  if$
               }
               if$
             }
            't
        if$
        nameptr #1 + 'nameptr :=          % nameptr += 1;
        namesleft #1 - 'namesleft :=      % namesleft =- 1;
    }
  while$
}
FUNCTION {format.names.fml}
{ 's :=
  #1 'nameptr :=               % nameptr = 1;
  s num.names$ 'numnames :=    % numnames = num.name$(s);
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
              { ", " * t * }
               { numnames #2 >
                    { "," * }
                    'skip$
                  if$
                  t "others" =
                        { " et~al." * }
                        { " and " * t * }
                      if$
                }
               if$
             }
            't
        if$
        nameptr #1 + 'nameptr :=          % nameptr += 1;
        namesleft #1 - 'namesleft :=      % namesleft =- 1;
    }
  while$
    smallcaps
}
FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}
FUNCTION {format.key}
{ empty$
    { key field.or.null }
    { "" }
  if$
}
FUNCTION {format.editors.fml}
{ editor empty$
    { "" }
    { editor format.names.fml
      editor num.names$ #1 >
    { " Eds." * }
    { " Ed." * }
      if$
    }
  if$
}
FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
    { " Eds." * }
    { " Ed." * }
      if$
    }
  if$
}
FUNCTION {format.title}
{ title empty$
    { "" }
    { title "t" change.case$ }
  if$
}
FUNCTION {n.dashify}
{ 't :=
  ""
  { t empty$ not }
    { t #1 #1 substring$ "-" =
    { t #1 #2 substring$ "--" = not
        { "--" *
          t #2 global.max$ substring$ 't :=
        }
        {   { t #1 #1 substring$ "-" = }
        { "-" *
          t #2 global.max$ substring$ 't :=
        }
          while$
        }
      if$
    }
    { t #1 #1 substring$ *
      t #2 global.max$ substring$ 't :=
    }
      if$
    }
  while$
}
FUNCTION {format.btitle}
{ edition empty$
  { title emphasize }
  { title empty$
    { title emphasize }
    { volume empty$     % gnp - check for volume, then don't need period
       { title emphasize.ic edition " ed." * parenthesize * "." * }
       { title emphasize.ic edition " ed." * parenthesize * }
      if$
    }
    if$
  }
  if$
}
FUNCTION {format.emphasize.booktitle}
{ edition empty$
  { booktitle emphasize.ic }
  { booktitle empty$
    { booktitle emphasize }
    { volume empty$  % gnp - extra period an error if book has a volume
        { booktitle emphasize.ic edition " ed." * parenthesize * "." *}
        { booktitle emphasize.ic edition " ed." * parenthesize * }
      if$
      }
    if$
    }
  if$
}
FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}
FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { "Volume" volume tie.or.space.connect % gnp - changed to mixed case
      series empty$
        'skip$
        { " of " * series emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
    { series field.or.null }
    { "Number" number tie.or.space.connect
      series empty$
        { "there's a number but no series in " cite$ * warning$ }
        { " in " * series * }
      if$
    }
      if$
    }
    { "" }
  if$
}
INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
    { #1 'multiresult := }
    { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}
FUNCTION {format.pages}
{ pages empty$
  { "" }
  {
        "pp.\ " pages n.dashify tie.or.space.connect
  }
  if$
}
FUNCTION {format.jour.vol}
{ journal empty$
    { "no journal in " cite$ * warning$
      "" }
    { journal emphasize.ic }
    if$
  number empty$
    { volume empty$
       { "no number and no volume in " cite$ * warning$
         "" * }
       { "~" * volume emphasize * }
      if$
    }
    { volume empty$
      {"no volume for " cite$ * warning$
       "~" * number * }
      { "~" *
        volume emphasize.comma
        "~" * number * * }
      if$
     month empty$
     'skip$
     {month parenthesize *}
     if$
    }
  if$
  pages empty$
    {"page numbers missing in " cite$ * warning$
     "" * } % gnp - place a null string on the stack for output
    { duplicate$ empty$
      { pop$ format.pages }
      { ", " *  pages n.dashify * } % gnp - removed pp. for articles
      if$
    }
  if$
}
FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { "Chapter" } % gnp - changed to mixed case
        { type "t" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
        {"page numbers missing in " cite$ * warning$}%gnp - added check
        { ", " * format.pages * }
      if$
    }
  if$
}
FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
  { "" }
  { editor empty$
    { "In " format.emphasize.booktitle * }
    { "In " format.editors.fml * ", " * format.emphasize.booktitle * }
    if$
  }
  if$
}
FUNCTION {format.thesis.title}
{ title empty$
    'skip$
   { { "" }
     title "t" change.case$
   }
  if$
}
FUNCTION {format.tr.number}
{ type empty$
    { "Technical Report" }
    'type
  if$
  number empty$
    { "t" change.case$ }
    { number tie.or.space.connect }
  if$
}
FUNCTION {format.lab.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #2 >    % change number to number of others allowed before
                   % forcing "et al".
    { s #1 "{vv~}{ll}" format.name$ " et~al." * }
    {
      numnames #1 - 'namesleft :=
      #2 'nameptr :=
      s #1 "{vv~}{ll}" format.name$
        { namesleft #0 > }
        { nameptr numnames =
            { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ "others" =
                { " et~al." * }
                { " and " * s nameptr "{vv~}{ll}" format.name$ * }
              if$
            }
            { ", " * s nameptr "{vv~}{ll}" format.name$ * }
          if$
          nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      while$
    }
  if$
}
FUNCTION {author.key.label}
{ author empty$
    { key empty$
          { "no key, author in " cite$ * warning$
            cite$ #1 #3 substring$ }
         'key
      if$
    }
    { author format.lab.names }
  if$
}
FUNCTION {editor.key.label}
{ editor empty$
    { key empty$
          { "no key, editor in " cite$ * warning$
            cite$ #1 #3 substring$ }
          'key
        if$
     }
     { editor format.lab.names }
  if$
}
FUNCTION {author.key.org.label}
{ author empty$
    { organization empty$
        { key empty$
            { "no key, author or organization in " cite$ * warning$
              cite$ #1 #3 substring$ }
            'key
          if$
        }
        'organization
      if$
    }
    { author format.lab.names }
  if$
}
FUNCTION {editor.key.org.label}
{ editor empty$
    { organization empty$
        { key empty$
            { "no key, editor or organization in " cite$ * warning$
              cite$ #1 #3 substring$ }
            'key
          if$
        }
        'organization
      if$
    }
    { editor format.lab.names }
  if$
}
FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
          { key empty$
               { "no key, author, or editor in " cite$ * warning$
                 cite$ #1 #3 substring$ }
             'key
           if$
         }
          { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}
FUNCTION {calc.label}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.org.label
        { type$ "manual" =
            'author.key.org.label
            'author.key.label
          if$
        }
      if$
    }
  if$
  author empty$  % generate the full label citation information.
    { editor empty$
        { organization empty$
           { "no author, editor, or organization in " cite$ * warning$
             "??" }
           'organization
           if$
        }
        { editor my.full.label }
        if$
    }
    { author my.full.label }
  if$
  "}{" * swap$ * "}{" *
  year field.or.null purify$ #-1 #4 substring$
  'label.year :=
}
FUNCTION {output.bibitem}
{ newline$
  "\bibitem[\protect\citeauthoryear{" write$
  calc.label write$
  sort.year write$
  "}]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}
FUNCTION {article}
{
    output.bibitem
    format.authors "author" output.check
    author format.key output          % added
    output.year.check                 % added
    new.block
    format.title "title" output.check
    new.block
            format.jour.vol output
  new.block
  note output
  fin.entry
}
FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors
          output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  output.year.check       % added
  new.block
  format.btitle "title" output.check
      format.bvolume output
      new.block
      format.number.series output
      new.sentence
      publisher "publisher" output.check
      address output
  new.block
  note output
  fin.entry
}
FUNCTION {booklet}
{ output.bibitem
  format.authors output
  author format.key output          % added
  output.year.check                 % added
  new.block
  format.title "title" output.check
  new.block
  howpublished output
  address output
  new.block
  note output
  fin.entry
}
FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
    { "author and editor" editor either.or.check }
    'skip$
      if$
    }
  if$
  output.year.check                 % added
  new.block
  format.btitle "title" output.check
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.number.series output
      new.sentence
      publisher "publisher" output.check.colon
      address output
  new.block
  note output
  fin.entry
}
FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  author format.key output       % added
  output.year.check              % added
  new.block
  format.title "title" output.check
  new.block
    format.in.ed.booktitle "booktitle" output.check
    format.bvolume output
    format.number.series output
    format.chapter.pages output % gnp - was special.output.nonnull
    new.sentence
    address output
    publisher "publisher" output.check.colon
  new.block
  note output
  fin.entry
}
FUNCTION {format.date}
{ year empty$
    { month empty$
        { "" }
        { "there's a month but no year in " cite$ * warning$
          month
        }
      if$
    }
    { month empty$
        'year
        { month " " * year * }
      if$
    }
  if$
}
FUNCTION {format.proc.loc.date}
{
    address field.or.null
    format.date
    duplicate$ empty$
        'pop$
        { swap$
          duplicate$ empty$
            'pop$
            { ", " * swap$ * }
          if$
        }
    if$
    parenthesize
    *
}
FUNCTION {inproceedings}
{
  output.bibitem
  format.authors "author" output.check
  author format.key output            % added
  output.year.check                   % added
  new.block
  format.title "title" output.check
  new.block
      format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      format.proc.loc.date %output
      format.pages output
      new.sentence
      organization output
      publisher output.colon
  new.block
  note output
  fin.entry
}
FUNCTION {conference} { inproceedings }
FUNCTION {manual}
{ output.bibitem
  author empty$
    { editor empty$
      { organization "organization" output.check
        organization format.key output }  % if all else fails, use key
      { format.editors "author and editor" output.check }
      if$
    }
    { format.authors output.nonnull }
    if$
  output.year.check                 % added
  new.block
  format.btitle "title" output.check
  organization address new.block.checkb
  address output
  organization "organization" output.check.colon
  new.block
  note output
  fin.entry
}
FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output          % added
  output.year.check                 % added
  new.block
  format.title "title" output.check
  new.block
%
% original which caused problems:  "Master's thesis" format.thesis.title output.nonnull
% The following is its replacement (Gerry Murray 10/29/96)
%
 "Master's thesis" format.title output.check   % added by GM 10/29/96
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}
FUNCTION {misc}
{ output.bibitem
  format.authors output
  author format.key output            % added
  output.year.check                   % added
  title howpublished new.block.checkb
  format.title output
  new.block
  howpublished output
  new.block
  note output
  fin.entry
}
FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output            % added
  output.year.check                   % added
  new.block
  format.btitle "title" output.check
  new.block
% original which caused problems:  format.thesis.title "Ph.\ D. thesis" output.nonnull
%
% Below is its replacment (Gerry Murray 10/29/96)
%
  "Ph.\ D. thesis" format.title output.check        % added by GM 10/29/96
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}
FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization output
      organization format.key output } % gnp - changed from
    { format.editors output.nonnull }  % author format.key
  if$
  output.year.check                    % added (newapa)
  new.block
  format.btitle "title" output.check
  format.bvolume output
  format.number.series output
    format.proc.loc.date % output
  new.sentence
  organization output
  publisher output.colon
  new.block
  note output
  fin.entry
}
FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  author format.key output             % added
  output.year.check                    % added
  new.block
  format.title "title" output.check
  new.block
  format.tr.number
     month empty$
     'skip$
     { month parenthesize * }
     if$
  output.nonnull
  institution "institution" output.check
  address output
  new.block
  note output
  fin.entry
}
FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  author format.key output              % added
  output.year.check                      % added
  new.block
  format.title "title" output.check
  new.block
  note "note" output.check
  fin.entry
}
FUNCTION {default.type} { misc }
MACRO {jan} {"Jan."}
MACRO {feb} {"Feb."}
MACRO {mar} {"March"}
MACRO {apr} {"April"}
MACRO {may} {"May"}
MACRO {jun} {"June"}
MACRO {jul} {"July"}
MACRO {aug} {"Aug."}
MACRO {sep} {"Sept."}
MACRO {oct} {"Oct."}
MACRO {nov} {"Nov."}
MACRO {dec} {"Dec."}
MACRO {acmcs} {"ACM Computing Surveys"}
MACRO {acta} {"Acta Informatica"}
MACRO {ai} {"Artificial Intelligence"}
MACRO {cacm} {"Commun. ACM"}
MACRO {ibmjrd} {"IBM Journal of Research and Development"}
MACRO {ibmsj} {"IBM Systems Journal"}
MACRO {ieeese} {"IEEE Trans. Softw. Eng."}
MACRO {ieeetc} {"IEEE Transactions on Computers"}
MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}
MACRO {ipl} {"Inf. Process. Lett."}
MACRO {ic} {"Inf. Comput."}
MACRO {jacm} {"Journal of the ACM"}
MACRO {jcss} {"J. Comput. Syst. Sci."}
MACRO {scp} {"Science of Computer Programming"}
MACRO {sicomp} {"SIAM Journal on Computing"}
MACRO {tocs} {"ACM Transactions on Computer Systems"}
MACRO {tods} {"ACM Transactions on Database Systems"}
MACRO {tog} {"ACM Transactions on Graphics"}
MACRO {toms} {"ACM Transactions on Mathematical Software"}
MACRO {toois} {"ACM Transactions on Office Information Systems"}
MACRO {toplas} {"ACM Trans. on Programm. Lang. Syst."}
MACRO {tcs} {"Theor. Comp. Sci."}
READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
          { "   " * }
         'skip$
      if$
      s nameptr "{vv{ } }{ll{ }}{  f{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = t "others" = and
          { " et~al" * }
          { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}
FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}
FUNCTION {author.sort}
{ author empty$
    { key empty$
         { "to sort, need author or key in " cite$ * warning$
           "" }
         { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {editor.sort}
{ editor empty$
    { key empty$
         { "to sort, need editor or key in " cite$ * warning$
           ""
         }
         { key sortify }
      if$
    }
    { editor sort.format.names }
  if$
}
FUNCTION {author.editor.sort}
{
    author empty$
        { "missing author in " cite$ * warning$
          editor empty$
            { key empty$
                { "to sort, need author, editor, or key in "
                  cite$ * warning$
                  ""
                }
                {
                  key sortify
                }
              if$
            }
            {
              editor sort.format.names
            }
          if$
        }
        {
          author sort.format.names
        }
    if$
}
FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in "
              cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { organization sortify }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in "
              cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { organization sortify }
      if$
    }
    { editor sort.format.names }
  if$
}
FUNCTION {presort}
{ calc.label sortify % recalculate bibitem label
  year field.or.null purify$ #-1 #4 substring$ * % add year
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
        'editor.organization.sort
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  #1 entry.max$ substring$        % added for newapa
  'sort.label :=                  % added for newapa
  sort.label                      % added for newapa
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}
ITERATE {presort}
SORT    % by label, year, author/editor, title
STRINGS { last.label next.extra }
INTEGERS { last.extra.num }
FUNCTION {init.extra.label.stuff}
{ #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'last.extra.num :=
}
FUNCTION {forward.pass}
{ last.label
  calc.label year field.or.null purify$ #-1 #4 substring$ * % add year
  #1 entry.max$ substring$ =     % are they equal?
     { last.extra.num #1 + 'last.extra.num :=
       last.extra.num int.to.chr$ 'extra.label :=
     }
     { "a" chr.to.int$ 'last.extra.num :=
       "" 'extra.label :=
       calc.label
       year field.or.null purify$ #-1 #4 substring$ * % add year
       #1 entry.max$ substring$ 'last.label := % assign to last.label
     }
  if$
}
FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
     'skip$
  if$
  label.year extra.label * 'sort.year :=
  extra.label 'next.extra :=
}
EXECUTE {init.extra.label.stuff}
ITERATE {forward.pass}
REVERSE {reverse.pass}
FUNCTION {bib.sort.order}
{ sort.label
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}
ITERATE {bib.sort.order}
SORT % by sort.label, year, title --- giving final bib. order.
FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{}" write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
%%
%% End of file `esub2acm.bst'.
%% Mods and additional header information by Gerry Murray Dec. 27 1996
