% ESUB2ACM.CLS V1.2 - November 10th. 1999
\def\fileversion{v1.2}           % for ACM's tracking purposes
\def\filedate{November 10, 1999}    % Gerry Murray's tracking data
\def\docdate {Wed 10 Nov 1999}    % Gerry Murray (with deltas to doc}
\usepackage{epsfig}
%
% This is esub2acm.cls.
%
% It is the class (style) file that you should use
% if you intend to submit a LaTeX2e electronic file (.tex) to
% ACM for use in their electronic publishing system.
%
% It is directly derived from esub2acm.sty (LaTeX 2.09 version),
% itself a slightly modified version of David M. Jones's style file.
% It can work for both LaTeX2e and LaTeX 2.09; name the file
% esub2acm.sty to use it with LaTeX 2.09.
%
% Conversion scripts have been written to successfully convert a
% LaTeX2e submission into an SGML file, provided the submission
% has been authored using *this* particular (class) style file.
%
% Vanilla LaTeX is preferred.
%
% For additional information on submission formats check out:
% http://www.acm.org/pubs/submitting_accepted_articles/auth_rd.htm
%
% Gerry Murray 6/21/99 and 8/5/1999
% murray@hq.acm.org
%
%
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
%%
%
%  -> \documentclass{esub2acm}
%  -> \markboth{}{}
%         takes 2 arguments and it is for the left- and right-page headers:
%         the first set of braces is assigned for author's name(s)
%         and
%         the second set of braces is assigned for the title
%             (if the title is too long, contraction may be needed
%  -> \title{}
%         if the title is too long, it can be separated by \\
%  -> \author{}
%         author1 \\ author1 affiliation
%         \and
%         author2 \\ author2 affiliation
%  -> \begin{abstract}
%  -> \end{abstract}
%
%  -> \category{}{}{}
%         takes 3 arguments for the Computing Reviews Classification Scheme.
%         ex: \category{D.3.3}{Programming Languages}{Language Constructs and
%                   Features}[data types and structures]
%                   the last argument, in square brackets, is optional.
%  -> \terms{} (ex: \terms{Human Factors, Languages})
%  -> \keywords{} (in alphabetical order \keywords{document processing, sequences,
%                      string searching, subsequences, substrings})
%  -> \begin{document}
%
%  -> \begin{bottomstuff}
%          similar to \thanks
%          for authors' addresses; research/grant statements
%  -> \end{bottomstuff}
%  -> \maketitle
%
%     Now you can start the body of the paper; your figures, tables and
%          use all the latex constructs.
%
%  -> \begin{acks}
%          acknowledgements
%  -> \end{acks}
%
%  -> \bibliographystyle{esub2acm}
%  -> \bibliography{mybib_file}
%
%     ****
%     If your paper has been accepted with a separate (electronic only)
%        appendix, you need to add the following control sequence:
%
%  ->  \appendixhead{}{}{}{}
%        takes 4 arguments to identify your appendix section.
%        first = journal name (e.g.  toplas )
%        second = paper-id  (a four-digit number)
%        third = year of appearance, for copyright notice
%                 (approximation will suffice)
%        fourth = journal, volume and year of publication if known
%
%       body of appendix

%  -> \end{document}
%
% Do not worry about the other definitions in this style file
% Remember to compile: latex, bibtex, latex latex
%
% Bibliographic cite forms needed:
%
%  \cite{key}
%    which produces citations with full author list and year.
%    eg. [Brown 1978; Jarke, Turner, Stohl, et al. 1985]
%  \citeA{key}
%    which produces citations with only the full author list.
%    eg. [Brown; Jarke, Turner and Stohl]
%  \citeN{key}
%    which produces citations with the full author list and year, but
%    can be used as nouns in a sentence; no brackets appear around
%    the author names, but only around the year.
%      eg. Shneiderman [1978] states that......
%    \citeN should only be used for a single citation.
%  \shortcite{key}
%    which produces citations with abbreviated author list and year.
%  \shortciteA{key}
%    which produces only the abbreviated author list.
%  \shortciteN{key}
%    which produces the abbreviated author list and year, with only the
%    year in brackets. Use with only one citation.
%  \citeyear{key}
%    which produces the year information only, within brackets.
%
% Abbreviated author lists use the ``et al.'' construct.
%
% The above are examples of required ACM bibliographic cite formats needed.
% *******************
% Here is the complete list of cite forms from the chicago bibliographic style
%
%  \cite{key}
%    which produces citations with full author list and year.
%    eg. (Brown 1978; Jarke, Turner, Stohl, et al. 1985)
%  \citeNP{key}
%    which produces citations with full author list and year, but without
%    enclosing parentheses:
%    eg. Brown 1978; Jarke, Turner and Stohl 1985
%  \citeA{key}
%    which produces citations with only the full author list.
%    eg. (Brown; Jarke, Turner and Stohl)
%  \citeANP{key}
%    which produces citations with only the full author list, without
%    parentheses eg. Brown; Jarke, Turner and Stohl
%  \citeN{key}
%    which produces citations with the full author list and year, but
%    can be used as nouns in a sentence; no parentheses appear around
%    the author names, but only around the year.
%      eg. Shneiderman (1978) states that......
%    \citeN should only be used for a single citation.
%  \shortcite{key}
%    which produces citations with abbreviated author list and year.
%  \shortciteNP{key}
%    which produces citations with abbreviated author list and year.
%  \shortciteA{key}
%    which produces only the abbreviated author list.
%  \shortciteANP{key}
%    which produces only the abbreviated author list.
%  \shortciteN{key}
%    which produces the abbreviated author list and year, with only the
%    year in parentheses. Use with only one citation.
%  \citeyear{key}
%    which produces the year information only, within parentheses.
%  \citeyearNP{key}
%    which produces the year information only.
%
% Abbreviated author lists use the ``et al.'' construct.
%
% `NP' means `no parentheses'
%
%%% End of inserted ACM - esub2acm comments

%% CHANGES ON NEXT LINES
\newif\if@ll % to record which version of LaTeX is in use

\expandafter\ifx\csname LaTeXe\endcsname\relax % LaTeX2.09 is used
\else% LaTeX2e is used, so set ll to true
\global\@lltrue
\fi

\if@ll
  \NeedsTeXFormat{LaTeX2e}
  \ProvidesClass{esub2acm} [1999/07/18 based on esub2acm.sty <23 April 96>]
  \RequirePackage{latexsym}% QUERY: are these two really needed?
  %\RequirePackage{amssymb}
  \let\dooptions\ProcessOptions
\else
  \let\dooptions\@options
\fi
%% END CHANGES

\typeout{Document Style `esub2acm' Electronic Submissions \fileversion\space <\filedate> (ACM).}
\def\@height{height}
\def\@width{width}
\def\@minus{minus}
\def\@plus{plus}
\def\hb@xt@{\hbox to}
\newif\if@faircopy
\@faircopyfalse
\def\ds@faircopy{\@faircopytrue}

\def\ds@preprint{\@faircopyfalse}

\def\ds@twocolumn{%
    \typeout{Warning! twocolumn mode is not supported by esub2acm}%
}
\@twosidetrue
\@mparswitchtrue
\def\ds@draft{\overfullrule 5\p@}
%% CHANGE ON NEXT LINE
\dooptions

\lineskip \p@
\normallineskip \p@
\def\baselinestretch{1}
\def\@ptsize{0} %needed for amssymbols.sty

%% CHANGES ON NEXT LINES
\if@ll% allow use of old-style font change commands in LaTeX2e
\@maxdepth\maxdepth
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand*{\mit}{\@fontswitch{\relax}{\mathnormal}}
\fi

\if@ll
  \renewcommand{\@ptsize}{}
  \renewcommand{\normalsize}{%
    \@setfontsize\normalsize\@xpt{12\p@}%
    \abovedisplayskip 6\p@ \@plus2\p@ \@minus\p@
    \belowdisplayskip \abovedisplayskip
    \abovedisplayshortskip 6\p@ \@minus 3\p@
    \belowdisplayshortskip 6\p@ \@minus 3\p@
    \let\@listi\@listI
  }
\else
  \def\@normalsize{%
    \@setsize\normalsize{12\p@}\xpt\@xpt
    \abovedisplayskip 6\p@ \@plus2\p@ \@minus\p@
    \belowdisplayskip \abovedisplayskip
    \abovedisplayshortskip 6\p@ \@minus 3\p@
    \belowdisplayshortskip 6\p@ \@minus 3\p@
    \let\@listi\@listI
  }%
\fi
\if@ll
  \newcommand{\small}{%
    \@setfontsize\small\@ixpt{12\p@}%
    \abovedisplayskip 5\p@ \@plus 2\p@ \@minus \p@
    \belowdisplayskip \abovedisplayskip
    \abovedisplayshortskip 5\p@ \@minus 2\p@
    \belowdisplayshortskip 5\p@ \@minus 2\p@
    \def\@listi{%
        \leftmargin\leftmargini
        \topsep 5\p@ \@plus 2\p@ \@minus \p@
        \parsep \z@ \@plus .7\p@
        \itemsep 1.6\p@ \@plus .8\p@
    }%
  }%
\else
  \def\small{%
      \@setsize\small{11\p@}\ixpt\@ixpt
      \abovedisplayskip 5\p@ \@plus 2\p@ \@minus \p@
      \belowdisplayskip \abovedisplayskip
      \abovedisplayshortskip 5\p@ \@minus 2\p@
      \belowdisplayshortskip 5\p@ \@minus 2\p@
      \def\@listi{%
          \leftmargin\leftmargini
          \topsep 5\p@ \@plus 2\p@ \@minus \p@
          \parsep \z@ \@plus .7\p@
          \itemsep 1.6\p@ \@plus .8\p@
      }%
  }%
\fi

\if@ll
  \newcommand{\footnotesize}{%
      \@setfontsize\footnotesize\@viiipt{10\p@}%
      \abovedisplayskip 4\p@ \@plus \p@
      \belowdisplayskip \abovedisplayskip
      \abovedisplayshortskip 4\p@ \@minus \p@
      \belowdisplayshortskip 4\p@ \@minus \p@
      \def\@listi{%
          \leftmargin\leftmargini
          \topsep 4\p@ \@plus \p@
          \parsep \z@ \@plus .5\p@
          \itemsep \p@ \@plus .7\p@
      }%
  }%
\else
  \def\footnotesize{%
      \@setsize\footnotesize{10pt}\viiipt\@viiipt
      \abovedisplayskip 4\p@ \@plus \p@
      \belowdisplayskip \abovedisplayskip
      \abovedisplayshortskip 4\p@ \@minus \p@
      \belowdisplayshortskip 4\p@ \@minus \p@
      \def\@listi{%
          \leftmargin\leftmargini
          \topsep 4\p@ \@plus \p@
          \parsep \z@ \@plus .5\p@
          \itemsep \p@ \@plus .7\p@
      }%
  }%
\fi
\if@ll
  \newcommand\scriptsize{\@setfontsize\scriptsize\@viipt{8\p@}}
  \newcommand\tiny{\@setfontsize\tiny\@vpt{6\p@}}
  \newcommand\large{\@setfontsize\large\@xiipt{14\p@}}
  \newcommand\Large{\@setfontsize\Large\@xivpt{18\p@}}
  \newcommand\LARGE{\@setfontsize\LARGE\@xviipt{20\p@}}
  \newcommand\huge{\@setfontsize\huge\@xxpt{25\p@}}
  \newcommand\Huge{\@setfontsize\Huge\@xxvpt{30\p@}}
\else
  \def\scriptsize{\@setsize\scriptsize{8\p@}\viipt\@viipt}
  \def\tiny{\@setsize\tiny{6\p@}\vpt\@vpt}
  \def\large{\@setsize\large{14\p@}\xiipt\@xiipt}
  \def\Large{\@setsize\Large{18\p@}\xivpt\@xivpt}
  \def\LARGE{\@setsize\LARGE{20\p@}\xviipt\@xviipt}
  \def\huge{\@setsize\huge{25\p@}\xxpt\@xxpt}
  \def\Huge{\@setsize\Huge{30\p@}\xxvpt\@xxvpt}
\fi
\normalsize

\oddsidemargin .75in
\evensidemargin .75in
\marginparwidth .5in
\marginparsep .125in
\topmargin .25in
\headheight 12\p@
\headsep 16\p@
%% CHANGES ON NEXT LINES
\if@ll
  \newlength{\footheight}%
\else
\fi
%% END CHANGES
\footheight 10\p@
\footskip 15\p@

\textheight 47pc
\textwidth 30pc
\columnsep 10\p@
\columnseprule 0\p@

\footnotesep 7\p@
\skip\footins 15\p@ plus 4\p@ minus 3\p@

\floatsep 12\p@ plus 2\p@ minus 2\p@
\textfloatsep \floatsep

\intextsep 1pc plus 1pc

%% CHANGES ON NEXT LINES
\if@ll
  \newlength{\@maxsep}%
\else
\fi
%% END CHANGES
\@maxsep 1pc

\dblfloatsep 12\p@ plus 2\p@ minus 2\p@

\dbltextfloatsep 20\p@ plus 2\p@ minus 4\p@

%% CHANGES ON NEXT LINES
\if@ll
  \newlength{\@dblmaxsep}%
\else
\fi
%% END CHANGES
\@dblmaxsep 20\p@

\@fptop 0\p@ plus 1fil

\@fpsep 1pc plus 2fil

\@fpbot 0\p@ plus 1fil

\@dblfptop 0\p@ plus 1fil

\@dblfpsep 8\p@ plus 2fil

\@dblfpbot 0\p@ plus 1fil

\marginparpush 6\p@

\parskip 0\p@ plus .1\p@



\partopsep 0\p@

\@lowpenalty 51
\@medpenalty 151
\@highpenalty 301

\@beginparpenalty -\@lowpenalty
\@endparpenalty -\@lowpenalty
\@itempenalty -\@lowpenalty
\newcounter{part}
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\def\thepart{\Roman{part}}
\def\thesection{\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\newif\if@uchead
\@ucheadfalse

%% CHANGES: NEW NOTE
%% NOTE: OK to use old-style font commands below, since they were
%% suitably redefined for LaTeX2e
%% END CHANGES
\setcounter{secnumdepth}{3}
\def\part{%
    \@startsection{part}{9}{\z@}{-10\p@ \@plus -4\p@ \@minus -2\p@}
        {4\p@}{\normalsize\sf\@ucheadtrue}%
}
\def\section{%
    \@startsection{section}{1}{\z@}{-10\p@ \@plus -4\p@ \@minus -2\p@}%
    {4\p@}{\normalsize\sf\@ucheadtrue}%
}
\def\subsection{%
    \@startsection{subsection}{2}{\z@}{-8\p@ \@plus -2\p@ \@minus -\p@}
    {4\p@}{\normalsize\sf}%
}
\def\subsubsection{%
    \@startsection{subsubsection}{3}{\parindent}{6\p@ \@plus \p@}%
    {-5\p@}{\normalsize\it}%
}
\def\paragraph{%
    \@startsection{paragraph}{3}{\parindent}{6\p@ \@plus \p@}%
    {-5\p@}{\normalsize\it}%
}
\let\@period=.
\def\@startsection#1#2#3#4#5#6{%
    \if@noskipsec
        \global\let\@period\@empty
        \leavevmode
        \global\let\@period.%
    \fi
    \par
    \@tempskipa #4\relax
    \@afterindenttrue
    \ifdim \@tempskipa <\z@
        \@tempskipa -\@tempskipa
        \@afterindentfalse
    \fi
    \if@nobreak
        \everypar{}%
    \else
        \addpenalty\@secpenalty
        \addvspace\@tempskipa
    \fi
    \@ifstar
        {\@ssect{#3}{#4}{#5}{#6}}
        {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}%
}
\def\@sect#1#2#3#4#5#6[#7]#8{%
    \ifnum #2>\c@secnumdepth
        \let\@svsec\@empty
    \else
        \refstepcounter{#1}%
        \edef\@svsec{%
            \begingroup
                \ifnum#2>2 \noexpand\rm \fi
                \csname the#1\endcsname
            \endgroup
            \ifnum #2=1\relax .\fi
            \hskip .5em
        }%
    \fi
    \@tempskipa #5\relax
    \ifdim \@tempskipa>\z@
        \begingroup
            #6\relax
            \@hangfrom{\hskip #3\relax\@svsec}%
            \begingroup
                \interlinepenalty \@M
                \if@uchead
                    \uppercase{#8}%
                \else
                    #8%
                \fi
                \par
            \endgroup
        \endgroup
        \csname #1mark\endcsname{#7}%
        \addcontentsline{toc}{#1}{%
            \ifnum #2>\c@secnumdepth \else
                \protect\numberline{\csname the#1\endcsname}%
            \fi
            #7%
        }%
    \else
        \def\@svsechd{%
            #6%
            \hskip #3\relax
            \@svsec
            \if@uchead
                \uppercase{#8}%
            \else
                #8%
            \fi
            \csname #1mark\endcsname{#7}%
            \addcontentsline{toc}{#1}{%
                \ifnum #2>\c@secnumdepth \else
                    \protect\numberline{\csname the#1\endcsname}%
                \fi
                #7%
            }%
        }%
    \fi
    \@xsect{#5}%
}
\def\@xsect#1{%
    \@tempskipa #1\relax
    \ifdim \@tempskipa>\z@
        \par
        \nobreak
        \vskip \@tempskipa
        \@afterheading
    \else
        \global\@nobreakfalse
        \global\@noskipsectrue
        \everypar{%
            \if@noskipsec
                \global\@noskipsecfalse
                \clubpenalty\@M
                \hskip -\parindent
                \begingroup
                    \@svsechd
                    \@period
                \endgroup
                \unskip
                \@tempskipa #1\relax
                \hskip -\@tempskipa
            \else
                \clubpenalty \@clubpenalty
                \everypar{}%
            \fi
        }%
    \fi
    \ignorespaces
}
\def\@trivlist{%
    \@topsepadd\topsep
    \if@noskipsec
        \global\let\@period\@empty
        \leavevmode
        \global\let\@period.%
    \fi
    \ifvmode
        \advance\@topsepadd\partopsep
    \else
        \unskip
        \par
    \fi
    \if@inlabel
        \@noparitemtrue
        \@noparlisttrue
    \else
        \@noparlistfalse
        \@topsep\@topsepadd
    \fi
    \advance\@topsep \parskip
    \leftskip\z@skip
    \rightskip\@rightskip
    \parfillskip\@flushglue
    \@setpar{\if@newlist\else{\@@par}\fi}
    \global\@newlisttrue
    \@outerparskip\parskip
}
\newfont{\apbf}{cmbx9}
\def\@withappendix#1{\rm A--\number #1}
\def\appenheader{%
    \global\@topnum\z@
    \global\@botroom \textheight
    \begin{@float}{figure}[b]
        \footnotesize
        \newfont{\sc}{cmcsc10}% %%* ACK!
        \parindent\z@
        \null
        \vskip -\textfloatsep
        \vskip 11\p@
        \hrule \@height .2\p@ \@width 30pc
        \vskip 2\p@
        \rule{\z@}{10\p@}%
        \ignorespaces
}

\def\endappenheader{%
    \end{@float}
    \global\let\appendixhead\@empty
}
\long\def\appendixhead#1#2#3#4{%
    \section*{Appendix}

    An appendix to this paper is available in electronic form
    (PostScript$^{\mbox{\tiny TM}}$).  Any of the following methods
    may be used to obtain it; or see the inside back cover of a
    current issue for up-to-date instructions.

    \begin{itemize}
    \item By anonymous ftp from {\apbf acm.org}, file
    {\apbf [pubs.journals.#1.append]p#2.ps}
    \item Send electronic mail to {\apbf mailserve@acm.org} containing
    the line\\ {\apbf send~[anonymous.pubs.journals.#1.append]p#2.ps}
    \item By {\it Gopher} from {\apbf acm.org}
    \item By anonymous ftp from {\apbf ftp.cs.princeton.edu},
    file {\apbf pub/#1/append/p#2.ps}
    \item Hardcopy from {\it Article Express}, for a fee: phone
    800-238-3458, fax 201-216-8526, or write P.O.~Box~1801, Hoboken NJ
    07030; and request {\sc acm-#1-appendix-{\small #2}}.
    \end{itemize}

    \clearpage
    \pagenumbering{withappendix}
    \appendix
    \par
    \noindent{\sc this document is the appendix to the following
    paper:}
    \hfill
    \vskip 1em
    \vbox{\sf \parindent\z@\@title}%
    \vskip .5em
    \vbox{\sf \parindent\z@\@author}%
    \vskip .5em\noindent
    \vskip 10\p@\noindent #4
    \vskip 11\p@\noindent
    \hrule \@height .2\p@
    \par
    \begin{bottomstuff}
    \permission
    \copyright #3 ACM
    \par
    \end{bottomstuff}
}
\def\appendix{%
    \par
    \part{Appendix}%
    \par
    \setcounter{section}{0}%
    \setcounter{subsection}{0}%
    \def\thesection{\Alph{section}}%
    \def\thesubsection{\thesection.\arabic{subsection}}%
}
\labelsep 5\p@

\settowidth{\leftmargini}{(9)}
\addtolength\leftmargini\labelsep

\settowidth{\leftmarginii}{(b)}
\addtolength\leftmarginii\labelsep

\leftmarginiii  \leftmarginii
\leftmarginiv   \leftmarginii
\leftmarginv    \leftmarginii
\leftmarginvi   \leftmarginii
\leftmargin     \leftmargini

\labelwidth\leftmargini
\advance\labelwidth-\labelsep
\def\@listI{%
    \leftmargin\leftmargini
    \parsep \z@ \@plus \p@
    \topsep 6\p@ \@plus 2\p@ \@minus 2\p@
    \itemsep 2\p@ \@plus \p@ \@minus .5\p@
}

\let\@listi\@listI

\@listi
\def\@listii{%
    \leftmargin\leftmarginii
    \labelwidth\leftmarginii
    \advance\labelwidth-\labelsep
    \topsep \z@ \@plus \p@
    \parsep \z@ \@plus .5\p@
    \itemsep \parsep
}
\def\@listiii{%
    \leftmargin\leftmarginiii
    \labelwidth\leftmarginiii
    \advance\labelwidth-\labelsep
    \topsep \z@ \@plus \p@
    \parsep \z@ \@plus .5\p@
    \itemsep \parsep
}
\def\@listiv{%
    \leftmargin\leftmarginiv
    \labelwidth\leftmarginiv
    \advance\labelwidth-\labelsep
}
\def\@listv{%
    \leftmargin\leftmarginv
    \labelwidth\leftmarginv
    \advance\labelwidth-\labelsep
}
\def\@listvi{%
    \leftmargin\leftmarginvi
    \labelwidth\leftmarginvi
    \advance\labelwidth-\labelsep
}
\def\enumerate{%
    \ifnum \@enumdepth >3
        \@toodeep
    \else
        \advance\@enumdepth \@ne
        \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
        \list{\csname label\@enumctr\endcsname}{%
            \usecounter{\@enumctr}%
            \def\makelabel##1{##1\hss}%
        }%
    \fi
}

\let\endenumerate\endlist
\def\longenum{%
    \ifnum \@enumdepth >3
        \@toodeep
    \else
        \advance\@enumdepth \@ne
        \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
        \list{\csname label\@enumctr\endcsname}{%
            \usecounter{\@enumctr}%
            \labelwidth\z@
        }%
    \fi
}

\let\endlongenum\endlist
\def\labelenumi{(\arabic{enumi})}
\def\theenumi{\arabic{enumi}}
\def\labelenumii{(\alph{enumii})}
\def\theenumii{\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{\roman{enumiii}.}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{\Alph{enumiv}.}
\def\theenumiv{\Alph{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}
\def\itemize{%
    \list{---\hskip -\labelsep}{%
        \settowidth{\leftmargin}{---}%
        \labelwidth\leftmargin
        \addtolength{\labelwidth}{-\labelsep}%
    }%
}

\let\enditemize\endlist
\def\longitem{%
    \list{---}{%
        \labelwidth\z@
        \leftmargin\z@
        \itemindent\parindent
        \advance\itemindent\labelsep
    }%
}

\let\endlongitem\endlist
\def\verse{%
    \let\\=\@centercr
    \list{}{%
        \leftmargin 2pc
        \itemindent -1.5em
        \listparindent \itemindent
        \rightmargin\leftmargin
        \advance\leftmargin 1.5em
    }%
    \item[]
}

\let\endverse\endlist
\def\quotation{%
    \list{}{%
        \leftmargin 2pc
        \listparindent .5em
        \itemindent\listparindent
        \rightmargin\leftmargin
        \parsep \z@ \@plus \p@
    }%
    \item[]
}

\let\endquotation=\endlist
\def\quote{%
    \list{}{%
        \leftmargin 2pc
        \rightmargin\leftmargin
    }%
    \item[]
}

\let\endquote=\endlist

% This version of describe puts unwanted periods and the
% indenting is not aesthetic. See replacment below.
%\def\description{%
%    \list{}{%
%        \listparindent\parindent
%        \labelwidth\z@
%        \leftmargin\z@
%        \itemindent\parindent
%        \advance\itemindent\labelsep
%        \def\makelabel##1{\it ##1.}%
%    }%
%}
%\let\enddescription\endlist

% SUBSTITUTE VERSION
% GERRY MURRAY  - -  NOVEMBER 1999
\def\descriptionlabel#1{%
  \hspace\labelsep \normalfont\bfseries #1%
}%
\newenvironment{description}{%
 \list{}{%
    \labelwidth\z@ \itemindent-\leftmargin
    \let\makelabel\descriptionlabel
 }%
}{%
 \endlist
}%
% END OF SUBSTITUTE VERSION

\def\describe#1{%
    \list{}{%
        \listparindent\parindent
        \settowidth{\labelwidth}{#1}%
        \leftmargin\labelwidth
        \addtolength\leftmargin\labelsep
        \def\makelabel##1{##1\hfil}%
    }%
}

\let\enddescribe\endlist

\def\program{%
    \ifx\@currsize\normalsize
        \small
    \else
        \rm
    \fi
    \tabbing
}

\let\endprogram\endtabbing

\def\@begintheorem#1#2{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\sc #1\hskip 5\p@\relax#2.}%
    ]
    \it
}
\def\@opargbegintheorem#1#2#3{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\sc #1\  #2%             % Gil Neiger (for numbering collaries and such) Sept. 1999
        \savebox\@tempboxa{(#3)}% GN
        \ifdim \wd\@tempboxa>\z@
            \hskip5\p@\relax
            \box\@tempboxa
        \fi.}%
    ]
    \it
}
\newif\if@qeded
\global\@qededfalse
\def\proof{%
    \global\@qededfalse
    \@ifnextchar[{\@xproof}{\@proof}%
}

\def\endproof{%
    \if@qeded\else\qed\fi
    \endtrivlist
}
\def\@proof{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\sc Proof.}%
    ]
    \ignorespaces
}
\def\@xproof[#1]{%
    \trivlist
    \item[\hskip 10\p@\hskip \labelsep{\sc Proof #1.}]%
    \ignorespaces
}
\def\qed{%
    \unskip
    \kern 10\p@
    \begingroup
        \unitlength\p@
        \linethickness{.4\p@}%
        \framebox(6,6){}%
    \endgroup
    \global\@qededtrue
}
\def\newdef#1#2{%
    \expandafter\@ifdefinable\csname #1\endcsname
        {\@definecounter{#1}%
         \expandafter\xdef\csname the#1\endcsname{\@thmcounter{#1}}%
         \global\@namedef{#1}{\@defthm{#1}{#2}}%
         \global\@namedef{end#1}{\@endtheorem}%
    }%
}
\def\@defthm#1#2{%
    \refstepcounter{#1}%
    \@ifnextchar[{\@ydefthm{#1}{#2}}{\@xdefthm{#1}{#2}}%
}
\def\@xdefthm#1#2{%
    \@begindef{#2}{\csname the#1\endcsname}%
    \ignorespaces
}
\def\@ydefthm#1#2[#3]{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\it #2%
         \savebox\@tempboxa{#3}%
         \ifdim \wd\@tempboxa>\z@
            \ \box\@tempboxa
         \fi.%
        }]%
    \ignorespaces
}
\def\@begindef#1#2{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\it #1\ \rm #2.}%
    ]%
}
\def\theequation{\arabic{equation}}
\def\titlepage{%
    \@restonecolfalse
    \if@twocolumn
        \@restonecoltrue
        \onecolumn
    \else
        \newpage
    \fi
    \thispagestyle{empty}%
    \c@page\z@
}

\def\endtitlepage{%
    \if@restonecol
        \twocolumn
    \else
        \newpage
    \fi
}
\arraycolsep 2.5\p@

\tabcolsep 6\p@
\arrayrulewidth .4\p@
\doublerulesep 2\p@
\tabbingsep \labelsep

\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4\p@
\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}
\def\tableofcontents{%
    \section*{Contents\@mkboth{CONTENTS}{CONTENTS}}
    \@starttoc{toc}%
}
\def\l@part#1#2{%
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus \p@}%
    \begingroup
        \@tempdima 3em
        \parindent \z@
        \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \begingroup
            \large
            \bf
            \leavevmode
            #1%
            \hfil
            \hb@xt@\@pnumwidth{\hss #2}%
        \endgroup
        \par
        \nobreak
        \endgroup
}
\def\l@section#1#2{%
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus \p@}%
    \@tempdima 1.5em
    \begingroup
        \parindent \z@
        \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \bf
        \leavevmode
        #1%
        \hfil
        \hb@xt@\@pnumwidth{\hss #2}%
        \par
    \endgroup
}
\def\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}

\def\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\def\listoffigures{%
    \section*{List of Figures\@mkboth{LIST OF FIGURES}{LIST OF FIGURES}}
    \@starttoc{lof}%
}

\def\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\def\listoftables{%
    \section*{List of Tables\@mkboth{LIST OF TABLES}{LIST OF TABLES}}
    \@starttoc{lot}%
}

\let\l@table\l@figure
\def\thebibliography#1{%
    \par
    \footnotesize
    \@startsection{subsection}{2}{\z@}{16\p@ \@plus 2\p@ \@minus \p@}
        {2\p@}{\sf}*{\uppercase{\refname}}
    \list{\arabic{enumi}.}{%
        \settowidth\labelwidth{99.}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \topsep \z@
        \parsep \z@ \@plus .1\p@
        \itemsep \parsep
        \usecounter{enumi}%
    }%
    \def\newblock{\hskip .11em \@plus .33em \@minus .07em}%
    \sloppy
    \widowpenalty4500
    \clubpenalty4500
    \sfcode`\.=1000
    \relax
}

\let\endthebibliography=\endlist
\newif\if@restonecol
\def\theindex{%
    \@restonecoltrue
    \if@twocolumn \@restonecolfalse \fi
    \columnseprule \z@
    \columnsep 35\p@
    \twocolumn[\section*{Index}]
    \@mkboth{INDEX}{INDEX}%
    \thispagestyle{plain}%
    \parindent\z@
    \parskip\z@ \@plus .3\p@\relax
    \let\item\@idxitem
}

\def\endtheindex{%
    \if@restonecol
        \onecolumn
    \else
        \clearpage
    \fi
}
\def\@idxitem{\par\hangindent 40\p@}
\def\subitem{\par\hangindent 40\p@ \hspace*{20\p@}}
\def\subsubitem{\par\hangindent 40\p@ \hspace*{30\p@}}
\def\indexspace{\par \vskip 10\p@ \@plus 5\p@ \@minus 3\p@\relax}
\def\footnoterule{%
    \kern-3\p@
    \hrule \@width 47\p@ %\@height .004\p@
    \kern 2.6\p@
}
\long\def\@makefntext#1{%
    \parindent 1em
    \noindent $^{\@thefnmark}$#1%
}
\setcounter{topnumber}{2}
\def\topfraction{.89}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.5}
\setcounter{totalnumber}{3}
\def\textfraction{.11}
\def\floatpagefraction{.6}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}
\long\def\@makecaption#1#2{%
    \vskip 1pc
    \setbox\@tempboxa\hbox{#1.\hskip 1em\relax #2}%
    \ifdim \wd\@tempboxa >\hsize
        #1.\hskip 1em\relax #2\par
    \else
        \centerline{\box\@tempboxa}%
    \fi
}
\def\nocaption{%
    \refstepcounter\@captype
    \par
    \vskip 1pc
    \centerline{\footnotesize \csname fnum@\@captype \endcsname}%
}
\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\figurename{Fig.}
\def\fnum@figure{\figurename~\thefigure}
\def\figfilename{\gdef\@figfilename}
\let\@figfilename\@empty
\if@faircopy
    \newenvironment{figure}{%
        \let\normalsize\footnotesize
        \normalsize
        \@ifnextchar[{\@figure}{\@figure[]}%
    }{\end@float
        \global\setbox\@currbox\vbox to\textheight{%
            \vskip\@fptop
            \unvbox\@currbox
            \vskip\@fpbot
            \noindent \fnum@figure
            \ifx\@figfilename\@empty\else
                : {\tt \@figfilename}%
                \global\let\@figfilename\@empty
            \fi
        }%
    }
\else
    \newenvironment{figure}{%
        \let\normalsize\footnotesize
        \normalsize
        \@float{figure}
    }{\end@float}
\fi
\def\@figure[#1]{\@float{figure}[p]}
\expandafter\let\csname figure*\endcsname\figure
\expandafter\let\csname endfigure*\endcsname\endfigure
\newenvironment{table}{%
    \let\normalsize\footnotesize
    \normalsize
    \@float{table}%
}{\end@float}
\expandafter\let\csname table*\endcsname\table
\expandafter\let\csname endtable*\endcsname\endtable
\def\thetable{\@Roman\c@table}
\newcounter{table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\tablename{Table}
\def\fnum@table{\tablename~\thetable}
\newdimen\@narrowfig
\newbox\@nfigbox
\newbox\@nfcapbox
\newif\if@nfeven
\def\acmtable#1{%
    \@narrowfig #1\relax
    \let\caption\@atcap
    \let\nocaption\@atnocap
    \@ifnextchar[{\@ntab}{\@ntab[\fps@table]}%
}

\def\@ntab[#1]{%
    \expandafter\table\expandafter[#1]%
    \setbox\@nfigbox\vbox\bgroup
        \hsize \@narrowfig
        \@parboxrestore
}

\def\endacmtable{%
    \hb@xt@\textwidth{%
        \hfil
        \vbox{%
            \hsize \@narrowfig
            \box\@nfcapbox{%
                \baselineskip 4\p@
                \hbox{\vrule \@height .4\p@ \@width \hsize}%

            }%
            \vskip -\p@
            \box\@nfigbox
            \vskip -\p@
            \begingroup
                \baselineskip 4\p@
                \hbox{\vrule \@height .4\p@ \@width \hsize}%
            \endgroup
        }%
        \hfil
    }%
    \end@float
}
\def\@atmakecap#1#2{%
    \setbox\@tempboxa\hbox{#1.\hskip 1em\relax #2}%
    \ifdim \wd\@tempboxa >\hsize
        \sloppy #1.\hskip 1em\relax #2 \par
    \else
        \centerline{\box\@tempboxa}%
    \fi
}
\def\@atcap{%
        \par
    \egroup
    \refstepcounter\@captype
    \@dblarg{\@atcapx\@captype}%
}
\long\def\@atcapx#1[#2]#3{%
    \addcontentsline{\csname ext@#1\endcsname}{#1}{%
        \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}%
    }%
    \setbox\@nfcapbox\vbox{%
        \hsize \wd\@nfigbox
        \@parboxrestore
        \@atmakecap{\csname fnum@#1\endcsname}{\ignorespaces #3}%
        \par
    }%
}
\def\@atnocap{%
    \egroup
    \refstepcounter\@captype
    \setbox\@nfcapbox\vbox{%
        \hsize \wd\@nfigbox
        \centerline{\footnotesize \fnum@table}
    }%
}
\def\narrowfig#1{%
    \@narrowfig #1\relax
    \let\caption\@nfcap
    \let\nocaption\@nfnocap
    \@ifnextchar[{\@nfig}{\@nfig[\fps@figure]}%
}

\def\@nfig[#1]{%
    \expandafter\figure\expandafter[#1]%
    \setbox\@nfigbox\vbox\bgroup
        \hsize\@narrowfig
        \@parboxrestore
}

\def\endnarrowfig{%
    \hb@xt@\textwidth{%
        \if@nfeven
            \box\@nfcapbox  \hfil   \box\@nfigbox
        \else
            \box\@nfigbox   \hfil   \box\@nfcapbox
        \fi
    }%
    \endfigure
}
\def\@nfcap{%
        \par
    \egroup
    \refstepcounter\@captype
    \@dblarg{\@nfcapx\@captype}%
}
\def\@nfmakecap #1#2{%
    \setbox\@tempboxa\hbox{#1.\hskip 1em\relax #2}%
    \ifdim \wd\@tempboxa >\hsize
        \sloppy #1.\hskip 1em\relax #2 \par
    \else
        \@@line{%
            \if@nfeven\else\hfil\fi
            \box\@tempboxa
            \if@nfeven\hfil\fi
        }%
    \fi
}
\long\def\@nfcapx#1[#2]#3{%
    \addcontentsline{\csname ext@#1\endcsname}{#1}{%
        \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}%
    }%
    \@seteven
    \setbox\@nfcapbox\vbox to \ht\@nfigbox{%
        \hsize \textwidth
        \advance\hsize -2pc
        \advance\hsize -\wd\@nfigbox
        \@parboxrestore
        \vfil
        \@nfmakecap{\csname fnum@#1\endcsname}{\ignorespaces #3}%
        \par
        \vfil
    }%
}
\def\@nfnocap{%
    \egroup
    \refstepcounter\@captype
    \@seteven
    \setbox\@nfcapbox\vbox to \ht\@nfigbox{%
        \hsize \textwidth
        \advance\hsize -2pc
        \advance\hsize -\wd\@nfigbox
        \@parboxrestore
        \vfil
        \@@line{%
            \if@nfeven\else\hfil\fi
            \footnotesize \fnum@figure
            \if@nfeven\hfil\fi
        }%
        \vfil
    }%
}
% ________________________________________________________
% ACM / OCLC modifications begin
%
%\def\setbmp#1#2#3#4{\vskip#3\relax\noindent\hskip#1\relax
% \special{bmp:#4 x=#2, y=#3}}
%\def\centerbmp#1#2#3{\vskip#2\relax\centerline{\hbox to#1{\special
%  {bmp:#3 x=#1, y=#2}\hfil}}}
%
%\def\seteps#1#2#3#4{\vskip#3\relax\noindent\hskip#1\relax
% \special{eps:#4 x=#2, y=#3}}
%\def\centereps#1#2#3{\vskip#2\relax\centerline{\hbox to#1{\special
%  {eps:#3 x=#1, y=#2}\hfil}}}
%
%\def\centertif#1#2#3{\vskip#2\relax\centerline{{
%  \it{Figure {(#3)} to be inserted here.}}}}
%
% ACM / OCLC modifications end
% ________________________________________________________

\def\@seteven{%
    \@nfeventrue
    \@ifundefined{r@@nf\thefigure}{}{%
        \edef\@tmpnf{\csname r@@nf\thefigure\endcsname}%
        \edef\@tmpnf{\expandafter\@cdr\@tmpnf\@nil}%
        \ifodd\@tmpnf\relax
            \@nfevenfalse
        \fi
    }%
    \label{@nf\thefigure}%
    \edef\@tmpnfx{\if@nfeven e\else o\fi}%
    \edef\@tmpnf{%
        \write\@unused{%
            \noexpand\ifodd \noexpand\c@page
                \noexpand\if \@tmpnfx e%
                    \noexpand\@nfmsg{\thefigure}
                \noexpand\fi
            \noexpand\else
                \noexpand\if \@tmpnfx o%
                    \noexpand\@nfmsg{\thefigure}%
                \noexpand\fi
            \noexpand\fi
        }%
    }%
    \@tmpnf
}

\def\@nfmsg#1{Bad narrowfig: Figure #1 on page \thepage}
\newdimen\@narrowfig
\newbox\@nfigbox
\newbox\@nfcapbox
\newif\if@nfeven
\newenvironment{authinfo}{%
    \newcommand{\name}[1] {Name: ##1\newline }
    \newcommand{\address}[1] {Address: ##1\newline }
    \newcommand{\affiliation}[1]  {Affiliation: ##1\newline }
    \newcommand{\biography}[1]  {Biography: ##1\newline}
}{}
%
% _________________________________
% ACM / OCLC modifications
\newenvironment{sponsor}[2]{%
    Sponsor: #1\strut\newline
    \parbox{3.in}{#2}%
}{\newline}
\def\and{\\ and\\}
% _________________________________

\def\maketitle{%
    \newpage
    \thispagestyle{titlepage}%
    \global\@topnum\z@
    \begingroup
        \lineskip \z@
        \null
        \vskip -30\p@\relax
        \parindent\z@
        \begingroup
            \LARGE
            \raggedright
            \hyphenpenalty\@M
            \sf \@title
            \par
        \endgroup
        \vskip 13.5\p@
        %vskip 13.5\p@ between title and author
        \begingroup

{
\addtolength{\baselineskip}{2\p@}%
            \sf
            \@author\par
    \ifx \@sponsors\@empty
    \else
        %\baselineskip 17\p@%\relax
        \hbox{\vrule height .2\p@ width 30pc}
        \@sponsors \par
    \fi
}
\endgroup
        \begingroup
            \footnotesize
            \medskip
            \rule{\textwidth}{.2\p@}%
            \ifvoid\@abstract
                \typeout{^^JPackage acmart Warning: an abstract
                         is required^^J}%
            \else
                \vskip 8.5\p@
                \box\@abstract
            \fi


            \ifx\@categories\@empty \else
                \vskip 4\p@\relax %vskip8.5 space above abstract
                \def\and{\unskip{\rm;} }%
                Categories and Subject Descriptors: \@categories\par
            \fi
            \ifx\@terms\@empty\else
                \vskip 4\p@
                General Terms: \ignorespaces
                \@terms
                \par
            \fi
            \ifx\@keywords\@empty\else
                \vskip 4\p@
                Additional Key Words and Phrases: \ignorespaces
                \@keywords
                \par
            \fi
        \endgroup
        \label{@firstpg}%
        \rule{\textwidth}{.2\p@}%
        \vskip 23\p@
    \endgroup
    \let\maketitle\relax
    \global\let\@sponsors\@empty
    \global\let\@categories\@empty
    \global\let\@terms\@empty
    \global\let\@keywords\@empty
}

\def\sponsor#1{\@ifnextchar
 [{\@sponsor{#1}}{\@xsponsor{#1}}}
\def\@sponsor#1[#2]{\edef\@tempa{\ifx \@sponsors\@empty
 \else ; \fi}{\def\protect{\noexpand\protect
 \noexpand}\def\and{\noexpand\and}\xdef\@sponsors{\@sponsors\@tempa #1 }}}
\def\@xsponsor#1{\edef\@tempa{\ifx \@sponsors\@empty \else ;
\fi}{\def\protect{\noexpand\protect\noexpand}\def\and{\noexpand
 \and}\xdef\@sponsors{\@sponsors\@tempa #1}}}
\def\@sponsors{}


\newbox\@abstract
\def\abstract{%
    \everypar{}%
    \global\setbox\@abstract\vbox\bgroup
        \footnotesize
        \trivlist\item[]\ignorespaces
}

\def\endabstract{%
        \endtrivlist
    \egroup
    \ifx\maketitle\relax
       \typeout{^^JPackage acmart Warning:
            the abstract must be specified before the^^J%
            \string\maketitle\space command^^J%
        }%
        \box\@abstract
    \fi
}
\def\terms{\gdef\@terms}

\let\@terms\@empty
\def\keywords{\gdef\@keywords}

\let\@keywords\@empty
\def\category#1#2#3{%
    \@ifnextchar [{\@category{#1}{#2}{#3}}{\@category{#1}{#2}{#3}[]}%
}
\def\@category#1#2#3[#4]{%
    \edef\@tempa{\ifx \@categories\@empty \else ; \fi}%
    \begingroup
        \def\protect{\noexpand\protect\noexpand}%
        \let\and\relax
        \xdef\@categories{%
            \@categories
            \@tempa
            #1 [{\bf #2}]%
            \if!#4!%
                \if!#3!\else : #3\fi
            \else
                :\space
                \if!#3!\else #3\kern\z@---\hskip\z@\fi
                {\it #4}%
            \fi
        }%
    \endgroup
}

\let\@categories\@empty
\def\bottomstuff{%
    \global\@topnum\z@
    \global\@botroom \textheight
    \@float{figure}[b]
        \footnotesize
        \parindent\z@
        \null
        \vskip -\textfloatsep
        \vskip 10\p@
        \hrule \@height .2\p@ \@width 30pc
        \vskip 4\p@
        \ignorespaces
}
\def\endbottomstuff{%
    \par
    \strut
    \end@float
}
\long\def\permission{\par
    Permission to make digital or hard copies of part or all of this
    work for personal or classroom use is granted without fee provided
    that copies are not made or distributed for profit or direct
    commercial advantage and that copies show this notice on the first
    page or initial screen of a display along with the full citation.
    Copyrights for components of this work owned by others than ACM
    must be honored. Abstracting with credit is permitted. To copy
    otherwise, to republish, to post on servers, to redistribute to
    lists, or to use any component of this work in other works,
    requires prior specific permission and/or a fee.  Permissions may
    be requested from Publications Dept, ACM Inc., 1515 Broadway, New
    York, NY 10036 USA, fax +1 (212) 869-0481, or {\tt
    permissions@acm.org}.
    \par
}
% _______________________________________________________
%
%  Original acks did not sans serif heading at all.
%
%\newenvironment{acksold}{%
%    \@startsection{subsection}{2}{\z@}{-16\p@ \@plus -2\p@ \@minus -\p@}
%        {2\p@}{\sf}*{\footnotesize ACKNOWLEDGMENTS}%
%}{\par}
%
%
%  ACM / OCLC modification to sans serif the heading.
%
\newenvironment{acks}{%
    \@startsection{subsection}{2}{\z@}{-16\p@ \@plus -2\p@ \@minus -\p@}
        {2\p@}{}*{\footnotesize\sf ACKNOWLEDGMENTS}%
}{\par}
%
% _____________________________________________________
%
\newenvironment{received}{%
    \par
    \footnotesize
    \addvspace{18\p@ \@minus 4\p@}%
    \parindent\z@
}{%
    \label{@lastpg}%
    \par
}
\mark{{}{}}
\def\ps@myheadings{%
    \let\@mkboth\@gobbletwo
    \def\@oddhead{%
        \hfill\small\sf\rightmark
        \hskip 19\p@
        {\Large$\cdot$}%
        \hskip 17\p@
        \thepage
    }%
    \def\@oddfoot{\null\hfill\tiny\@runningfoot}
    \def\@evenhead{%
        \small\sf\thepage
        \hskip 17\p@
        {\Large$\cdot$}%
        \hskip 19\p@
        \leftmark
        \hfill
    }%
    \def\@evenfoot{\tiny\@runningfoot \hfill\null}%
    \let\partmark\@gobble
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
}
\def\runningfoot#1{\def\@runningfoot{#1}}

\let\@runningfoot\@empty

\def\firstfoot#1{\def\@firstfoot{#1}}

\let\@firstfoot\@empty

\def\pages{\pageref{@firstpg}--\pageref{@lastpg}}
\def\ps@titlepage{%
    \let\@mkboth\@gobbletwo
    \let\@oddhead\@empty
    \def\@oddfoot{\null\hfill\tiny\@firstfoot}%
    \let\@evenhead\@empty
    \def\@evenfoot{\tiny\@firstfoot\hfill\null}%
}
\def\today{%
    \ifcase\month \or
    January\or
    February\or
    March\or
    April\or
    May\or
    June\or
    July\or
    August\or
    September\or
    October\or
    November\or
    December
    \fi
    \space\number\day, \number\year
}
\def\marray{%
    \arraycolsep 2.5\p@
    \let\@acol\@arrayacol
    \let\@classz\@marrayclassz
    \let\@classiv\@marrayclassiv
    \let\\\@arraycr
    \let\@halignto\@empty
    \@tabarray
}

\let\endmarray\endarray
\def\@marrayclassiv{\@addtopreamble{$\displaystyle \@nextchar$}}

\def\@marrayclassz{%
    \ifcase \@lastchclass
    \@acolampacol
    \or \@ampacol
    \or
    \or
    \or \@addamp
    \or \@acolampacol
    \or \@firstampfalse \@acol
    \fi
    \edef\@preamble{%
        \@preamble
        \ifcase \@chnum
            \hfil$\relax\displaystyle\@sharp$\hfil
            \or $\relax\displaystyle\@sharp$\hfil
            \or \hfil$\relax\displaystyle\@sharp$%
        \fi
    }%
}
\ps@myheadings

\pagenumbering{arabic}

\onecolumn


\parindent10\p@
\typeout{Using ACM bibliography style}
\newcommand\bibyear[2]{%
    \unskip\quad\ignorespaces#1\unskip
    \if#2..\quad \else \quad#2 \fi
}
\newcommand{\bibemph}[1]{{\em#1}}
\newcommand{\bibemphic}[1]{{\em#1\/}}
\newcommand{\bibsc}[1]{{\sc#1}}
\def\@normalcite{%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
}
\def\@citeNB{%
    \def\@cite##1##2{##1\if@tempswa , ##2\fi}%
}
\def\@citeRB{%
    \def\@cite##1##2{##1\if@tempswa , ##2\fi]}%
}
\def\start@cite#1#2{%
    \edef\citeauthoryear##1##2##3{%
        ###1%
        \ifnum#2=\z@ \else\ ###2\fi
    }%
    \ifnum#1=\thr@@
        \let\@@cite\@citeyear
    \else
        \let\@@cite\@citenormal
    \fi
    \@ifstar{\@citeNB\@@cite}{\@normalcite\@@cite}%
}
\def\cite{\start@cite23}
\def\citeNP{\cite*}
\def\citeA{\start@cite10}
\def\citeANP{\citeA*}
\def\shortcite{\start@cite23}
\def\shortciteNP{\shortcite*}
\def\shortciteA{\start@cite20}
\def\shortciteANP{\shortciteA*}
\def\citeyear{\start@cite30}
\def\citeyearNP{\citeyear*}
\def\citeN{%
    \@citeRB
    \def\citeauthoryear##1##2##3{##1\ [##3%
        \def\reserved@a{##1}%
        \def\citeauthoryear####1####2####3{%
            \def\reserved@b{####1}%
            \ifx\reserved@a\reserved@b
                ####3%
            \else
                \errmessage{Package acmart Error: author mismatch
                         in \string\citeN^^J^^J%
                    See the acmart package documentation for explanation}%
            \fi
        }%
    }%
    \@ifstar\@citeyear\@citeyear
}
\def\shortciteN{%
    \@citeRB
    \def\citeauthoryear##1##2##3{##2\ [##3%
        \def\reserved@a{##2}%
        \def\citeauthoryear####1####2####3{%
            \def\reserved@b{####2}%
            \ifx\reserved@a\reserved@b
                ####3%
            \else
                \errmessage{Package acmart Error: author mismatch
                         in \string\shortciteN^^J^^J%
                    See the acmart package documentation for explanation}%
            \fi
        }%
    }%
    \@ifstar\@citeyear\@citeyear
}
\def\@citenormal{%
    \@ifnextchar [{\@tempswatrue\@citex;}
                  {\@tempswafalse\@citex;[]}%
}
\def\@citeyear{%
    \@ifnextchar [{\@tempswatrue\@citex,}%
                  {\@tempswafalse\@citex,[]}%
}
\def\@citex#1[#2]#3{%
    \let\@citea\@empty
    \@cite{%
        \@for\@citeb:=#3\do{%
            \@citea
            \def\@citea{#1 }%
            \edef\@citeb{\expandafter\@iden\@citeb}%
            \if@filesw
                \immediate\write\@auxout{\string\citation{\@citeb}}%
            \fi
            \@ifundefined{b@\@citeb}{%
                {\bf ?}%
                \@warning{%
                    Citation `\@citeb' on page \thepage\space undefined%
                }%
            }%
            {\csname b@\@citeb\endcsname}%
        }%
    }{#2}%
}
\let\@biblabel\@gobble
\newdimen\bibindent
\bibindent=1.5em
\def\refname{References}
\def\thebibliography#1{%
    \footnotesize
    \section*{%
       {\footnotesize\sf REFERENCES}
        \@mkboth{\uppercase{\refname}}{\uppercase{\refname}}%
    }%
    \list{[\arabic{enumi}]}{%
        \settowidth\labelwidth{[#1]}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \advance\leftmargin\bibindent
        \itemindent -\bibindent
        \listparindent \itemindent
        \parsep \z@
        \usecounter{enumi}%
    }%
    \let\newblock\@empty
    \sloppy
    \sfcode`\.=1000\relax
}
\endinput

%
%%
%% End of file `esub2acm.cls' Version 1.2 - Gerry Murray ACM November 10th. 1999

