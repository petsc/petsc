""" Sphinx extension for custom HTML processing for PETSc docs """

from typing import Any, Dict
import re
import os
import types

from docutils import nodes
from docutils.nodes import Element, Text

from sphinx import version_info as sphinx_version_info
from sphinx.writers.html5 import HTML5Translator
from sphinx.application import Sphinx

if not hasattr(re,'Pattern'): re.Pattern = re._pattern_type


PETSC_DOC_OUT_ROOT_PLACEHOLDER = 'PETSC_DOC_OUT_ROOT_PLACEHOLDER'

def setup(app: Sphinx) -> None:
    _check_version(app)

    app.connect('builder-inited', _setup_translators)
    return {'parallel_read_safe': True}


def _check_version(app: Sphinx) -> None:
    sphinx_version_info_source = (4, 2, 0, 'final', 0)
    app.require_sphinx('%s.%s' % (sphinx_version_info_source[0], sphinx_version_info_source[1]))
    if sphinx_version_info[:2] != sphinx_version_info_source[:2]:
        print('A custom extension duplicates code from Sphinx %s ' % (sphinx_version_info_source,),
              'which differs from the current version %s' % (sphinx_version_info,),
              'so unexpected behavior may be observed.')


def _setup_translators(app: Sphinx) -> None:
    """ Use a mixin strategy to add to the Sphinx HTML translator without overriding

    This allows use of other extensions which modify the translator.

    Duplicates the approach used here in sphinx-hoverref:
    https://github.com/readthedocs/sphinx-hoverxref/pull/42
    """
    if app.builder.format != 'html':
        return

    for name, klass in app.registry.translators.items():
        translator = types.new_class(
            'PETScHTMLTranslator',
            (
                PETScHTMLTranslatorMixin,
                klass,
            ),
            {},
        )
        app.set_translator(name, translator, override=True)

    translator = types.new_class(
        'PETScHTMLTranslator',
        (
            PETScHTMLTranslatorMixin,
            app.builder.default_translator_class,
        ),
        {},
    )
    app.set_translator(app.builder.name, translator, override=True)


class PETScHTMLTranslatorMixin:
    """
    A custom HTML translator which overrides methods to add PETSc-specific
    custom processing to the generated HTML.

    Replaces any string XXX that matches a manual page name with
    <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/manualpages/YY/XXX.html">XXX</a>
    or
    <a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/manualpages/YY/XXX">XXX</a>
    depending on if the Sphinx build is html or dirhtml
    """

    def __init__(self, *args: Any) -> None:
        self._manpage_map = None
        self._word_pattern = re.compile('\w+')
        super().__init__(*args)


    def _get_manpage_map(self) -> Dict[str,str]:
        """ Return the manpage strings to link, as a dict.  """
        if not self._manpage_map:
            htmlmap_filename = os.path.join('manualpages', 'htmlmap')
            if not os.path.isfile(htmlmap_filename):
                raise Exception("Expected file %s not found. Run script to build classic docs subset." %  htmlmap_filename)
            manpage_map_raw = htmlmap_to_dict(htmlmap_filename)
            manpage_prefix_base = PETSC_DOC_OUT_ROOT_PLACEHOLDER
            manpage_prefix = os.path.join(manpage_prefix_base, '')
            self._manpage_map = dict_complete_links(manpage_map_raw, manpage_prefix)
        return self._manpage_map


    def _add_manpage_links(self, string: str) -> str:
        """ Add plain HTML link tags to a string """
        manpage_map = self._get_manpage_map()
        def replace(matchobj):
            word = matchobj.group(0)
            if word in manpage_map:
                return manpage_map[word]
            return word

        return self._word_pattern.sub(replace, string)


    # This method consists mostly of code duplicated from Sphinx:
    # overwritten
    def visit_Text(self, node: Text) -> None:
        text = node.astext()
        encoded = self.encode(text)
        if self.protect_literal_text:
            # moved here from base class's visit_literal to support
            # more formatting in literal nodes
            for token in self.words_and_spaces.findall(encoded):
                if token.strip():
                    # Custom processing to add links to PETSc man pages ########
                    token_processed = self._add_manpage_links(token)

                    # protect literal text from line wrapping
                    self.body.append('<span class="pre">%s</span>' % token_processed)
                    # (end of custom processing) ###############################
                elif token in ' \n':
                    # allow breaks at whitespace
                    self.body.append(token)
                else:
                    # protect runs of multiple spaces; the last one can wrap
                    self.body.append('&#160;' * (len(token) - 1) + ' ')
        else:
            if self.in_mailto and self.settings.cloak_email_addresses:
                encoded = self.cloak_email(encoded)
            self.body.append(encoded)

    # This method consists mostly of code duplicated from Sphinx:
    # overwritten
    def visit_literal_block(self, node: Element) -> None:
        if node.rawsource != node.astext():
            # most probably a parsed-literal block -- don't highlight
            return super().visit_literal_block(node)

        lang = node.get('language', 'default')
        linenos = node.get('linenos', False)
        highlight_args = node.get('highlight_args', {})
        highlight_args['force'] = node.get('force', False)
        opts = self.config.highlight_options.get(lang, {})

        if linenos and self.config.html_codeblock_linenos_style:
            linenos = self.config.html_codeblock_linenos_style

        highlighted = self.highlighter.highlight_block(
            node.rawsource, lang, opts=opts, linenos=linenos,
            location=node, **highlight_args
        )

        # Custom processing to add links to PETSc man pages ####################
        highlighted = self._add_manpage_links(highlighted)
        # (end of custom processing) ###########################################

        starttag = self.starttag(node, 'div', suffix='',
                                 CLASS='highlight-%s notranslate' % lang)
        self.body.append(starttag + highlighted + '</div>\n')
        raise nodes.SkipNode

    # This method consists mostly of code duplicated from Sphinx:
    # overwritten to remove from CLASS literal that caused an ugly extra pre and post space
    # in the manual pages for all PETSc links
    def visit_literal(self, node: Element) -> None:
        if 'kbd' in node['classes']:
            self.body.append(self.starttag(node, 'kbd', '',
                                           CLASS='docutils notranslate'))
            return
        lang = node.get("language", None)
        if 'code' not in node['classes'] or not lang:
            self.body.append(self.starttag(node, 'code', '',
                                           CLASS='docutils notranslate'))
            self.protect_literal_text += 1
            return

        opts = self.config.highlight_options.get(lang, {})
        highlighted = self.highlighter.highlight_block(
            node.astext(), lang, opts=opts, location=node, nowrap=True)
        starttag = self.starttag(
            node,
            "code",
            suffix="",
            CLASS="docutils literal highlight highlight-%s" % lang,
        )
        self.body.append(starttag + highlighted.strip() + "</code>")
        raise nodes.SkipNode

def htmlmap_to_dict(htmlmap_filename: str) -> Dict[str,str]:
    """ Extract a dict from an htmlmap file, leaving URLs as they are."""
    with open(htmlmap_filename, 'r') as f:
        lines = [l for l in f.readlines() if l.startswith('man:')]
    string_to_link = dict()
    pattern        = re.compile(r'man:\+([a-zA-Z_0-9]*)\+\+([a-zA-Z_0-9 .:]*)\+\+\+\+man\+([a-zA-Z_0-9#./:-]*)')
    for line in lines:
        m = pattern.match(line)
        if m:
            string_to_link[m.group(1)] = m.group(3)
        else:
            print("Warning: skipping unexpected line in " + htmlmap_filename + ":")
            print(line)
    return string_to_link


def dict_complete_links(string_to_link: Dict[str,str], prefix: str = '') -> Dict[str,str]:
    """
    Prepend a prefix to any links not starting with 'http' so Sphinx will recognize them as URLs
    """
    def link_string(name: str, link: str, prefix: str) -> str:
        url = link if link.startswith('http') else prefix + link
        return '<a href=\"' + url + '\">' + name + '</a>'
    return dict((k, link_string(k, v, prefix)) for (k, v) in string_to_link.items())
