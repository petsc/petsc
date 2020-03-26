""" Sphinx extension for custom HTML processing for PETSc docs """

from typing import Any, Dict
import re
import os
import subprocess

from docutils import nodes
from docutils.nodes import Element, Text

from sphinx import version_info as sphinx_version_info
from sphinx.writers.html5 import HTML5Translator
from sphinx.application import Sphinx


def setup(app: Sphinx) -> None:
    _check_version(app)
    app.set_translator('html', HTML5PETScTranslator, override=True)

    # Also set the translator for ReadTheDocs's custom builder
    # This is dangerous, since they could change the name and silently
    # deactivate our translator
    app.set_translator('readthedocs', HTML5PETScTranslator, override=True)


def _check_version(app: Sphinx) -> None:
    app.require_sphinx('2.4')
    sphinx_version_info_source = (2, 4, 4, 'final', 0)
    if sphinx_version_info != sphinx_version_info_source:
        error_message = 'This extension duplicates code from Sphinx '
        error_message += str(sphinx_version_info_source)
        error_message += ' which is incompatible with the current version'
        error_message += str(sphinx_version_info)
        raise NotImplementedError(error_message)


class HTML5PETScTranslator(HTML5Translator):
    """
    A custom HTML5 translator which overrides methods to add PETSc-specific
    custom processing to the generated HTML.
    """

    def __init__(self, *args: Any) -> None:
        self._manpage_map = None
        self._manpage_pattern = None
        super().__init__(*args)


    def _get_manpage_map(self) -> Dict[str,str]:
        """ Return the manpage strings to link, as a dict.

        This may involve generating or reading from a file, so may be slow.

        This is done lazily, so this function should always be used,
        instead of the direct data member, which may not be populated yet
        """
        if not self._manpage_map:
            htmlmap_stash_filename = _generate_htmlmap_stash()
            manpage_map_raw = htmlmap_to_dict(htmlmap_stash_filename)
            manpage_prefix = 'https://www.mcs.anl.gov/petsc/petsc-current/docs/'
            self._manpage_map = dict_complete_links(manpage_map_raw, manpage_prefix)
        return self._manpage_map

    def _get_manpage_pattern(self) -> re.Pattern:
        """ Return the manpage links pattern.

        This is done lazily, so this function should always be used,
        instead of the direct data member, which may not be populated yet
        """

        if not self._manpage_pattern:
            self._manpage_pattern = get_multiple_replace_pattern(self._get_manpage_map())
        return self._manpage_pattern

    def _add_manpage_links(self, string: str) -> str:
        """ Add plain HTML link tags to a string """
        manpage_map = self._get_manpage_map()
        manpage_pattern = self._get_manpage_pattern()
        return replace_from_dict_and_pattern(string, manpage_map, manpage_pattern)

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
        if lang is self.builder.config.highlight_language:
            # only pass highlighter options for original language
            opts = self.builder.config.highlight_options
        else:
            opts = {}

        highlighted = self.highlighter.highlight_block(
            node.rawsource, lang, opts=opts, linenos=linenos,
            location=(self.builder.current_docname, node.line), **highlight_args
        )
        starttag = self.starttag(node, 'div', suffix='',
                                 CLASS='highlight-%s notranslate' % lang)

        # Custom processing to add links to PETSc man pages ####################
        highlighted = self._add_manpage_links(highlighted)
        # (end of custom processing) ###########################################

        self.body.append(starttag + highlighted + '</div>\n')
        raise nodes.SkipNode

def htmlmap_to_dict(htmlmap_filename: str) -> Dict[str,str]:
    """ Extract a dict from an htmlmap file, leaving URLs as they are."""
    pattern = re.compile(r'man:\+([a-zA-Z_0-9]*)\+\+([a-zA-Z_0-9 .:]*)\+\+\+\+man\+([a-zA-Z_0-9#./:-]*)')
    string_to_link = dict()
    with open(htmlmap_filename, 'r') as f:
        for line in f.readlines():
            m = re.match(pattern, line)
            if m:
                string = m.group(1)
                string_to_link[string] = m.group(3)
            else:
                print("Warning: skipping unexpected line in " + htmlmap_filename + ":")
                print(line)
    return string_to_link


def dict_complete_links(string_to_link: Dict[str,str], prefix: str = '') -> Dict[str,str]:
    """ Complete HTML links

    Prepend a prefix to any links not starting with 'http',
    and add HTML tags
    """
    def link_string(name: str, link: str, prefix: str) -> str:
        url = link if link.startswith('http') else prefix + link
        return '<a href=\"' + url + '\">' + name + '</a>'
    return dict((k, link_string(k, v, prefix)) for (k, v) in string_to_link.items())


def _generate_htmlmap_stash() -> str:
    """ Perform a minimal PETSc configuration and docs build to obtain htmlmap

    This may be quite slow (on the order of ~5 minutes), as
    we configure and build PETSc and traverse the source.
    """
    htmlmap_stash_filename = os.path.join('_build', 'docs', 'manualpages', 'htmlmap')
    if not os.path.isfile(htmlmap_stash_filename):
        petsc_dir = os.path.abspath(os.path.join('..', '..', '..'))
        configure = [
                './configure',
                '--with-mpi=0',
                '--with-blaslapack=0',
                '--with-fortran=0',
                '--with-cxx=0',
                '--with-x=0',
                '--with-cmake=0',
                '--with-pthread=0',
                '--with-regexp=0',
                '--download-sowing',
                '--with-mkl_sparse_optimize=0',
                '--with-mkl_sparse=0',
                'PETSC_ARCH=arch-sphinxdocs-minimal'
                ]
        status = subprocess.run(configure, cwd = petsc_dir).check_returncode()
        docs_destination = os.path.join(os.getcwd(),'_build')
        allcite = ['make', 'PETSC_DIR='+petsc_dir, 'allcite', 'LOC='+docs_destination]
        subprocess.run(allcite, cwd=petsc_dir).check_returncode()
    return htmlmap_stash_filename


def get_multiple_replace_pattern(source_dict: Dict[str,str]) -> re.Pattern:
    """ Generate a regex to match any of the keys in source_dict, as full words """
    def process_word(word):
        """ add escape characters and word boundaries """
        return r'\b' + re.escape(word) + r'\b'
    return re.compile(r'|'.join(map(process_word, source_dict)))


def replace_from_dict_and_pattern(string: str, replacements: Dict, pattern: re.Pattern) -> str:
    """ Given a pattern which matches keys in replacements, replace keys found in string with their values"""
    return pattern.sub(lambda match: replacements[match.group(0)], string)
