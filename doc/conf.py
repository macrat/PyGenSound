import os
import sys
sys.path.insert(0, os.path.abspath('..'))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

source_suffix = '.rst'
master_doc = 'index'


project = 'PyGenSound'
copyright = '2017, MacRat'
author = 'MacRat'

version = '0.0.1'
release = '0.0.1 dev'


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'alabaster'
html_theme_options = {
    'description': 'The python library for generating sound and compute.',
    'github_user': 'macrat',
    'github_repo': 'PyGenSound',
    'github_banner': True,
    'github_button': False,
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}


intersphinx_mapping = {'https://docs.python.org/': None}


autodoc_member_order = 'bysource'
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']
