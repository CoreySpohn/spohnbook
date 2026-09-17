"""Sphinx configuration."""

import sys
from importlib.metadata import version as get_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_map import write_map_table

project = "The Spohn Book"
release = get_version("spohnbook")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
exclude_patterns = ["_build", "_generated/*.md"]

html_theme = "sphinx_book_theme"
html_title = "The Spohn Book"
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/spohnbook",
    "use_repository_button": True,
}
html_context = {"default_mode": "light"}

nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

write_map_table(Path(__file__).parent)
