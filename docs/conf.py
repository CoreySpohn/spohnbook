"""Sphinx configuration."""

import sys
from importlib.metadata import version as get_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_map import write_map_table
from handbook import write_handbook_pages

project = "The Spohn Book"
release = get_version("spohnbook")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
exclude_patterns = ["_build", "_generated/*.md", "_evidence"]

html_theme = "sphinx_book_theme"
html_title = "The Spohn Book"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/spohnbook",
    "use_repository_button": True,
}
html_context = {"default_mode": "light"}

nb_execution_mode = "auto"
nb_execution_timeout = 900
nb_execution_raise_on_error = True

write_map_table(Path(__file__).parent)
# Validates the catalog (raising on any structural error) and renders the
# references, profile, edition, environment and evidence fragments.
write_handbook_pages(Path(__file__).parent)
