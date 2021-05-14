define help

Supported targets: prepare, develop, sdist, clean, test, and pypi.

The 'prepare' target installs Epitome's build requirements into the current virtualenv.

The 'develop' target creates an editable install of Epitome and its runtime requirements in the
current virtualenv. The install is called 'editable' because changes to the source code
immediately affect the virtualenv.

The 'clean' target undoes the effect of 'develop'.

The 'test' target runs Epitome's unit tests. Set the 'tests' variable to run a particular test, e.g.

        make test tests=epitome/test/models_test.py:test

The 'pypi' target publishes the current commit of Epitome to PyPI after enforcing that the working
copy and the index are clean, and tagging it as an unstable .dev build.

endef
export help
help:
	@printf "$$help"

SHELL=bash
python=python
pip=pip
tests=epitome
extras=
epitome_version:=$(shell $(python) version.py)
sdist_name:=epitome-$(epitome_version).tar.gz
current_commit:=$(shell git log --pretty=oneline -n 1 -- $(pwd) | cut -f1 -d " ")
dirty:=$(shell (git diff --exit-code && git diff --cached --exit-code) > /dev/null || printf -- --DIRTY)

green=\033[0;32m
normal=\033[0m\n
red=\033[0;31m

prepare:
	$(pip) install -r requirements.txt

develop:
	$(pip) install -e .$(extras)
clean_develop:
	- $(pip) uninstall -y epitome
	- rm -rf src/*.egg-info
sdist: dist/$(sdist_name)
dist/$(sdist_name):
	@test -f dist/$(sdist_name) && mv dist/$(sdist_name) dist/$(sdist_name).old || true
	$(python) setup.py sdist bdist_egg
	@test -f dist/$(sdist_name).old \
            && ( cmp -s <(tar -xOzf dist/$(sdist_name)) <(tar -xOzf dist/$(sdist_name).old) \
                 && mv dist/$(sdist_name).old dist/$(sdist_name) \
                 && printf "$(green)No significant changes to sdist, reinstating backup.$(normal)" \
                 || rm dist/$(sdist_name).old ) \
            || true

clean_sdist:
	- rm -rf dist
sdist:

clean: clean_develop clean_pypi

check_build_reqs:
	@$(python) -c 'import pytest' \
                || ( printf "$(redpip)Build requirements are missing. Run 'make prepare' to install them.$(normal)" ; false )

test: check_build_reqs
	$(python) -m pytest -vv --junitxml target/pytest-reports/tests.xml $(tests)

check_clean_working_copy:
	@printf "$(green)Checking if your working copy is clean ...$(normal)"
	@git diff --exit-code > /dev/null \
                || ( printf "$(red)Your working copy looks dirty.$(normal)" ; false )
	@git diff --cached --exit-code > /dev/null \
                || ( printf "$(red)Your index looks dirty.$(normal)" ; false )

pypi: clean clean_sdist check_clean_working_copy
	set -x \
	&& $(python) setup.py egg_info sdist bdist_egg \
	&& twine check dist/* \
	&& twine upload dist/*
clean_pypi:
	- rm -rf build/
