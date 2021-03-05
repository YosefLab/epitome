#!/bin/sh

set -e -x

# make sure gh is installed
if ! command -v gh &> /dev/null
then
    echo "github client (gh) needs to be installed to make a release. See https://cli.github.com/"
    exit
fi

# do we have enough arguments?
if [ $# -lt 2 ]; then
    echo "Usage:"
    echo
    echo "./release.sh <release version> <development version>"
    exit 1
fi

# get current version
current_version=$(python version.py)

# get current branch
branch=$(git status -bs | awk '{ print $2 }' | awk -F'.' '{ print $1 }' | head -n 1)

# clean up if something goes wrong
function clean_up {

    find . -name "*.bak" -exec rm -f {} \;
    git checkout version.py
    git checkout ${branch}
    git branch -D ${release}
}
trap clean_up EXIT

# pick arguments
release=$1
devel=$2

# checkout release
git checkout -b ${release} ${branch}

# update current version
find . -name "version.py" -exec sed -e "s/${current_version}/${release}/g" \
    -i.${current_version}.bak '{}' \;

find . -name "*${current_version}.bak" -exec rm -f {} \;

# commit version changes
git add version.py
# allow empty in case version was already release version (mainly for pre-releases)
git commit --allow-empty -m "bumped version from ${current_version} to release version ${release}"

# build sdist and push to pypi
pip install twine
make pypi
if [ $? != 0 ]; then
  echo "Releasing epitome to PyPi failed."
  exit 1
fi

push branch to upstream
git push upstream ${release}

# update version to devel
current_version=$(python version.py)
find . -name "version.py" -exec sed -e "s/${current_version}/${devel}/g" \
    -i.${release}.bak '{}' \;

find . -name "*${release}.bak" -exec rm -f {} \;

commit version changes
git add version.py
git commit -m "bumped version from ${release} to ${devel}"

pull request devel to master
git push origin ${release}

# if prompted, push to YOUR remote (not YosefLab/epitome)!
gh pr create --title "v${devel}" \
        --body "bumped version from ${release} to ${devel}" \
        --repo YosefLab/epitome

git checkout master
echo "Done. Now tag a release on github for ${release}"
