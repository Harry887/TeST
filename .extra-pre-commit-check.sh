#!/bin/bash

cmd=${1}
shift 1

# check README.md exists
if [[ ${cmd} == readme ]]; then
    if [[ -z `git ls-files -c | grep ^README\.md$` ]]; then
        echo "README.md not found!!!"
        exit 1
    fi
# check requirements.txt exists
elif [[ ${cmd} == requirements ]]; then
    if [[ -z `git ls-files -c | grep ^requirements\.txt$` ]]; then
        echo "requirements.txt not found!!!"
        exit 1
    fi
# perform isort first, then black
# pre-commit will fail if any file changes after these two steps
elif [[ ${cmd} == isort-black ]]; then
    #if [[ -f .isort.cfg ]]; then
    #    cp .isort.cfg .isort.cfg.bak
    #fi
    # seed-isort-config
    # isort ${@}
    black ${@} --line-length 120

    #if [[ -f .isort.cfg.bak ]]; then
    #    mv .isort.cfg.bak .isort.cfg
    #else
    #    rm .isort.cfg
    #fi
fi
