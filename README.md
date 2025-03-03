# Base branch for Research

This is the base branch for implementing new research methods.

The implementation should proceed as follows:
1. Implement the test code that the new method must pass under `tests/`.
2. Implement the new method under `src/`, focusing on modularity.
3. Ensure that the new method passes the tests.

Our research group is preparing a codebase for large-scale experiments using LLMs.
The new methods to be implemented must be compatible with this codebase.
The guideline for maintaining compatibility with the codebase is to base it on `reference/train.py`.
This is a script to train a small language model, and if the new method introduced into this script can be executed, it is guaranteed to be compatible with our codebase.
Therefore, the test code must consider the implementation of the module as input and regard the training process, which applies the module to a part of `reference/train.py`, as passing the test if it completes correctly.