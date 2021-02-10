#!/bin/bash
cd {remote_results_path}/{name}/{version}/src

cd {repository}
pip install -e .
{cmd}
