"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os


def get_module_path():
    """ Get module path
    Returns:
        path (str): path to current module.
    """
    file_path = os.path.abspath(__file__)
    module_path = os.path.dirname(file_path)
    return module_path


def res_path(path):
    """ Resource path
    Arguments:
        path (str): related path from module dir to some resources.
    Returns:
        path (str): absolute path to module dir.
    """
    return os.path.join(get_module_path(), path)
