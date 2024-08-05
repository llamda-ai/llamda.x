import os
import subprocess

import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, List

from src.llamda import Llamda


dir_path: Path = Path(os.getcwd()) / "user_llamdas"


def get_file_path(name: str) -> Path:
    """
    Get the path to the file with the given name.
    """
    file_path: Path = dir_path / f"{name}.py"
    return file_path


def get_file(name: str) -> str:
    """
    Get the content of the file with the given name.
    """
    file_path: Path = get_file_path(name)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    return file_path.read_text(encoding="utf-8")


def get_functions() -> List[str]:
    """
    Get the list of functions in the user_llamdas directory.
    """
    return [f.name for f in dir_path.iterdir() if f.suffix == ".py"]


def install_dependencies(dependencies: List[str]) -> str:
    """
    Install the dependencies for your function.

    Args:
        dependencies: A list of dependencies to install, ie ["pydantic"]

    """
    os.chdir(dir_path)
    # install the dependencies
    subprocess.check_call(
        [sys.executable, "-m", "poetry", "add", "llamda", *dependencies]
    )
    os.chdir("..")
    return f"Dependencies installed: {dependencies}"


def create_function(ll: Llamda) -> Callable[[str, str, List[str], str], str]:
    @ll.fy(name="create_function")
    def create_function(
        name: str,
        description: str,
        imports: List[str],
        function_body: str,
    ) -> str:
        """

        Create a function file with the given name, description, imports, arguments and body.

        Once you do that, the functions you write will be added to your own list of tools.

        Args:
            name: The name of the function.
            description: A description of the function.
            imports: A list of import statements for your function
                ie ["import os","from pydantic import BaseModel"]
            function_body: The body of the function
                ie def my_function(arg1: str, arg2: int) -> str: return arg1
        """

        file_path: Path = get_file_path(f"{name}_stub")
        imports_str = "\n".join(imports)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{imports_str}\n\n{function_body}")
            f.close()
        try:
            stub_module: ModuleType = importlib.import_module(
                f"user_llamdas.{name}_stub"
            )
            getattr(stub_module, name)
            with open(get_file_path(name), "w", encoding="utf-8") as f:
                f.write(f"\n{imports_str}\n\n{function_body}")
                f.close()
            new_module: ModuleType = importlib.import_module(f"user_llamdas.{name}")
            f = getattr(new_module, name)
            ll.fy(name=name, description=description)(f)
        except Exception as e:
            print(e)
            raise ValueError(f"error importing {name}: {e}") from e

        os.remove(get_file_path(f"{name}_stub"))

        return f"Function created: {name}"

    return create_function


def edit_function(
    name: str,
    body: str,
) -> None:
    """
    Edit the function body.
    """
    file_path = f"{name}.py"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Function stub file {file_path} not found.")

    # Here you would implement the logic to generate the function body
    # For this example, we'll just add a placeholder comment
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(body)
