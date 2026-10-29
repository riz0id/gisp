{ lib
, buildPythonPackage
, fetchPypi
, hatch-fancy-pypi-readme
, hatchling
}:

# A dependency of `pydoctor` that nixpkgs does not package, so it is built here
# rather than relaxed away: `pydoctor` imports it unconditionally for the search
# index it writes into generated documentation.
buildPythonPackage rec {
  pname = "lunr";
  version = "0.8.0";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-tGz1BZV40nehS/yQG7PVZm0BO/c8A1MxrAIi/aw1gig=";
  };

  build-system = [
    hatchling
    hatch-fancy-pypi-readme
  ];

  # No `dependencies`: every runtime requirement lunr declares is conditioned on
  # `python_version < "3.8"`, so on 3.14 the list is empty.

  # The suite pulls in nltk and pytest-benchmark for the optional language
  # stemmers, which nothing here uses. The import check is what proves the
  # build produced a working module.
  doCheck = false;

  pythonImportsCheck = [ "lunr" ];

  meta = {
    description = "Python port of lunr.js, a full-text search index";
    homepage = "https://github.com/yeraydiazdiaz/lunr.py";
    license = lib.licenses.mit;
  };
}
