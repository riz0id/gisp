{ lib
, attrs
, buildPythonPackage
, cachecontrol
, configargparse
, docutils
, fetchPypi
, filelock
, lunr
, platformdirs
, requests
, twisted
, urllib3
}:

buildPythonPackage rec {
  pname = "pydoctor";
  version = "25.10.1";
  format = "wheel";
  src = fetchPypi {
    inherit pname version format;
    dist = "py3";
    python = "py3";
    hash = "sha256-KqhfjWThHAZdcaIxe4JySlg2EXPZRSkFCTZ2gWZb3Hw=";
  };
  dependencies = [
    attrs
    cachecontrol
    configargparse
    docutils
    filelock
    lunr
    platformdirs
    requests
    twisted
    urllib3
  ];
  doCheck = false;
  pythonImportsCheck = [ "pydoctor" ];
  meta = {
    description = "API documentation generator for Python";
    homepage = "https://github.com/twisted/pydoctor";
    license = lib.licenses.mit;
    mainProgram = "pydoctor";
  };
}
