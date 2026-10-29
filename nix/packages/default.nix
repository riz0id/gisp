final: prev:

{
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pythonFinal: pythonPrev: {
      lunr = pythonFinal.callPackage ./lunr.nix { };
      pydoctor = pythonFinal.callPackage ./pydoctor.nix { };
    })
  ];
}
