{
  description = "A minimal Lisp interpreter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable }:
    let
      overlays = [ (import ./nix/packages) ];
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs { inherit system overlays; };
          unstable = nixpkgs-unstable.legacyPackages.${system};
        in
        f { inherit pkgs unstable; });
    in
    {
      devShells = forAllSystems ({ pkgs, unstable, ... }: {
        default = pkgs.mkShell {
          packages = [
            pkgs.python314
            pkgs.uv
            pkgs.ruff
            unstable.ty
            pkgs.python3Packages.pydoctor
          ];

          shellHook = ''
            export UV_PYTHON="${pkgs.python314}/bin/python3.14"
            export UV_PYTHON_DOWNLOADS=never
          '';
        };
      });
    };
}
