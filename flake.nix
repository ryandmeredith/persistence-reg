{
  inputs = {
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      flake-parts,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = lib.systems.flakeExposed;
        perSystem =
          { pkgs, ... }:
          {
            devShells.default =
              let
                project = pyproject-nix.lib.project.loadPyproject {
                  projectRoot = ./.;
                };
                python = pkgs.python3;
                dependencies = project.renderers.withPackages {
                  inherit python;
                  groups = [ "dev" ];
                } python.pkgs;
                tools = with pkgs; [
                  python
                  ruff
                  ty
                ];
              in
              pkgs.mkShell { packages = tools ++ dependencies; };
          };
      }
    );
}
