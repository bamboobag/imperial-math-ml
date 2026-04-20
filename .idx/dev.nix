# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  channel = "stable-25.05";

  packages = [
    pkgs.python312     
    pkgs.uv            
    pkgs.stdenv.cc.cc.lib 
  ];

  env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  };

  idx = {
    extensions = [
      "ms-python.python"
      "ms-python.vscode-pylance"
      "charliermarsh.ruff"      
      "ms-toolsai.jupyter"       
      "google.gemini-cli-vscode-ide-companion"
    ];

    workspace = {
      onCreate = {
        # High-speed dependency sync
        uv-sync = "uv sync";
        default.openFiles = [ "README.md" ];
      };
    };
  };
}