"""Shell completion for the small public CLI surface."""

from __future__ import annotations

from zippergen.connector_wiring import human_action_sites
from zippergen.connectors import connector_requirements_from_module
from zippergen.project_configuration import _model_targets
from zippergen.workspace import Workspace, WorkspaceError
from zippergen.workflow_io import load_workflow_spec


COMMANDS = (
    "config", "model", "connector", "completion", "run", "show", "validate",
    "init", "skill", "snapshot", "diff", "deploy", "configure", "start",
    "stop", "remove", "compact", "restart", "logs", "doctor", "status",
    "inspect", "trace", "tasks", "approve", "notify",
)
MODEL_ACTIONS = ("configure", "assign", "unassign", "check", "remove")
CONNECTOR_ACTIONS = (
    "configure", "assign", "unassign", "unbind", "check", "remove",
    "authorize", "accept",
)
def completion_candidates(
    kind: str,
    project: str | None = None,
    path: tuple[str, ...] = (),
) -> list[str]:
    """Return current project or deployment names for shell completion."""

    if kind == "commands":
        return list(COMMANDS)
    if kind == "model-actions":
        return list(MODEL_ACTIONS)
    if kind == "connector-actions":
        return list(CONNECTOR_ACTIONS)
    if kind == "options":
        return _option_candidates(path)
    if kind == "deployments":
        from zippergen.deployment_platform import deployments_dir

        directory = deployments_dir()
        return sorted(path.stem for path in directory.glob("*.json"))

    try:
        workspace = Workspace(project)
        if kind == "model-configurations":
            return sorted(workspace.model_configurations())
        if kind == "connector-configurations":
            return sorted(workspace.connector_configurations())
        workflow_spec = workspace.resolve_workflow()
        workflow, module = load_workflow_spec(workspace.absolute_spec(workflow_spec))
    except (OSError, SystemExit, WorkspaceError, ValueError):
        return []
    if kind == "model-targets":
        return ["default", *_model_targets(workflow, module)]
    if kind == "connector-targets":
        sites = human_action_sites(workflow, module)
        return sorted(
            {
                *sites,
                *(
                    f"{participant}.{action}"
                    for participant, actions in sites.items()
                    for action in actions
                ),
            }
        )
    if kind == "connector-requirements":
        return sorted(item.name for item in connector_requirements_from_module(module))
    return []


def _option_candidates(path: tuple[str, ...]) -> list[str]:
    """Derive flags from the real argparse tree, so completion cannot drift."""

    import argparse

    from zippergen.serve import _parse_cli_args

    parser, _arguments = _parse_cli_args([])
    current = parser
    for name in path:
        subparsers = next(
            (
                action
                for action in current._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            None,
        )
        if subparsers is None or name not in subparsers.choices:
            return []
        current = subparsers.choices[name]
    return sorted(
        {
            option
            for action in current._actions
            for option in action.option_strings
        }
    )


def render_completion(shell: str) -> str:
    """Render a self-contained completion script for zsh, bash, or fish."""

    if shell == "zsh":
        return _ZSH
    if shell == "bash":
        return _BASH
    if shell == "fish":
        return _FISH
    raise ValueError(f"Unsupported shell: {shell}")


_ZSH = r'''# Install with: eval "$(zg completion zsh)"
autoload -Uz compinit
(( $+functions[compdef] )) || compinit
_zg() {
  local cmd action kind
  if (( CURRENT == 2 )); then
    compadd -- ${(f)"$(zg __complete commands 2>/dev/null)"}
    return
  fi
  cmd=$words[2]
  action=$words[3]
  if [[ $words[CURRENT] == -* ]]; then
    compadd -- ${(f)"$(zg __complete options $cmd $action 2>/dev/null)"}
    return
  fi
  if [[ $cmd == model && CURRENT == 3 ]]; then
    compadd -- ${(f)"$(zg __complete model-actions 2>/dev/null)"}; return
  fi
  if [[ $cmd == connector && CURRENT == 3 ]]; then
    compadd -- ${(f)"$(zg __complete connector-actions 2>/dev/null)"}; return
  fi
  case "$cmd:$action:$CURRENT" in
    model:assign:4|model:unassign:4) kind=model-targets ;;
    model:assign:5|model:check:4|model:remove:4) kind=model-configurations ;;
    connector:assign:4|connector:unassign:4) kind=connector-targets ;;
    connector:assign:5|connector:check:4|connector:remove:4) kind=connector-configurations ;;
    connector:unbind:4) kind=connector-requirements ;;
    configure:*:3|start:*:3|stop:*:3|remove:*:3|compact:*:3|restart:*:3|logs:*:3|doctor:*:3|status:*:3|inspect:*:3|trace:*:3|tasks:*:3|approve:*:3) kind=deployments ;;
  esac
  [[ -n $kind ]] && compadd -- ${(f)"$(zg __complete $kind 2>/dev/null)"}
}
compdef _zg zg zippergen'''


_BASH = r'''# Install with: eval "$(zg completion bash)"
_zg_complete() {
  local cur cmd action kind
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  cmd="${COMP_WORDS[1]}"
  action="${COMP_WORDS[2]}"
  if [[ $cur == -* ]]; then
    COMPREPLY=( $(compgen -W "$(zg __complete options "$cmd" "$action" 2>/dev/null)" -- "$cur") ); return
  elif [[ $COMP_CWORD -eq 1 ]]; then kind=commands
  elif [[ $cmd == model && $COMP_CWORD -eq 2 ]]; then kind=model-actions
  elif [[ $cmd == connector && $COMP_CWORD -eq 2 ]]; then kind=connector-actions
  elif [[ $cmd == model && $action =~ ^(assign|unassign)$ && $COMP_CWORD -eq 3 ]]; then kind=model-targets
  elif [[ $cmd == model && $action == assign && $COMP_CWORD -eq 4 ]] || [[ $cmd == model && $action =~ ^(check|remove)$ && $COMP_CWORD -eq 3 ]]; then kind=model-configurations
  elif [[ $cmd == connector && $action =~ ^(assign|unassign)$ && $COMP_CWORD -eq 3 ]]; then kind=connector-targets
  elif [[ $cmd == connector && $action == assign && $COMP_CWORD -eq 4 ]] || [[ $cmd == connector && $action =~ ^(check|remove)$ && $COMP_CWORD -eq 3 ]]; then kind=connector-configurations
  elif [[ $cmd == connector && $action == unbind && $COMP_CWORD -eq 3 ]]; then kind=connector-requirements
  elif [[ " ${cmd} " == *" configure "* || " ${cmd} " == *" start "* || " ${cmd} " == *" stop "* || " ${cmd} " == *" remove "* || " ${cmd} " == *" compact "* || " ${cmd} " == *" restart "* || " ${cmd} " == *" logs "* || " ${cmd} " == *" doctor "* || " ${cmd} " == *" status "* || " ${cmd} " == *" inspect "* || " ${cmd} " == *" trace "* || " ${cmd} " == *" tasks "* || " ${cmd} " == *" approve "* ]] && [[ $COMP_CWORD -eq 2 ]]; then kind=deployments
  fi
  [[ -n $kind ]] && COMPREPLY=( $(compgen -W "$(zg __complete "$kind" 2>/dev/null)" -- "$cur") )
}
complete -F _zg_complete zg zippergen'''


_FISH = r'''# Install with: zg completion fish | source
complete -c zg -c zippergen -f
complete -c zg -c zippergen -n 'not __fish_seen_subcommand_from config model connector completion run show validate init skill snapshot diff deploy configure start stop remove compact restart logs doctor status inspect trace tasks approve notify' -a '(zg __complete commands 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from configure assign unassign check remove' -a '(zg __complete model-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector; and not __fish_seen_subcommand_from configure assign unassign unbind check remove authorize accept' -a '(zg __complete connector-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model assign; and test (count (commandline -opc)) -eq 3' -a '(zg __complete model-targets 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model assign; and test (count (commandline -opc)) -eq 4' -a '(zg __complete model-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model check remove; and test (count (commandline -opc)) -eq 3' -a '(zg __complete model-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector assign; and test (count (commandline -opc)) -eq 3' -a '(zg __complete connector-targets 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector assign; and test (count (commandline -opc)) -eq 4' -a '(zg __complete connector-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector check remove; and test (count (commandline -opc)) -eq 3' -a '(zg __complete connector-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector unbind; and test (count (commandline -opc)) -eq 3' -a '(zg __complete connector-requirements 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from start stop remove compact restart logs doctor status inspect trace tasks approve; and test (count (commandline -opc)) -eq 2' -a '(zg __complete deployments 2>/dev/null)'
'''
