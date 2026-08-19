"""Shell completion for the small public CLI surface."""

from __future__ import annotations

from zippergen.project_configuration import (
    _model_targets,
    connector_target_kinds,
)
from zippergen.workspace import Workspace, WorkspaceError
from zippergen.workflow_io import load_workflow_spec


def _parser_choices(path: tuple[str, ...] = ()) -> list[str]:
    """Read command choices from the real parser instead of copying them."""

    import argparse

    from zippergen.serve import HIDDEN_COMMANDS, _parse_cli_args

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
    subparsers = next(
        (
            action
            for action in current._actions
            if isinstance(action, argparse._SubParsersAction)
        ),
        None,
    )
    if subparsers is None:
        return []
    hidden = HIDDEN_COMMANDS if not path else frozenset()
    return [name for name in subparsers.choices if name not in hidden]


def completion_candidates(
    kind: str,
    project: str | None = None,
    path: tuple[str, ...] = (),
) -> list[str]:
    """Return parser-derived commands or project-derived configuration names."""

    if kind == "commands":
        return _parser_choices()
    if kind in {
        "config-actions",
        "workflow-actions",
        "provider-actions",
        "model-actions",
        "assistant-actions",
        "connector-actions",
        "run-actions",
        "deploy-actions",
    }:
        return _parser_choices((kind.removesuffix("-actions"),))
    if kind == "assistant-backends":
        return ["codex", "claude"]
    if kind == "provider-kinds":
        from zippergen.provider_connections import PROVIDER_KINDS

        return list(PROVIDER_KINDS)
    if kind == "connector-kinds":
        return ["telegram", "gmail", "google-sheets"]
    if kind == "options":
        return _option_candidates(path)
    try:
        workspace = Workspace(project)
        if kind == "provider-connections":
            return sorted(workspace.provider_connections())
        if kind == "provider-connections-model":
            from zippergen.provider_connections import provider_supports_models

            return sorted(
                name
                for name, values in workspace.provider_connections().items()
                if provider_supports_models(values.get("kind"))
            )
        if kind == "provider-connections-connector":
            return sorted(
                name
                for name, values in workspace.provider_connections().items()
                if values.get("kind") in {"telegram", "google"}
            )
        if kind == "model-configurations":
            return sorted(workspace.model_configurations())
        if kind == "assistant-configurations":
            return sorted(workspace.assistant_configurations())
        if kind == "connector-configurations":
            return sorted(workspace.connector_configurations())
        workflow_spec = workspace.resolve_workflow()
        workflow, module = load_workflow_spec(workspace.absolute_spec(workflow_spec))
    except (OSError, SystemExit, WorkspaceError, ValueError):
        return []
    if kind == "model-targets":
        return ["default", *_model_targets(workflow, module)]
    if kind == "assistant-targets":
        from zippergen.assistant_configuration import assistant_targets

        return assistant_targets(workflow, module)
    if kind == "connector-targets":
        return sorted(connector_target_kinds(workflow, module))
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
  case "$cmd:$CURRENT" in
    workflow:3|provider:3|model:3|assistant:3|connector:3|run:3|deploy:3)
      compadd -- ${(f)"$(zg __complete ${cmd}-actions 2>/dev/null)"}; return ;;
  esac
  case "$cmd:$action:$CURRENT" in
    model:assign:4|model:unassign:4) kind=model-targets ;;
    provider:configure:5) kind=provider-kinds ;;
    provider:set-credential:4|provider:check:4|provider:remove:4|provider:accept:4) kind=provider-connections ;;
    model:configure:5) kind=provider-connections-model ;;
    model:assign:5|model:check:4|model:remove:4) kind=model-configurations ;;
    assistant:assign:4|assistant:unassign:4) kind=assistant-targets ;;
    assistant:assign:5|assistant:check:4|assistant:remove:4) kind=assistant-configurations ;;
    assistant:configure:5) kind=assistant-backends ;;
    connector:assign:4|connector:unassign:4) kind=connector-targets ;;
    connector:assign:5|connector:check:4|connector:remove:4|connector:rename:4) kind=connector-configurations ;;
    connector:configure:5) kind=provider-connections-connector ;;
    connector:configure:6) kind=connector-kinds ;;
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
  elif [[ $cmd == workflow && $COMP_CWORD -eq 2 ]]; then kind=workflow-actions
  elif [[ $cmd == provider && $COMP_CWORD -eq 2 ]]; then kind=provider-actions
  elif [[ $cmd == model && $COMP_CWORD -eq 2 ]]; then kind=model-actions
  elif [[ $cmd == assistant && $COMP_CWORD -eq 2 ]]; then kind=assistant-actions
  elif [[ $cmd == connector && $COMP_CWORD -eq 2 ]]; then kind=connector-actions
  elif [[ $cmd == run && $COMP_CWORD -eq 2 ]]; then kind=run-actions
  elif [[ $cmd == deploy && $COMP_CWORD -eq 2 ]]; then kind=deploy-actions
  elif [[ $cmd == model && $action =~ ^(assign|unassign)$ && $COMP_CWORD -eq 3 ]]; then kind=model-targets
  elif [[ $cmd == provider && $action == configure && $COMP_CWORD -eq 4 ]]; then kind=provider-kinds
  elif [[ $cmd == provider && $action =~ ^(set-credential|check|remove|accept)$ && $COMP_CWORD -eq 3 ]]; then kind=provider-connections
  elif [[ $cmd == model && $action == configure && $COMP_CWORD -eq 4 ]]; then kind=provider-connections-model
  elif [[ $cmd == model && $action == assign && $COMP_CWORD -eq 4 ]] || [[ $cmd == model && $action =~ ^(check|remove)$ && $COMP_CWORD -eq 3 ]]; then kind=model-configurations
  elif [[ $cmd == assistant && $action =~ ^(assign|unassign)$ && $COMP_CWORD -eq 3 ]]; then kind=assistant-targets
  elif [[ $cmd == assistant && $action == assign && $COMP_CWORD -eq 4 ]] || [[ $cmd == assistant && $action =~ ^(check|remove)$ && $COMP_CWORD -eq 3 ]]; then kind=assistant-configurations
  elif [[ $cmd == assistant && $action == configure && $COMP_CWORD -eq 4 ]]; then kind=assistant-backends
  elif [[ $cmd == connector && $action =~ ^(assign|unassign)$ && $COMP_CWORD -eq 3 ]]; then kind=connector-targets
  elif [[ $cmd == connector && $action == assign && $COMP_CWORD -eq 4 ]] || [[ $cmd == connector && $action =~ ^(check|remove)$ && $COMP_CWORD -eq 3 ]]; then kind=connector-configurations
  elif [[ $cmd == connector && $action == configure && $COMP_CWORD -eq 4 ]]; then kind=provider-connections-connector
  elif [[ $cmd == connector && $action == configure && $COMP_CWORD -eq 5 ]]; then kind=connector-kinds
  fi
  [[ -n $kind ]] && COMPREPLY=( $(compgen -W "$(zg __complete "$kind" 2>/dev/null)" -- "$cur") )
}
complete -F _zg_complete zg zippergen'''


_FISH = r'''# Install with: zg completion fish | source
function __zg_complete_options
    set -l words (commandline -opc)
    zg __complete options $words[2..3] 2>/dev/null
end
complete -c zg -c zippergen -f
complete -c zg -c zippergen -n 'string match -q -- "-*" (commandline -ct)' -a '(__zg_complete_options)'
complete -c zg -c zippergen -n 'not __fish_seen_subcommand_from config check workflow provider model assistant completion connector run show validate init skill snapshot diff deploy' -a '(zg __complete commands 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from workflow; and not __fish_seen_subcommand_from select' -a '(zg __complete workflow-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from provider; and not __fish_seen_subcommand_from configure set-credential check rename remove authorize accept' -a '(zg __complete provider-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete model-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete assistant-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete connector-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from run; and not __fish_seen_subcommand_from status reset inspect trace tasks approve' -a '(zg __complete run-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from deploy; and not __fish_seen_subcommand_from list prune start stop remove compact logs check status reset inspect trace tasks approve' -a '(zg __complete deploy-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model assign; and test (count (commandline -opc)) -eq 3' -a '(zg __complete model-targets 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model assign; and test (count (commandline -opc)) -eq 4' -a '(zg __complete model-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from provider configure; and test (count (commandline -opc)) -eq 4' -a '(zg __complete provider-kinds 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from provider set-credential check remove accept; and test (count (commandline -opc)) -eq 3' -a '(zg __complete provider-connections 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model configure; and test (count (commandline -opc)) -eq 4' -a '(zg __complete provider-connections-model 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model check remove; and test (count (commandline -opc)) -eq 3' -a '(zg __complete model-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant assign unassign; and test (count (commandline -opc)) -eq 3' -a '(zg __complete assistant-targets 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant assign; and test (count (commandline -opc)) -eq 4' -a '(zg __complete assistant-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant check remove; and test (count (commandline -opc)) -eq 3' -a '(zg __complete assistant-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant configure; and test (count (commandline -opc)) -eq 4' -a '(zg __complete assistant-backends 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector assign; and test (count (commandline -opc)) -eq 3' -a '(zg __complete connector-targets 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector assign; and test (count (commandline -opc)) -eq 4' -a '(zg __complete connector-configurations 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector configure; and test (count (commandline -opc)) -eq 4' -a '(zg __complete provider-connections-connector 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector configure; and test (count (commandline -opc)) -eq 5' -a '(zg __complete connector-kinds 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector check remove; and test (count (commandline -opc)) -eq 3' -a '(zg __complete connector-configurations 2>/dev/null)'
'''
