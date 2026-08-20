"""Shell completion for the small public CLI surface."""

from __future__ import annotations

from zippergen.project_configuration import (
    _model_targets,
    connector_target_kinds,
)
from zippergen.workspace import Workspace, WorkspaceError
from zippergen.workflow_io import load_workflow_spec


#: Which candidates complete a positional argument, as
#: ``(command, action, index) -> candidate kind`` where the index counts words
#: from the program name. Three shells need this same knowledge, and writing it
#: out three times is how `rename` reached zsh but not bash or fish. It is
#: written once here and each script is generated from it.
POSITIONAL_COMPLETIONS: dict[tuple[str, str, int], str] = {
    ("provider", "configure", 3): "provider-kinds",
    ("provider", "set-credential", 2): "provider-connections",
    ("provider", "check", 2): "provider-connections",
    ("provider", "remove", 2): "provider-connections",
    ("provider", "rename", 2): "provider-connections",
    ("provider", "accept", 2): "provider-connections-google",
    ("provider", "authorize", 2): "provider-connections-google",
    ("model", "configure", 3): "provider-connections-model",
    ("model", "assign", 2): "model-targets",
    ("model", "unassign", 2): "model-targets",
    ("model", "assign", 3): "model-configurations",
    ("model", "check", 2): "model-configurations",
    ("model", "remove", 2): "model-configurations",
    ("model", "rename", 2): "model-configurations",
    ("assistant", "configure", 3): "assistant-backends",
    ("assistant", "assign", 2): "assistant-targets",
    ("assistant", "unassign", 2): "assistant-targets",
    ("assistant", "assign", 3): "assistant-configurations",
    ("assistant", "check", 2): "assistant-configurations",
    ("assistant", "remove", 2): "assistant-configurations",
    ("assistant", "rename", 2): "assistant-configurations",
    ("connector", "configure", 3): "provider-connections-connector",
    ("connector", "configure", 4): "connector-kinds",
    ("connector", "assign", 2): "connector-targets",
    ("connector", "unassign", 2): "connector-targets",
    ("connector", "assign", 3): "connector-configurations",
    ("connector", "check", 2): "connector-configurations",
    ("connector", "remove", 2): "connector-configurations",
    ("connector", "rename", 2): "connector-configurations",
    ("workflow", "select", 2): "workflow-specs",
}


def _zsh_positionals() -> str:
    return "\n".join(
        f"    {command}:{action}:{index + 2}) kind={candidate} ;;"
        for (command, action, index), candidate in POSITIONAL_COMPLETIONS.items()
    )


def _bash_positionals() -> str:
    # COMP_CWORD counts the program as word 0, so the word being completed
    # sits one past the table index. zsh's compadd key counts from 1 and
    # includes the action, and fish counts the words already typed.
    return "\n".join(
        f"  elif [[ $cmd == {command} && $action == {action} "
        f"&& $COMP_CWORD -eq {index + 1} ]]; then kind={candidate}"
        for (command, action, index), candidate in POSITIONAL_COMPLETIONS.items()
    )


def _fish_positionals() -> str:
    return "\n".join(
        f"complete -c zg -c zippergen -n '__fish_seen_subcommand_from {command}; "
        f"and __fish_seen_subcommand_from {action}; "
        f"and test (count (commandline -opc)) -eq {index + 1}' "
        f"-a '(zg __complete {candidate} 2>/dev/null)'"
        for (command, action, index), candidate in POSITIONAL_COMPLETIONS.items()
    )


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
        if kind == "provider-connections-google":
            return sorted(
                name
                for name, values in workspace.provider_connections().items()
                if values.get("kind") == "google"
            )
        if kind == "workflow-specs":
            from zippergen.workspace import discover_workflow_specs

            return sorted(discover_workflow_specs(workspace.root))
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

    templates = {
        "zsh": (_ZSH, "@@ZSH_POSITIONALS@@", _zsh_positionals),
        "bash": (_BASH, "@@BASH_POSITIONALS@@", _bash_positionals),
        "fish": (_FISH, "@@FISH_POSITIONALS@@", _fish_positionals),
    }
    if shell not in templates:
        raise ValueError(f"Unsupported shell: {shell}")
    template, marker, render = templates[shell]
    return template.replace(marker, render())


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
@@ZSH_POSITIONALS@@
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
@@BASH_POSITIONALS@@
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
@@FISH_POSITIONALS@@
complete -c zg -c zippergen -n 'string match -q -- "-*" (commandline -ct)' -a '(__zg_complete_options)'
complete -c zg -c zippergen -n 'not __fish_seen_subcommand_from config check workflow provider model assistant completion connector run show validate init skill snapshot diff deploy' -a '(zg __complete commands 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from workflow; and not __fish_seen_subcommand_from select' -a '(zg __complete workflow-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from provider; and not __fish_seen_subcommand_from configure set-credential check rename remove authorize accept' -a '(zg __complete provider-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete model-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from assistant; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete assistant-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from connector; and not __fish_seen_subcommand_from configure assign unassign check rename remove' -a '(zg __complete connector-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from run; and not __fish_seen_subcommand_from status reset inspect trace tasks approve' -a '(zg __complete run-actions 2>/dev/null)'
complete -c zg -c zippergen -n '__fish_seen_subcommand_from deploy; and not __fish_seen_subcommand_from list prune start stop remove compact logs check status reset inspect trace tasks approve' -a '(zg __complete deploy-actions 2>/dev/null)'
'''
