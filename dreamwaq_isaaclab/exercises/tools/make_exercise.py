#!/usr/bin/env python3
"""Generate exercise starters from the solution sources.

The completed code *is* the solution. A starter is always generated from it — never hand-written —
so the two cannot drift apart. There are two ways to declare an exercise, chosen by whether the
solution is production code or lecture-only code.

1. Spec files — for **project modules** (`dreamwaq_manager/`, `dreamwaq_direct/`)
-------------------------------------------------------------------------------
The production source stays pristine: no markers, no comments, nothing. The exercise is declared
in `exercises/specs/<id>.toml`, and `body` holds the exact snippet to cut out:

    id     = "cenet-loss"
    level  = 2
    stage  = "stage4_cenet"
    source = "dreamwaq_manager/.../cenet.py"
    task   = "CENet 의 세 손실 항을 완성한다"
    hints  = ["...", "..."]
    body   = '''            mse_loss = nn.MSELoss()
    ...
    '''

At generation time the tool reads the real file and replaces `body` verbatim. If `body` is no
longer found — because the source changed — it fails loudly instead of silently producing a
stale starter.

Three optional fields tune what the learner sees:

    inline_hints = 1                  # only the first N hints go into the TODO block.
                                      # omit → all of them (the historical behaviour)
    also_blank   = ["cenet-loss"]     # blank these other exercises' bodies too, so this
                                      # starter does not ship the next exercise's answer.
                                      # They must live in the same source file, and this
                                      # exercise's check.py must not need them.
    [[also_mask]]                     # mask PROSE (docstrings, comments) that gives the
    find = "Encoder: ... → 35"        # answer away. `also_blank` stubs out code; this one
    replace = "구조는 직접 세운다"      # swaps a verbatim snippet for a neutral one.
                                      # Repeat the [[also_mask]] block per snippet.

`also_blank` and `also_mask` both fail loudly when their target is missing or ambiguous —
a silent no-op would ship the answer without anyone noticing.

2. Inline markers — for **standalone exercises** that live under `exercises/`
----------------------------------------------------------------------------
Those files are lecture material to begin with, so marking them in place costs nothing:

    # ex:begin id=pd-target level=1 stage=stage1_isaac_basics task=관절 목표를 계산한다
    #   hint: joint_target = default_joint_pos + action * ACTION_SCALE
    return robot.data.default_joint_pos + action * ACTION_SCALE
    # ex:end

`inline_hints=N` works here too (as a `key=value` on the `ex:begin` line): every hint stays
in the solution file, but only the first N reach the starter's TODO block. The rest belong in
the README behind a `<details>`.

Usage
-----
    make_exercise.py --list                 # show every exercise found
    make_exercise.py --id cenet-loss        # generate one starter
    make_exercise.py --all                  # generate every starter
    make_exercise.py --all --check          # verify starters are current (exit 1 if stale)
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXERCISES = REPO_ROOT / "exercises"
SPECS = EXERCISES / "specs"

# Only standalone exercises under exercises/ are scanned for inline markers. Project modules
# are declared in specs/ so their sources stay untouched.
MARKER_ROOTS = [EXERCISES]
SKIP_DIRS = {"logs", "wandb", "outputs", "__pycache__", ".git", "starter", "tools", "specs"}

BEGIN_RE = re.compile(r"^(?P<indent>\s*)#\s*ex:begin\s+(?P<attrs>.*)$")
END_RE = re.compile(r"^\s*#\s*ex:end\s*$")
HINT_RE = re.compile(r"^\s*#\s+hint:\s*(?P<hint>.*)$")
# `key=value` where value runs to the next ` key=` or end of line — lets task= hold spaces.
ATTR_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")


@dataclass
class Exercise:
    id: str
    level: int
    task: str
    stage: str
    source: Path
    hints: list[str] = field(default_factory=list)
    out: str | None = None
    # spec-based: the exact snippet to remove. marker-based: the line range instead.
    body: str | None = None
    start: int = -1
    end: int = -1
    indent: str = ""
    origin: str = "spec"  # "spec" | "marker"
    # How many hints to inline in the TODO block. None = all of them (the default, and what
    # every exercise did before this field existed). Set it low when the README already carries
    # the full hint list behind a <details> — an always-expanded inline hint removes the
    # learner's option to try without one.
    inline_hints: int | None = None
    # Other exercises' bodies (by id) that must ALSO be blanked in this starter. Without this,
    # cutting only your own region out of a shared production file leaves the *next* exercise's
    # finished answer sitting in the same file.
    also_blank: list[str] = field(default_factory=list)
    # Prose to mask: [{"find": <verbatim snippet>, "replace": <neutral text>}, ...]. Blanking the
    # code is not enough when a docstring or an inline comment right above it spells the answer
    # out in words.
    also_mask: list[dict[str, str]] = field(default_factory=list)

    def starter_path(self) -> Path:
        if self.out:
            return (EXERCISES / self.out).resolve()
        if EXERCISES in self.source.parents:
            # Standalone: the solution already lives in the task folder, so the starter is
            # its sibling (`solution.py` → `starter.py`).
            return self.source.with_name("starter.py")
        # Project module: mirror the real file under the exercise's starter/ dir.
        return EXERCISES / self.stage / self.id / "starter" / self.source.name


def _todo_block(ex: Exercise, indent: str) -> str:
    lines = [
        f"{indent}# ── TODO({ex.id}) ─ level L{ex.level} ─────────────────────────────",
        f"{indent}# {ex.task}",
    ]
    shown = ex.hints if ex.inline_hints is None else ex.hints[: ex.inline_hints]
    lines += [f"{indent}#   hint: {h}" for h in shown]
    if len(shown) < len(ex.hints):
        lines.append(f"{indent}#   나머지 힌트는 README.md 의 '힌트' 절에 접혀 있다 — 먼저 안 보고 해 본다.")
    lines += [
        f"{indent}# 통과 기준은 이 실습 폴더의 README.md 를 본다.",
        f'{indent}raise NotImplementedError("TODO({ex.id})")',
        f"{indent}# ─────────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


def _other_block(other: Exercise, indent: str) -> str:
    """Placeholder for a *different* exercise's region — no task text, no hints, no answer."""
    return "\n".join(
        [
            f"{indent}# ── 다른 실습({other.id})의 구간이다 — 여기는 지금 채우지 않는다 ──────",
            f'{indent}raise NotImplementedError("TODO({other.id}) — 이 실습의 범위가 아니다")',
            f"{indent}# ─────────────────────────────────────────────────────────────────────",
        ]
    )


# --------------------------------------------------------------------------------------
# spec files
# --------------------------------------------------------------------------------------


def load_specs() -> list[Exercise]:
    if not SPECS.exists():
        return []
    out: list[Exercise] = []
    for path in sorted(SPECS.glob("*.toml")):
        data = tomllib.loads(path.read_text())
        missing = {"id", "level", "stage", "source", "task", "body"} - data.keys()
        if missing:
            raise SystemExit(f"{path}: spec is missing {sorted(missing)}")
        src = (REPO_ROOT / data["source"]).resolve()
        if not src.exists():
            raise SystemExit(f"{path}: source does not exist: {data['source']}")
        masks = list(data.get("also_mask", []))
        for m in masks:
            if {"find", "replace"} - m.keys():
                raise SystemExit(f"{path}: [[also_mask]] needs both 'find' and 'replace': {m}")
        out.append(
            Exercise(
                id=data["id"],
                level=int(data["level"]),
                task=data["task"],
                stage=data["stage"],
                source=src,
                hints=list(data.get("hints", [])),
                out=data.get("out"),
                body=data["body"],
                origin="spec",
                inline_hints=data.get("inline_hints"),
                also_blank=list(data.get("also_blank", [])),
                also_mask=masks,
            )
        )
    return out


def _locate(ex: Exercise, text: str, body: str, owner: str) -> str:
    if body not in text:
        raise SystemExit(
            f"[{ex.id}] spec 의 body 를 소스에서 찾지 못했다{'' if owner == ex.id else f' (also_blank={owner})'}.\n"
            f"  source: {ex.source.relative_to(REPO_ROOT)}\n"
            f"  소스가 바뀌었다면 exercises/specs/{owner}.toml 의 body 를 현재 코드로 갱신한다.\n"
            f"  찾던 첫 줄: {body.splitlines()[0] if body.splitlines() else '<empty>'!r}"
        )
    if text.count(body) > 1:
        raise SystemExit(f"[{owner}] body 가 소스에 {text.count(body)}번 나온다 — 유일해야 한다")
    return re.match(r"\s*", body.splitlines()[0]).group(0)


def render_from_spec(ex: Exercise, all_ex: list[Exercise]) -> str:
    text = ex.source.read_text()
    indent = _locate(ex, text, ex.body, ex.id)
    text = text.replace(ex.body, _todo_block(ex, indent) + "\n", 1)

    for other_id in ex.also_blank:
        other = next((e for e in all_ex if e.id == other_id), None)
        if other is None or other.body is None:
            raise SystemExit(f"[{ex.id}] also_blank 가 가리키는 spec 이 없다: {other_id!r}")
        if other.source != ex.source:
            raise SystemExit(
                f"[{ex.id}] also_blank={other_id!r} 의 source 가 다르다 — 같은 파일이어야 한다"
            )
        other_indent = _locate(ex, text, other.body, other_id)
        text = text.replace(other.body, _other_block(other, other_indent) + "\n", 1)

    for mask in ex.also_mask:
        find = mask["find"]
        if find not in text:
            raise SystemExit(
                f"[{ex.id}] also_mask 의 find 를 소스에서 찾지 못했다.\n"
                f"  source: {ex.source.relative_to(REPO_ROOT)}\n"
                f"  소스가 바뀌었다면 exercises/specs/{ex.id}.toml 의 also_mask 를 현재 문구로 갱신한다.\n"
                f"  찾던 첫 줄: {find.splitlines()[0] if find.splitlines() else '<empty>'!r}"
            )
        if text.count(find) > 1:
            raise SystemExit(
                f"[{ex.id}] also_mask 의 find 가 소스에 {text.count(find)}번 나온다 — 유일해야 한다\n"
                f"  찾던 첫 줄: {find.splitlines()[0]!r}"
            )
        text = text.replace(find, mask["replace"], 1)
    return text


# --------------------------------------------------------------------------------------
# inline markers
# --------------------------------------------------------------------------------------


def _parse_attrs(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in ATTR_RE.finditer(text.strip())}


def load_markers(paths: list[Path] | None = None) -> list[Exercise]:
    found: list[Exercise] = []
    files: list[Path] = []
    for root in paths or MARKER_ROOTS:
        if not root.exists():
            continue
        files += [p for p in sorted(root.rglob("*.py")) if not SKIP_DIRS & set(p.parts)]

    for path in files:
        lines = path.read_text().splitlines()
        open_ex: Exercise | None = None
        hints_closed = False
        for i, line in enumerate(lines):
            begin = BEGIN_RE.match(line)
            if begin:
                if open_ex is not None:
                    raise SystemExit(f"{path}:{i + 1}: 'ex:begin' inside an unclosed region")
                attrs = _parse_attrs(begin.group("attrs"))
                miss = {"id", "level", "task"} - attrs.keys()
                if miss:
                    raise SystemExit(f"{path}:{i + 1}: marker is missing {sorted(miss)}")
                open_ex = Exercise(
                    id=attrs["id"],
                    level=int(attrs["level"]),
                    task=attrs["task"],
                    stage=attrs.get("stage", "unsorted"),
                    source=path,
                    out=attrs.get("out"),
                    start=i,
                    indent=begin.group("indent"),
                    origin="marker",
                    # `task=` swallows the rest of the line, so this must precede it on ex:begin.
                    inline_hints=int(attrs["inline_hints"]) if "inline_hints" in attrs else None,
                )
                hints_closed = False
                continue
            if open_ex is None:
                continue
            hint = HINT_RE.match(line)
            if hint and not hints_closed:
                open_ex.hints.append(hint.group("hint").strip())
                continue
            hints_closed = True
            if END_RE.match(line):
                open_ex.end = i
                found.append(open_ex)
                open_ex = None
        if open_ex is not None:
            raise SystemExit(f"{path}:{open_ex.start + 1}: 'ex:begin' has no matching 'ex:end'")
    return found


def render_from_marker(ex: Exercise, same_file: list[Exercise]) -> str:
    lines = ex.source.read_text().splitlines()
    # Other exercises' marker comments must not leak into this starter.
    drop: set[int] = set()
    for other in same_file:
        drop.add(other.start)
        drop.add(other.end)
        drop.update(range(other.start + 1, other.start + 1 + len(other.hints)))
    kept = lines[: ex.start] + _todo_block(ex, ex.indent).splitlines() + lines[ex.end + 1 :]
    if drop:
        offset = len(_todo_block(ex, ex.indent).splitlines()) - (ex.end - ex.start + 1)
        shifted = {i if i < ex.start else i + offset for i in drop}
        kept = [ln for i, ln in enumerate(kept) if i not in shifted]
    text = "\n".join(kept) + "\n"
    # "python solution.py" in a docstring is right for the solution and wrong for the starter —
    # the learner runs the file they are editing. Only the `solution.py` → `starter.py` sibling
    # case: other sources (e.g. stage3's shared `ppo.py`) are referenced by their real name.
    out_name = ex.starter_path().name
    if ex.source.name == "solution.py" and out_name != "solution.py":
        text = text.replace("solution.py", out_name)
    return text


# --------------------------------------------------------------------------------------


def load_all() -> list[Exercise]:
    exercises = load_specs() + load_markers()
    dupes = {e.id for e in exercises if sum(1 for o in exercises if o.id == e.id) > 1}
    if dupes:
        raise SystemExit(f"duplicate exercise ids: {sorted(dupes)}")
    return exercises


def generate(ex: Exercise, all_ex: list[Exercise], *, check: bool) -> bool:
    if ex.origin == "spec":
        text = render_from_spec(ex, all_ex)
    else:
        same_file = [e for e in all_ex if e.origin == "marker" and e.source == ex.source and e.id != ex.id]
        text = render_from_marker(ex, same_file)

    out = ex.starter_path()
    if check:
        if not out.exists():
            print(f"[STALE] missing  {out.relative_to(REPO_ROOT)}")
            return False
        if out.read_text() != text:
            print(f"[STALE] outdated {out.relative_to(REPO_ROOT)}")
            return False
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[ok] {ex.id:24s} L{ex.level}  {ex.origin:6s} → {out.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list every exercise and exit")
    ap.add_argument("--id", help="generate a single exercise by id")
    ap.add_argument("--all", action="store_true", help="generate every exercise")
    ap.add_argument("--check", action="store_true", help="verify starters are current; do not write")
    args = ap.parse_args()

    exercises = load_all()

    if args.list or not (args.id or args.all):
        if not exercises:
            print("no exercises found — add a spec under exercises/specs/ or an inline marker")
            return 0
        print(f"{'id':22s} {'lvl':>3s} {'from':7s} {'stage':22s} task")
        for e in sorted(exercises, key=lambda x: (x.stage, x.id)):
            print(f"{e.id:22s} L{e.level:<2d} {e.origin:7s} {e.stage:22s} {e.task}")
        return 0

    targets = exercises if args.all else [e for e in exercises if e.id == args.id]
    if not targets:
        print(f"no exercise with id={args.id!r}", file=sys.stderr)
        return 1

    # not `all(...)` — it short-circuits and would hide every stale starter after the first.
    ok = all([generate(e, exercises, check=args.check) for e in targets])
    if args.check:
        print("all starters up to date" if ok else "run without --check to regenerate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
