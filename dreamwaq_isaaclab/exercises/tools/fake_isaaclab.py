"""Isaac Sim 없이 프로젝트 모듈을 import 하기 위한 최소 스텁.

`rewards.py` 같은 파일은 `isaaclab.*` 을 import 하지만, 실습이 검사하는 계산 자체는
순수 torch 다. Isaac Sim 을 띄우면 25초가 걸리므로, import 만 통과시키는 가짜 모듈을
`sys.modules` 에 심어 1초 만에 검사한다.

이름을 하나씩 열거하지 않는다 — 어떤 속성 접근이든 더미 클래스를 돌려주므로,
프로젝트가 새 심볼을 import 하기 시작해도 검사기를 고칠 필요가 없다.

    from fake_isaaclab import install_stubs, load_module

    install_stubs()
    rewards = load_module(Path(".../rewards.py"))
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

# 이 접두사로 시작하는 모듈은 전부 가짜로 만든다. 이름을 열거하지 않고 meta path finder 로
# 가로채므로, 프로젝트가 새 isaaclab 서브모듈을 import 하기 시작해도 여기를 고칠 필요가 없다.
_STUB_PREFIXES = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "usdrt")


class _DummyMeta(type):
    """클래스 레벨 속성 접근도 받아낸다 (예: RayCasterCfg.OffsetCfg)."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _DummyMeta(name, (_Dummy,), {})
        setattr(cls, name, sub)
        return sub

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class _Dummy(metaclass=_DummyMeta):
    """무엇으로 쓰이든 죽지 않는 자리표시자 (기반 클래스, 타입 힌트, 데코레이터)."""

    def __init__(self, *args, **kwargs):
        # ManagerTermBase 처럼 (cfg, env) 를 받는 기반 클래스로도 쓰인다.
        self.cfg = kwargs.get("cfg", args[0] if args else None)
        self._env = kwargs.get("env", args[1] if len(args) > 1 else None)

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _Dummy()

    @classmethod
    def __class_getitem__(cls, item):
        return cls


class _AnyModule(types.ModuleType):
    """어떤 속성을 물어도 더미를 돌려주는 모듈. __path__ 가 있어 패키지로도 동작한다."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # 서브모듈 import 를 허용한다

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        # 대문자로 시작하면 클래스 자리, 아니면 함수/값 자리로 본다.
        value = _Dummy if name[:1].isupper() else _Dummy()
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`isaaclab.*` 등을 요청하면 즉석에서 가짜 모듈을 만들어 준다."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        # 접두사 매칭 — isaaclab_physx, isaaclab_tasks, isaaclab_assets 등도 잡는다.
        if not any(root.startswith(pfx) for pfx in _STUB_PREFIXES):
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        return _AnyModule(spec.name)

    def exec_module(self, module):
        pass


_FINDER = _StubFinder()


def install_stubs() -> None:
    """`isaaclab.*` 스텁 로더를 sys.meta_path 에 심는다 (중복 설치하지 않는다)."""
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)
    # 이미 진짜가 로드돼 있으면 스텁이 가려지므로 걷어낸다.
    for name in list(sys.modules):
        root = name.split(".")[0]
        if any(root.startswith(p) for p in _STUB_PREFIXES) and not isinstance(sys.modules[name], _AnyModule):
            del sys.modules[name]


def load_module(path: Path, alias: str | None = None, extra_paths: tuple[Path, ...] = ()):
    """임의 경로의 .py 를 import 한다 (sys.path 를 건드리지 않는다).

    `from .sibling import X` 같은 상대 import 를 쓰는 파일도 불러올 수 있도록,
    파일이 있는 디렉토리를 __path__ 로 갖는 가짜 부모 패키지를 만들어 붙인다.

    starter 는 형제 모듈 없이 혼자 생성되므로, 원본 소스 디렉토리를 `extra_paths` 로
    넘기면 거기서도 형제를 찾는다.
    """
    if not path.exists():
        raise SystemExit(
            f"파일이 없다: {path}\n"
            f"  exercises/tools/make_exercise.py --all 로 starter 를 생성한다"
        )
    install_stubs()

    pkg_name = f"_expkg_{abs(hash(str(path.parent))) & 0xFFFFFFFF:x}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(path.parent), *(str(x) for x in extra_paths)]
        sys.modules[pkg_name] = pkg

    name = alias or f"{pkg_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
