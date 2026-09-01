# task01 — 빈 씬 띄우기  `L0 · READ`

## 목표

Isaac Lab 스크립트가 **항상 갖는 3단 구조**를 눈으로 확인한다. 쓸 코드는 없다.

```
1) AppLauncher 로 Omniverse 앱을 먼저 띄운다
2) 그 다음에야 isaaclab.* 을 import 한다      ← 이 순서는 바꿀 수 없다
3) SimulationContext 를 만들고 reset() → step()
```

## 실행

```bash
python solution.py             # headless (기본)
python solution.py --viz kit   # GUI 로 보기
```

## 확인할 것

- `import` 문이 파일 중간, `simulation_app = app_launcher.app` **아래**에 있다.
  Isaac Sim 확장이 앱 기동 후에 등록되므로, 위로 올리면 `ModuleNotFoundError` 가 난다.
  이 프로젝트의 `train.py` / `play.py` 도 전부 같은 모양이다.
- `dt=1/200` — 이 프로젝트의 물리 스텝이다. 정책은 `decimation=4` 마다 한 번
  동작하므로 **제어 주기는 50 Hz** 가 된다.
- `sim.reset()` 이 물리 씬을 실제로 만든다. 그 전에는 어떤 asset 의 `data` 도 못 읽는다.

## 통과 기준

끝까지 돌면 `[PASS]` 2줄과 `[OK]` 가 찍힌다.

## 직접 해 보기 (선택)

`dt` 를 `1/50` 로 바꿔 GUI 로 돌려 본다. 물리가 거칠어지는 것이 보이는가?
locomotion 에서 물리 스텝을 크게 잡으면 접촉이 뚫리거나 튀는 이유다.
