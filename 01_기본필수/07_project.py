"""실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""

# 0. 패키지 불러오기
from pathlib import Path
import csv
import json
import torch
from torch import nn

# 1. 데이터 불러오기 및 데이터 train_test_split
# 파일 위치를 기준으로 읽으므로 다른 작업 폴더에서 실행해도 됩니다.
base = Path(__file__).resolve().parent
with (base / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# 기본필수는 CPU 고정, full-batch 80회입니다. test는 Tensor로 만들지 않습니다.

## 재현성을 위한 시드 설정
torch.manual_seed(42)

# 2. 모델링 준비
## 입력 특성을 2개로 하고 중간 레이어를 8로 잡았습니다. 활성화 함수는 ReLU로 하게되며 분류 예측을 수행하게 됩니다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

## 손실함수는 CrossEntropyLoss로 합니다.
loss_fn = nn.CrossEntropyLoss()

## SGD 옵티마이저 설정 (학습률 0.1, 모멘텀 등 추가 설정 없습니다.)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
history = [[0, loss_fn(model(X_train), y_train).item()]]

# 3. 모델링
## epoch은 80회
model.train()
for epoch in range(1, 81):
    ## 기울기 초기화
    optimizer.zero_grad()
    ## 손실함수 통한 손실 계산
    loss = loss_fn(model(X_train), y_train)
    # 06번에서 배운 역전파와 파라미터 갱신 두 문장을 연결하세요.
    # 이미 위에서 이전 gradient를 비우고 loss를 계산했습니다.
    # >>> B08
    # TODO B08: 위 요구사항을 읽고 이 블록을 완성하세요.
    ## 역전파
    loss.backward()
    ## 파라미터 갱신
    optimizer.step()
    
    # <<< B08
    # 갱신 후 같은 train 자료의 loss를 기록합니다. 마지막 batch loss와 구분합니다.
    with torch.no_grad():
        history.append([epoch, loss_fn(model(X_train), y_train).item()])

# 5. 모델 평가
model.eval()
with torch.no_grad():
    # 학습이 끝난 파라미터로 입력 건에 대한 점수 다시 계산
    logits = model(X_valid)
    # 각 행에서 가장 큰 logit의 열 번호를 prediction으로 구하세요.
    # argmax의 dim=1은 클래스 방향입니다. 결과 shape는 (4,)입니다.
    # >>> B09
    # TODO B09: 위 요구사항을 읽고 이 블록을 완성하세요.
    prediction = torch.argmax(logits, dim=1)
    # <<< B09
    # 정확도 계산
    accuracy = (prediction == y_valid).float().mean().item()
with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval"])
    writer.writerows(history)
## 5-1. training 과정 표 및 결과 장표 말들기
metrics = {"device": "cpu", "epochs": 80, "train_rows": 12, "valid_rows": 4,
           "test_used": False, "valid_accuracy": accuracy,
           "initial_train_loss": history[0][1], "final_train_loss": history[-1][1],
           "valid_ids": [r["id"] for r in valid_rows], "actual": y_valid.tolist(),
           "predicted": prediction.tolist()}
(base / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
(base / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")
print("분할: train 12 / valid 4 / test 4(보관)")
print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
print("valid 정확도:", round(accuracy, 4))
print("valid 예측:", prediction.tolist())
print("생성: gpu_check.txt, training_history.csv, final_metrics.json")
