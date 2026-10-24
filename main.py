import os
import glob
import time
import subprocess
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, iirnotch, filtfilt
from scipy.stats import ttest_rel

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from xgboost import XGBClassifier


# =========================================================
# 1. 기본 설정
# =========================================================

FS = 1000
START = 1500
SEG_LEN = 2000
TRIAL_LEN = 5000

WIN = 200
HOP = 100
ZC_THRESHOLD = 0.01

RANDOM_STATE = 42

DATASET_DIR = "sEMG-Hand-Gesture-Elbow-Angle-Dataset"
DATASET_URL = (
    "https://github.com/"
    "TestUser-alt-dev/"
    "sEMG-Hand-Gesture-Elbow-Angle-Dataset.git"
)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# 2. 데이터셋 준비
# =========================================================

if not os.path.exists(DATASET_DIR):
    subprocess.run(
        ["git", "clone", DATASET_URL],
        check=True
    )

files = sorted(
    glob.glob(
        os.path.join(
            DATASET_DIR,
            "data",
            "**",
            "*.csv"
        ),
        recursive=True
    )
)

if not files:
    raise FileNotFoundError(
        "CSV 파일을 찾을 수 없습니다."
    )


def label_of(path):
    """
    a_30 fist.csv -> 30_fist
    a_60 extor.csv -> 60_extor
    """
    name = os.path.basename(
        path
    ).replace(".csv", "")

    return (
        name
        .split("_", 1)[1]
        .replace(" ", "_")
    )


# =========================================================
# 3. 전처리 / 특징 추출
# =========================================================

def preprocess(x, fs=FS):

    # 60 Hz notch
    bn, an = iirnotch(
        60,
        30,
        fs
    )

    x = filtfilt(
        bn,
        an,
        x,
        axis=0
    )

    # 20~450 Hz band-pass
    b, a = butter(
        4,
        [
            20 / (fs / 2),
            450 / (fs / 2)
        ],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        x,
        axis=0
    )


def cut_segment(x):
    return x[
        START:
        START + SEG_LEN
    ]


def make_windows(x):

    n = (
        (len(x) - WIN)
        // HOP
        + 1
    )

    return np.stack([
        x[
            i * HOP:
            i * HOP + WIN
        ]
        for i in range(n)
    ])


def td_features(
    w,
    th=ZC_THRESHOLD
):

    rms = np.sqrt(
        np.mean(
            w ** 2,
            axis=0
        )
    )

    mav = np.mean(
        np.abs(w),
        axis=0
    )

    d = np.diff(
        w,
        axis=0
    )

    wl = np.sum(
        np.abs(d),
        axis=0
    )

    zc = np.sum(
        (
            (w[:-1] * w[1:])
            < 0
        )
        &
        (
            np.abs(d)
            >= th
        ),
        axis=0
    )

    ssc = np.sum(
        (
            d[:-1] * d[1:]
        ) < 0,
        axis=0
    )

    return np.concatenate([
        rms,
        mav,
        wl,
        zc,
        ssc
    ])


# =========================================================
# 4. Trial 목록 생성
# =========================================================

trial_records = []

for path in files:

    raw = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1
    )

    mark_idx = np.where(
        raw[:, 7] == 1
    )[0]

    label = label_of(path)

    for start in mark_idx:

        if (
            start + TRIAL_LEN
            <= len(raw)
        ):
            trial_records.append(
                (
                    path,
                    int(start),
                    label
                )
            )


print(
    "CSV 파일 수:",
    len(files)
)

print(
    "전체 trial 수:",
    len(trial_records)
)


# =========================================================
# 5. Trial -> Window 특징 생성
# =========================================================

def build(records):

    X = []
    y = []

    groups = defaultdict(list)

    for path, start, label in records:
        groups[path].append(
            (
                start,
                label
            )
        )

    for path, items in groups.items():

        raw = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1
        )

        # 앞 5개 열 = sEMG
        semg = raw[:, :5]

        for start, label in items:

            trial = semg[
                start:
                start + TRIAL_LEN
            ]

            trial_f = preprocess(
                trial
            )

            segment = cut_segment(
                trial_f
            )

            windows = make_windows(
                segment
            )

            for window in windows:

                X.append(
                    td_features(
                        window
                    )
                )

                y.append(
                    label
                )

    return (
        np.array(X),
        np.array(y)
    )


# =========================================================
# 6. Hold-out Train / Test
# =========================================================

trial_labels = [
    record[2]
    for record in trial_records
]

tr_trials, te_trials = (
    train_test_split(
        trial_records,
        test_size=0.2,
        stratify=trial_labels,
        random_state=RANDOM_STATE
    )
)


Xtr_raw, ytr_text = build(
    tr_trials
)

Xte_raw, yte_text = build(
    te_trials
)


scaler = StandardScaler()

Xtr = scaler.fit_transform(
    Xtr_raw
)

Xte = scaler.transform(
    Xte_raw
)


le = LabelEncoder()

ytr = le.fit_transform(
    ytr_text
)

yte = le.transform(
    yte_text
)


print(
    "Train:",
    Xtr.shape
)

print(
    "Test:",
    Xte.shape
)


# =========================================================
# 7. 모델
# =========================================================

model_prototypes = {

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr"
    ),

    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGB": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
}


models = {}

for name, proto in (
    model_prototypes.items()
):

    model = clone(proto)

    model.fit(
        Xtr,
        ytr
    )

    models[name] = model

    print(
        name,
        "학습 완료"
    )


# =========================================================
# 8. Window-level 성능
# =========================================================

window_rows = []

for name, model in (
    models.items()
):

    pred = model.predict(
        Xte
    )

    window_rows.append([
        name,

        accuracy_score(
            yte,
            pred
        ),

        precision_score(
            yte,
            pred,
            average="macro"
        ),

        recall_score(
            yte,
            pred,
            average="macro"
        ),

        f1_score(
            yte,
            pred,
            average="macro"
        )
    ])


window_df = pd.DataFrame(
    window_rows,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
)


print(
    "\n===== Window-level ====="
)

print(
    window_df.round(4)
)


# =========================================================
# 9. Trial-level Majority Voting
# =========================================================

trial_predictions = {}
trial_rows = []

for name, model in (
    models.items()
):

    true_trials = []
    pred_trials = []

    for record in te_trials:

        X_one_raw, y_one = build(
            [record]
        )

        X_one = scaler.transform(
            X_one_raw
        )

        pred_windows = (
            model.predict(
                X_one
            )
        )

        # 19개 window 중 최빈값
        vote = Counter(
            pred_windows
        ).most_common(1)[0][0]

        true_label = (
            le.transform(
                [y_one[0]]
            )[0]
        )

        true_trials.append(
            true_label
        )

        pred_trials.append(
            vote
        )


    true_trials = np.array(
        true_trials
    )

    pred_trials = np.array(
        pred_trials
    )


    trial_predictions[name] = {
        "true": true_trials,
        "pred": pred_trials
    }


    trial_rows.append([

        name,

        accuracy_score(
            true_trials,
            pred_trials
        ),

        precision_score(
            true_trials,
            pred_trials,
            average="macro"
        ),

        recall_score(
            true_trials,
            pred_trials,
            average="macro"
        ),

        f1_score(
            true_trials,
            pred_trials,
            average="macro"
        )
    ])


trial_df = pd.DataFrame(
    trial_rows,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
)


print(
    "\n===== Trial-level ====="
)

print(
    trial_df.round(4)
)


# =========================================================
# 10. Trial-level Confusion Matrix
# =========================================================

labels = le.classes_


for name in models.keys():

    true = (
        trial_predictions[name]
        ["true"]
    )

    pred = (
        trial_predictions[name]
        ["pred"]
    )


    cm = confusion_matrix(
        true,
        pred
    )


    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    im = ax.imshow(
        cm,
        cmap="Blues"
    )


    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right"
    )


    ax.set_yticks(
        range(len(labels)),
        labels
    )


    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        f"{name} - "
        "Trial-level Confusion Matrix"
    )


    for i in range(
        len(labels)
    ):

        for j in range(
            len(labels)
        ):

            if cm[i, j] != 0:

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8
                )


    plt.colorbar(im)
    plt.tight_layout()


    path = os.path.join(
        RESULT_DIR,
        f"confusion_trial_{name}.png"
    )


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# 11. Trial-level 5-fold
# =========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


trial_labels_np = np.array(
    trial_labels
)


fold_trial_acc = {
    name: []
    for name in model_prototypes
}


for fold, (
    tr_i,
    va_i
) in enumerate(
    skf.split(
        trial_records,
        trial_labels_np
    ),
    start=1
):

    print(
        f"\n===== Fold {fold} ====="
    )


    train_records = [
        trial_records[i]
        for i in tr_i
    ]


    valid_records = [
        trial_records[i]
        for i in va_i
    ]


    Xa_raw, ya_text = build(
        train_records
    )


    fold_scaler = (
        StandardScaler()
    )


    Xa = (
        fold_scaler
        .fit_transform(
            Xa_raw
        )
    )


    ya = le.transform(
        ya_text
    )


    fold_models = {}


    for name, proto in (
        model_prototypes.items()
    ):

        model = clone(
            proto
        )

        model.fit(
            Xa,
            ya
        )

        fold_models[name] = (
            model
        )


    for name, model in (
        fold_models.items()
    ):

        true_trials = []
        pred_trials = []


        for record in (
            valid_records
        ):

            X_one_raw, y_one = (
                build(
                    [record]
                )
            )


            X_one = (
                fold_scaler
                .transform(
                    X_one_raw
                )
            )


            pred_windows = (
                model.predict(
                    X_one
                )
            )


            vote = Counter(
                pred_windows
            ).most_common(1)[0][0]


            true_label = (
                le.transform(
                    [y_one[0]]
                )[0]
            )


            true_trials.append(
                true_label
            )

            pred_trials.append(
                vote
            )


        acc = accuracy_score(
            true_trials,
            pred_trials
        )


        fold_trial_acc[
            name
        ].append(
            acc
        )


        print(
            f"{name}: "
            f"{acc * 100:.2f}%"
        )


print(
    "\n===== "
    "Trial-level 5-fold 결과 "
    "====="
)


for name, values in (
    fold_trial_acc.items()
):

    print(
        f"{name}: "
        f"{np.mean(values) * 100:.2f} "
        f"± "
        f"{np.std(values) * 100:.2f}%"
    )


# =========================================================
# 12. Paired t-test
# =========================================================

print(
    "\n===== Paired t-test ====="
)


for a, b in combinations(
    model_prototypes.keys(),
    2
):

    _, p = ttest_rel(
        fold_trial_acc[a],
        fold_trial_acc[b]
    )


    verdict = (
        "유의함"
        if p < 0.05
        else "유의하지 않음"
    )


    if p < 0.0001:

        p_text = "< 0.0001"

    else:

        p_text = (
            f"= {p:.4f}"
        )


    print(
        f"{a} vs {b}: "
        f"p {p_text} "
        f"({verdict})"
    )


# =========================================================
# 13. 계산 비용
# =========================================================

cost_rows = []


for name, proto in (
    model_prototypes.items()
):

    model = clone(
        proto
    )


    start_time = (
        time.perf_counter()
    )


    model.fit(
        Xtr,
        ytr
    )


    train_time = (
        time.perf_counter()
        - start_time
    )


    start_time = (
        time.perf_counter()
    )


    model.predict(
        Xte
    )


    inference_time = (
        time.perf_counter()
        - start_time
    )


    cost_rows.append([
        name,
        train_time,
        inference_time
    ])


cost_df = pd.DataFrame(
    cost_rows,
    columns=[
        "Model",
        "Train Time (s)",
        "Inference Time (s)"
    ]
)


print(
    "\n===== 계산 비용 ====="
)

print(
    cost_df.round(4)
)