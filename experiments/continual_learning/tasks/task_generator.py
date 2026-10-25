"""
experiments/continual_learning/tasks/task_generator.py
Splits a fully-preprocessed dataset into sequential CL tasks.

Each task contains only the samples belonging to a predefined group of classes.
The model sees tasks one at a time; it never revisits raw task data (only replay
buffers or regularisation penalties approximate past knowledge).
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Task:
    task_id:       int
    class_names:   list        # human-readable names for classes in this task
    class_indices: list        # integer label indices (from LabelEncoder)
    X_train:       np.ndarray
    y_train:       np.ndarray
    X_val:         np.ndarray
    y_val:         np.ndarray
    X_test:        np.ndarray
    y_test:        np.ndarray
    n_classes_cumulative: int  # total distinct classes seen up to and incl. this task


def build_tasks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    class_names: list,         # full list from LabelEncoder (len = n_total_classes)
    task_class_groups: list,   # list of lists of class-name strings, one list per task
) -> list:
    """
    Build a list of Task objects from a preprocessed dataset.

    Parameters
    ----------
    X_train/val/test   : preprocessed feature arrays (float32)
    y_train/val/test   : integer label arrays
    class_names        : full sorted class name list from LabelEncoder
    task_class_groups  : e.g. [['normal','dos'], ['ddos','password'], ...]

    Returns
    -------
    list of Task objects in task order
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    tasks = []
    seen_indices = set()

    for task_id, group in enumerate(task_class_groups):
        # Resolve class names → indices (case-insensitive match)
        indices = []
        for name in group:
            matched = [i for n, i in name_to_idx.items()
                       if n.lower() == name.lower()]
            if not matched:
                print(f"  [TaskGen] Warning: class '{name}' not found in dataset — skipping")
                continue
            indices.extend(matched)

        if not indices:
            print(f"  [TaskGen] Task {task_id}: no valid classes found — skipping task")
            continue

        seen_indices.update(indices)
        mask_tr = np.isin(y_train, indices)
        mask_va = np.isin(y_val,   indices)
        mask_te = np.isin(y_test,  indices)

        task = Task(
            task_id              = task_id,
            class_names          = [class_names[i] for i in indices],
            class_indices        = indices,
            X_train              = X_train[mask_tr],
            y_train              = y_train[mask_tr],
            X_val                = X_val[mask_va],
            y_val                = y_val[mask_va],
            X_test               = X_test[mask_te],
            y_test               = y_test[mask_te],
            n_classes_cumulative = len(seen_indices),
        )
        print(f"  [TaskGen] Task {task_id}: classes={task.class_names}  "
              f"train={len(task.X_train):,}  val={len(task.X_val):,}  "
              f"test={len(task.X_test):,}")
        tasks.append(task)

    return tasks
