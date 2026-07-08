"""Contact-based RSA/FCA metrics for LIBERO rollouts."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


IGNORE_OBJECT_PREFIXES = (
    "floor",
    "main_table",
    "mount",
    "robot",
    "table",
    "wall",
)


@dataclass
class ContactEvent:
    step: int
    robot_geom: str
    object_geom: str
    object_name: str


@dataclass
class FirstContactTracker:
    target_object: Optional[str]
    object_geoms: Dict[str, List[str]]
    first_contact: Optional[ContactEvent] = None
    first_grasped_object: Optional[str] = None
    first_grasped_step: Optional[int] = None
    first_moved_object: Optional[str] = None
    first_moved_step: Optional[int] = None
    initial_object_positions: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def normalize_object_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_inner_env(env: Any) -> Any:
    return getattr(env, "env", env)


def get_sim(env: Any) -> Any:
    inner_env = get_inner_env(env)
    return getattr(inner_env, "sim", getattr(env, "sim", None))


def get_geom_name(model: Any, geom_id: int) -> Optional[str]:
    if hasattr(model, "geom_id2name"):
        return model.geom_id2name(geom_id)
    return None


def get_all_geom_names(env: Any) -> List[str]:
    sim = get_sim(env)
    if sim is None:
        return []
    return [name for i in range(sim.model.ngeom) if (name := get_geom_name(sim.model, i))]


def get_gripper_tip_geoms(env: Any) -> List[str]:
    geom_names = get_all_geom_names(env)
    tip_geoms = [
        name
        for name in geom_names
        if name.startswith("gripper") and "finger" in name and ("pad_collision" in name or "tip" in name)
    ]
    if tip_geoms:
        return tip_geoms
    return [name for name in geom_names if name.startswith("gripper") and "finger" in name and "collision" in name]


def get_gripper_contact_geoms(env: Any) -> List[str]:
    geom_names = get_all_geom_names(env)
    contacts = [
        name
        for name in geom_names
        if name.startswith("gripper") and "finger" in name and "collision" in name
    ]
    return contacts or get_gripper_tip_geoms(env)


def _candidate_object_names(env: Any) -> List[str]:
    inner_env = get_inner_env(env)
    candidates = set()
    for attr in ("objects_dict", "object_states_dict"):
        value = getattr(inner_env, attr, None)
        if isinstance(value, dict):
            candidates.update(str(key) for key in value.keys())
    for obj in getattr(inner_env, "objects", []) or []:
        name = getattr(obj, "name", None)
        if name:
            candidates.add(str(name))
    for name in getattr(inner_env, "obj_of_interest", []) or []:
        candidates.add(str(name))

    filtered = []
    for name in candidates:
        normalized = normalize_object_name(name)
        if not normalized:
            continue
        if normalized.startswith(IGNORE_OBJECT_PREFIXES):
            continue
        if normalized.endswith("_region") or "_region_" in normalized:
            continue
        filtered.append(name)
    return sorted(set(filtered), key=len, reverse=True)


def get_task_object_geoms(env: Any) -> Dict[str, List[str]]:
    object_names = _candidate_object_names(env)
    object_geoms = {name: [] for name in object_names}
    for geom_name in get_all_geom_names(env):
        normalized_geom = normalize_object_name(geom_name)
        if normalized_geom is None or normalized_geom.startswith(IGNORE_OBJECT_PREFIXES):
            continue
        for object_name in object_names:
            normalized_object = normalize_object_name(object_name)
            if normalized_geom == normalized_object or normalized_geom.startswith(f"{normalized_object}_"):
                object_geoms[object_name].append(geom_name)
                break
    return {name: sorted(geoms) for name, geoms in object_geoms.items() if geoms}


def get_target_object(env: Any) -> Optional[str]:
    inner_env = get_inner_env(env)
    obj_of_interest = getattr(inner_env, "obj_of_interest", None)
    if obj_of_interest:
        return str(obj_of_interest[0])
    return None


def get_distractor_objects(target_object: Optional[str], object_geoms: Dict[str, List[str]]) -> List[str]:
    if target_object is None:
        return []
    target_base = re.sub(r"_\d+$", "", normalize_object_name(target_object) or target_object)
    distractors = []
    for object_name in object_geoms:
        normalized = normalize_object_name(object_name) or object_name
        if object_name == target_object:
            continue
        if re.sub(r"_\d+$", "", normalized) == target_base:
            distractors.append(object_name)
    return sorted(distractors)


def geom_to_object(geom_name: Optional[str], object_geoms: Dict[str, List[str]]) -> Optional[str]:
    if geom_name is None:
        return None
    for object_name, geoms in object_geoms.items():
        if geom_name in geoms:
            return object_name
    return None


def get_contacted_objects_by_gripper_tip(env: Any, object_geoms: Dict[str, List[str]]) -> List[ContactEvent]:
    sim = get_sim(env)
    if sim is None:
        return []
    gripper_tip_geoms = set(get_gripper_tip_geoms(env))
    events = []
    for contact_idx in range(sim.data.ncon):
        contact = sim.data.contact[contact_idx]
        geom1 = get_geom_name(sim.model, contact.geom1)
        geom2 = get_geom_name(sim.model, contact.geom2)
        if geom1 in gripper_tip_geoms:
            robot_geom, object_geom = geom1, geom2
        elif geom2 in gripper_tip_geoms:
            robot_geom, object_geom = geom2, geom1
        else:
            continue

        object_name = geom_to_object(object_geom, object_geoms)
        if object_name is None:
            continue
        events.append(ContactEvent(step=-1, robot_geom=robot_geom, object_geom=object_geom, object_name=object_name))
    return events


def get_contacted_objects_by_gripper(env: Any, object_geoms: Dict[str, List[str]]) -> List[ContactEvent]:
    sim = get_sim(env)
    if sim is None:
        return []
    gripper_geoms = set(get_gripper_contact_geoms(env))
    events = []
    for contact_idx in range(sim.data.ncon):
        contact = sim.data.contact[contact_idx]
        geom1 = get_geom_name(sim.model, contact.geom1)
        geom2 = get_geom_name(sim.model, contact.geom2)
        if geom1 in gripper_geoms:
            robot_geom, object_geom = geom1, geom2
        elif geom2 in gripper_geoms:
            robot_geom, object_geom = geom2, geom1
        else:
            continue
        object_name = geom_to_object(object_geom, object_geoms)
        if object_name is None:
            continue
        events.append(ContactEvent(step=-1, robot_geom=robot_geom, object_geom=object_geom, object_name=object_name))
    return events


def get_object_position(env: Any, object_name: str):
    sim = get_sim(env)
    if sim is None:
        return None
    try:
        body_id = sim.model.body_name2id(object_name)
    except Exception:
        return None
    try:
        return sim.data.body_xpos[body_id].copy()
    except Exception:
        return None


def update_first_moved_object(tracker: FirstContactTracker, env: Any, step: int, threshold: float = 0.03) -> None:
    if tracker.first_moved_object is not None:
        return
    if not tracker.initial_object_positions:
        for object_name in tracker.object_geoms:
            position = get_object_position(env, object_name)
            if position is not None:
                tracker.initial_object_positions[object_name] = position
        return
    for object_name, initial_position in tracker.initial_object_positions.items():
        current_position = get_object_position(env, object_name)
        if current_position is None:
            continue
        try:
            moved = float(((current_position - initial_position) ** 2).sum() ** 0.5)
        except Exception:
            continue
        if moved >= threshold:
            tracker.first_moved_object = object_name
            tracker.first_moved_step = step
            return


def update_first_grasped_object(tracker: FirstContactTracker, env: Any, step: int) -> None:
    if tracker.first_grasped_object is not None:
        return
    contacts = get_contacted_objects_by_gripper(env, tracker.object_geoms)
    by_object: Dict[str, set] = {}
    for event in contacts:
        by_object.setdefault(event.object_name, set()).add(event.robot_geom)
    for object_name, robot_geoms in by_object.items():
        if len(robot_geoms) >= 2:
            tracker.first_grasped_object = object_name
            tracker.first_grasped_step = step
            return


def update_first_contact_tracker(tracker: FirstContactTracker, env: Any, step: int) -> Optional[ContactEvent]:
    update_first_moved_object(tracker, env, step)
    update_first_grasped_object(tracker, env, step)
    if tracker.first_contact is not None:
        return tracker.first_contact
    contacted_objects = get_contacted_objects_by_gripper_tip(env, tracker.object_geoms)
    if not contacted_objects:
        return None
    first_contact = contacted_objects[0]
    first_contact.step = step
    tracker.first_contact = first_contact
    return first_contact


def compute_fca(first_contacted_object: Optional[str], target_object: Optional[str]) -> bool:
    return first_contacted_object is not None and target_object is not None and first_contacted_object == target_object


def infer_rsa_object(
    target_object: Optional[str],
    selected_object: Optional[str] = None,
    first_grasped_object: Optional[str] = None,
    first_moved_object: Optional[str] = None,
    first_contacted_object: Optional[str] = None,
) -> tuple[Optional[str], str]:
    if selected_object is not None:
        return selected_object, "selected_object"
    if first_grasped_object is not None:
        return first_grasped_object, "first_grasped_object"
    if first_moved_object is not None:
        return first_moved_object, "first_significantly_moved_object"
    if first_contacted_object is not None:
        return first_contacted_object, "fallback_to_first_contact"
    return None, "unavailable"


def compute_rsa(target_object: Optional[str], rsa_object: Optional[str]) -> bool:
    return rsa_object is not None and target_object is not None and rsa_object == target_object


def make_episode_metrics(
    episode_index: int,
    seed: int,
    task_id: int,
    task_name: str,
    instruction: str,
    target_object: Optional[str],
    distractor_objects: Iterable[str],
    tracker: FirstContactTracker,
    success: bool,
    episode_length: int,
    error_message: str = "",
    rsa_definition: str = "first_contact_proxy",
) -> Dict[str, Any]:
    first_contact = tracker.first_contact
    first_contacted_object = first_contact.object_name if first_contact is not None else None
    fca_correct = compute_fca(first_contacted_object, target_object)
    rsa_object, rsa_method = infer_rsa_object(
        target_object=target_object,
        first_contacted_object=first_contacted_object,
        first_grasped_object=tracker.first_grasped_object,
        first_moved_object=tracker.first_moved_object,
    )
    rsa_correct = compute_rsa(target_object=target_object, rsa_object=rsa_object)
    warnings = list(tracker.warnings)
    if rsa_method == "fallback_to_first_contact":
        warnings.append("RSA fell back to first contact because no grasped or moved object was detected.")

    return {
        "episode_index": episode_index,
        "seed": seed,
        "task_id": task_id,
        "task_name": task_name,
        "instruction": instruction,
        "target_object": target_object,
        "target_object_normalized": normalize_object_name(target_object),
        "distractor_objects": list(distractor_objects),
        "first_contacted_object": first_contacted_object,
        "first_contact_step": first_contact.step if first_contact is not None else None,
        "first_contact_geom_robot": first_contact.robot_geom if first_contact is not None else None,
        "first_contact_geom_object": first_contact.object_geom if first_contact is not None else None,
        "selected_object": None,
        "first_grasped_object": tracker.first_grasped_object,
        "first_grasped_step": tracker.first_grasped_step,
        "first_moved_object": tracker.first_moved_object,
        "first_moved_step": tracker.first_moved_step,
        "rsa_object": rsa_object,
        "rsa_method": rsa_method,
        "rsa_definition": "first_grasped_then_moved_then_contact_fallback",
        "rsa_correct": bool(rsa_correct),
        "fca_definition": "gripper_tip_first_contact",
        "fca_correct": bool(fca_correct),
        "no_contact": first_contacted_object is None,
        "success": bool(success),
        "episode_length": episode_length,
        "error_message": error_message,
        "warnings": warnings,
    }


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_episodes = len(rows)
    success_count = sum(1 for row in rows if row.get("success"))
    fca_count = sum(1 for row in rows if row.get("fca_correct"))
    rsa_count = sum(1 for row in rows if row.get("rsa_correct"))
    num_no_contact = sum(1 for row in rows if row.get("first_contacted_object") in (None, ""))
    num_wrong_first_contact = sum(
        1
        for row in rows
        if row.get("first_contacted_object") not in (None, "") and not row.get("fca_correct")
    )
    rsa_fallback_count = sum(1 for row in rows if row.get("rsa_method") == "fallback_to_first_contact")
    return {
        "success_count": success_count,
        "success_rate": float(success_count) / float(num_episodes) if num_episodes else 0.0,
        "rsa_count": rsa_count,
        "referent_selection_accuracy": float(rsa_count) / float(num_episodes) if num_episodes else 0.0,
        "fca_count": fca_count,
        "first_contact_accuracy": float(fca_count) / float(num_episodes) if num_episodes else 0.0,
        "num_episodes": num_episodes,
        "num_no_contact_episodes": num_no_contact,
        "num_wrong_first_contact_episodes": num_wrong_first_contact,
        "rsa_fallback_count": rsa_fallback_count,
        "rsa_fallback_rate": float(rsa_fallback_count) / float(num_episodes) if num_episodes else 0.0,
        "rsa_definition": "first_grasped_then_moved_then_contact_fallback",
        "rsa_note": "RSA uses first grasped object, then first significantly moved object, and falls back to first contact only if neither is available.",
        "fca_definition": "gripper_tip_first_contact",
    }
