#!/usr/bin/env python3
"""Convert a URDF robot to a UniLab robot MJCF asset directory.

Usage:
  uv run unilab-import-robot <urdf_path> [robot_name]

"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_ASSET_ROOT = REPO_ROOT / "src" / "unilab" / "assets" / "robots"
TEMP_MESH_PREFIX = "meshes/meshes/"
DEFAULT_MATERIAL = "default_material"
IMU_SITE = "imu"
FOOT_LINK_SUFFIX = "_foot_link"
TOUCH_SITE_SUFFIX = "_touch_site"
LEG_SENSOR_ORDER = ("RF", "RM", "RB", "LF", "LM", "LB")
_FREE_X_HELPER_JOINT = "__unilab_keyframe_x"
_FREE_Y_HELPER_JOINT = "__unilab_keyframe_y"
_HEIGHT_HELPER_JOINT = "__unilab_keyframe_height"
_FREE_BALL_HELPER_JOINT = "__unilab_keyframe_orientation"
_HEIGHT_HELPER_ACTUATOR = "__unilab_keyframe_height_ctrl"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf_path", help="Path to the input URDF file.")
    parser.add_argument(
        "robot_name",
        nargs="?",
        help="Robot asset directory/XML name. Defaults to the URDF file stem.",
    )
    return parser.parse_args(argv)


def _resolve_urdf(path: str) -> Path:
    urdf = Path(path).expanduser()
    if not urdf.is_absolute():
        urdf = (Path.cwd() / urdf).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF path does not exist or is not a file: {urdf}")
    if urdf.suffix.lower() != ".urdf":
        raise ValueError(f"URDF path must end with .urdf: {urdf}")
    return urdf


def _robot_name(urdf: Path, raw_name: str | None) -> str:
    name = raw_name or urdf.stem
    if not name:
        raise ValueError("robot_name must not be empty")
    if Path(name).name != name:
        raise ValueError("robot_name must be a single path component")
    return name


def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    print(f"[unilab-import-robot] {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=check)


def _convert_urdf(urdf: Path, output_xml: Path) -> None:
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "uv",
            "run",
            "--with",
            "urdf-to-mjcf",
            "urdf-to-mjcf",
            str(urdf),
            "-o",
            str(output_xml),
        ]
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_by_tag(element: ET.Element, tag: str) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == tag:
            return child
    return None


def _children_by_tag(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_name(child.tag) == tag)


def _parse_urdf_vector(text: str | None, *, default: Sequence[float]) -> list[float]:
    if text is None:
        return list(default)
    values = [float(part) for part in text.split()]
    if len(values) != len(default):
        raise ValueError(f"expected {len(default)} values, got {len(values)}: {text}")
    return values


def _quat_from_rpy(rpy: Sequence[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _rotation_from_rpy(rpy: Sequence[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotate_inertia(inertia: Sequence[Sequence[float]], rpy: Sequence[float]) -> list[list[float]]:
    rotation = _rotation_from_rpy(rpy)
    rotated: list[list[float]] = []
    for row in range(3):
        rotated_row: list[float] = []
        for col in range(3):
            value = 0.0
            for i in range(3):
                for j in range(3):
                    value += rotation[row][i] * inertia[i][j] * rotation[col][j]
            rotated_row.append(value)
        rotated.append(rotated_row)
    return rotated


def _full_inertia_values(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [matrix[0][0], matrix[1][1], matrix[2][2], matrix[0][1], matrix[0][2], matrix[1][2]]


def _urdf_root_link(urdf_root: ET.Element) -> ET.Element | None:
    links: dict[str, ET.Element] = {}
    for link in _children_by_tag(urdf_root, "link"):
        name = link.get("name")
        if name is not None:
            links[name] = link
    child_links: set[str] = set()
    for joint in _children_by_tag(urdf_root, "joint"):
        child = _child_by_tag(joint, "child")
        if child is not None and child.get("link"):
            child_links.add(cast(str, child.get("link")))

    root_names = [name for name in links if name not in child_links]
    if not root_names:
        return None
    return links[root_names[0]]


def _urdf_root_link_inertial_attrs(urdf: Path) -> tuple[str, dict[str, str]] | None:
    urdf_root = ET.parse(urdf).getroot()
    root_link = _urdf_root_link(urdf_root)
    if root_link is None:
        return None

    inertial = _child_by_tag(root_link, "inertial")
    if inertial is None:
        return None

    mass = _child_by_tag(inertial, "mass")
    inertia = _child_by_tag(inertial, "inertia")
    if mass is None or inertia is None or mass.get("value") is None:
        return None

    origin = _child_by_tag(inertial, "origin")
    xyz = _parse_urdf_vector(
        origin.get("xyz") if origin is not None else None,
        default=[0.0, 0.0, 0.0],
    )
    rpy = _parse_urdf_vector(
        origin.get("rpy") if origin is not None else None,
        default=[0.0, 0.0, 0.0],
    )
    ixx = float(inertia.get("ixx", "0"))
    iyy = float(inertia.get("iyy", "0"))
    izz = float(inertia.get("izz", "0"))
    ixy = float(inertia.get("ixy", "0"))
    ixz = float(inertia.get("ixz", "0"))
    iyz = float(inertia.get("iyz", "0"))

    attrs = {
        "pos": _format_values(xyz),
        "mass": _format_float(float(mass.get("value"))),
    }
    if any(not math.isclose(value, 0.0, abs_tol=1e-15) for value in (ixy, ixz, iyz)):
        inertia_matrix = [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
        rotated_inertia = _rotate_inertia(inertia_matrix, rpy)
        attrs["fullinertia"] = _format_values(_full_inertia_values(rotated_inertia))
    else:
        attrs["quat"] = _format_values(_quat_from_rpy(rpy))
        attrs["diaginertia"] = _format_values([ixx, iyy, izz])
    return cast(str, root_link.get("name")), attrs


def _insert_body_inertial(body: ET.Element, attrs: dict[str, str]) -> None:
    for child in list(body):
        if child.tag == "inertial":
            body.remove(child)

    inertial = ET.Element("inertial", attrs)
    insert_at = 0
    for index, child in enumerate(list(body)):
        if child.tag in {"freejoint", "joint"}:
            insert_at = index + 1
            continue
        break
    body.insert(insert_at, inertial)


def _has_direct_child(element: ET.Element, tag: str, attrs: dict[str, str]) -> bool:
    return any(child.tag == tag and _attrs_match(child, attrs) for child in element)


def _root_body(root: ET.Element) -> ET.Element | None:
    return root.find("./worldbody/body")


def _leg_sort_key(name: str) -> tuple[int, str]:
    prefix = name.split("_", 1)[0]
    try:
        return LEG_SENSOR_ORDER.index(prefix), name
    except ValueError:
        return len(LEG_SENSOR_ORDER), name


def _joint_sensor_key(joint_name: str) -> tuple[int, str, int, str]:
    prefix, _, suffix = joint_name.partition("_joint")
    try:
        prefix_order = LEG_SENSOR_ORDER.index(prefix)
    except ValueError:
        prefix_order = len(LEG_SENSOR_ORDER)
    try:
        joint_order = int(suffix)
    except ValueError:
        joint_order = 0
    return prefix_order, prefix, joint_order, suffix


def _insert_site_after_geoms(body: ET.Element, site: ET.Element) -> None:
    insert_at = 0
    for index, child in enumerate(list(body)):
        if child.tag in {"inertial", "joint", "freejoint", "geom"}:
            insert_at = index + 1
            continue
        break
    body.insert(insert_at, site)


def _ensure_imu_site(root: ET.Element) -> bool:
    body = _root_body(root)
    if body is None:
        return False
    if _has_direct_child(body, "site", {"name": IMU_SITE}):
        return True

    _insert_site_after_geoms(
        body,
        ET.Element(
            "site",
            {"name": IMU_SITE, "pos": "0 0 0", "size": "0.01", "rgba": "1 0 0 1"},
        ),
    )
    return True


def _ensure_foot_touch_sites(root: ET.Element) -> list[str]:
    touch_site_names: list[str] = []
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name is None or not body_name.endswith(FOOT_LINK_SUFFIX):
            continue
        foot_prefix = body_name[: -len(FOOT_LINK_SUFFIX)]
        site_name = f"{foot_prefix}{TOUCH_SITE_SUFFIX}"
        touch_site_names.append(site_name)
        if _has_direct_child(body, "site", {"name": site_name}):
            continue
        _insert_site_after_geoms(
            body,
            ET.Element("site", {"name": site_name, "size": "0.01", "rgba": "0 0 0 0"}),
        )
    return sorted(touch_site_names, key=_leg_sort_key)


def _sensor_name_for_joint(joint_name: str, sensor_type: str) -> str:
    prefix, separator, suffix = joint_name.partition("_joint")
    if separator:
        return f"{prefix}_{suffix}_{sensor_type}"
    return f"{joint_name}_{sensor_type}"


def _actuated_joint_names(root: ET.Element) -> list[str]:
    joint_names: list[str] = []
    seen: set[str] = set()
    for actuator in root.findall("./actuator/*"):
        joint_name = actuator.get("joint")
        if joint_name is None or joint_name in seen:
            continue
        seen.add(joint_name)
        joint_names.append(joint_name)
    return sorted(joint_names, key=_joint_sensor_key)


def _append_imu_sensors(sensor: ET.Element) -> None:
    sensor_specs = [
        ("gyro", {"site": IMU_SITE, "name": "gyro"}),
        ("velocimeter", {"site": IMU_SITE, "name": "local_linvel"}),
        ("framepos", {"objtype": "site", "objname": IMU_SITE, "name": "position"}),
        ("framezaxis", {"objtype": "site", "objname": IMU_SITE, "name": "upvector"}),
        ("framelinvel", {"objtype": "site", "objname": IMU_SITE, "name": "global_linvel"}),
        ("frameangvel", {"objtype": "site", "objname": IMU_SITE, "name": "global_angvel"}),
    ]
    for tag, attrs in sensor_specs:
        ET.SubElement(sensor, tag, attrs)


def _append_joint_sensors(sensor: ET.Element, joint_names: Sequence[str]) -> None:
    for joint_name in joint_names:
        ET.SubElement(
            sensor,
            "jointpos",
            {"name": _sensor_name_for_joint(joint_name, "pos"), "joint": joint_name},
        )
        ET.SubElement(
            sensor,
            "jointvel",
            {"name": _sensor_name_for_joint(joint_name, "vel"), "joint": joint_name},
        )


def _append_touch_sensors(sensor: ET.Element, touch_site_names: Sequence[str]) -> None:
    for site_name in touch_site_names:
        foot_prefix = site_name[: -len(TOUCH_SITE_SUFFIX)]
        ET.SubElement(sensor, "touch", {"name": f"{foot_prefix}_foot_contact", "site": site_name})
    for site_name in touch_site_names:
        foot_prefix = site_name[: -len(TOUCH_SITE_SUFFIX)]
        ET.SubElement(
            sensor,
            "framepos",
            {"objtype": "site", "objname": site_name, "name": f"{foot_prefix}_pos"},
        )
    for site_name in touch_site_names:
        foot_prefix = site_name[: -len(TOUCH_SITE_SUFFIX)]
        ET.SubElement(
            sensor,
            "framelinvel",
            {"objtype": "site", "objname": site_name, "name": f"{foot_prefix}_vel"},
        )


def _rebuild_generated_sensors(
    root: ET.Element, *, include_imu: bool, touch_site_names: Sequence[str]
) -> None:
    joint_names = _actuated_joint_names(root)
    if not include_imu and not touch_site_names and not joint_names:
        return

    for sensor in root.findall("sensor"):
        root.remove(sensor)

    sensor = ET.Element("sensor")
    if include_imu:
        _append_imu_sensors(sensor)
    _append_joint_sensors(sensor, joint_names)
    _append_touch_sensors(sensor, touch_site_names)
    root.append(sensor)


def _ensure_generated_sites_and_sensors(root: ET.Element) -> None:
    include_imu = _ensure_imu_site(root)
    touch_site_names = _ensure_foot_touch_sites(root)
    if not include_imu and not touch_site_names:
        return
    _rebuild_generated_sensors(root, include_imu=include_imu, touch_site_names=touch_site_names)


def _preserve_root_link_inertial(urdf: Path, xml_path: Path) -> None:
    inertial = _urdf_root_link_inertial_attrs(urdf)
    if inertial is None:
        return
    root_link_name, attrs = inertial

    tree = ET.parse(xml_path)
    root = tree.getroot()
    body = root.find(f"./worldbody/body[@name='{root_link_name}']")
    if body is None:
        body = root.find("./worldbody/body")
    if body is None:
        raise ValueError("converted robot XML must define a root body under <worldbody>")

    _insert_body_inertial(body, attrs)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode")


def _xml_references_generated_mesh_assets(xml_path: Path) -> bool:
    root = ET.parse(xml_path).getroot()
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is None:
            continue
        normalized = mesh_file.replace("\\", "/")
        if normalized.startswith(TEMP_MESH_PREFIX):
            return True
    return False


def _move_mesh_assets(robot_dir: Path, robot_xml: Path) -> None:
    generated_mesh_dir = robot_dir / "meshes" / "meshes"
    target_assets_dir = robot_dir / "assets"
    if not generated_mesh_dir.is_dir():
        if not _xml_references_generated_mesh_assets(robot_xml):
            return
        raise FileNotFoundError(f"expected generated mesh directory: {generated_mesh_dir}")
    if target_assets_dir.exists():
        raise FileExistsError(f"target assets directory already exists: {target_assets_dir}")

    generated_mesh_dir.rename(target_assets_dir)
    meshes_parent = robot_dir / "meshes"
    try:
        meshes_parent.rmdir()
    except OSError:
        pass


def _iter_with_parent(root: ET.Element) -> Iterable[tuple[ET.Element | None, ET.Element]]:
    yield None, root
    for parent in root.iter():
        for child in list(parent):
            yield parent, child


def _remove_element(parent: ET.Element | None, child: ET.Element) -> None:
    if parent is not None:
        parent.remove(child)


def _attrs_match(element: ET.Element, attrs: dict[str, str]) -> bool:
    return all(element.get(key) == value for key, value in attrs.items())


def _has_only_child(element: ET.Element, tag: str, attrs: dict[str, str]) -> bool:
    children = list(element)
    return len(children) == 1 and children[0].tag == tag and _attrs_match(children[0], attrs)


def _strip_generated_scene_bits(root: ET.Element) -> None:
    for parent, element in list(_iter_with_parent(root)):
        if element.tag == "default" and element.get("class") == "floor":
            _remove_element(parent, element)
            continue
        if element.tag == "light":
            _remove_element(parent, element)
            continue
        if element.tag == "visual" and _has_only_child(
            element,
            "global",
            {"offwidth": "3840", "offheight": "2160"},
        ):
            _remove_element(parent, element)
            continue
        if element.tag == "visual" and _has_only_child(
            element,
            "rgba",
            {"haze": "0.15 0.25 0.35 1"},
        ):
            _remove_element(parent, element)
            continue
        if element.tag == "texture" and _attrs_match(
            element,
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
            },
        ):
            _remove_element(parent, element)
            continue
        if element.tag == "material" and element.get("name") == "groundplane":
            _remove_element(parent, element)
            continue
        if element.tag == "texture" and _attrs_match(
            element,
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
            },
        ):
            _remove_element(parent, element)


def _remove_default_material(root: ET.Element) -> None:
    for parent, element in list(_iter_with_parent(root)):
        if element.tag == "material" and element.get("name") == DEFAULT_MATERIAL:
            _remove_element(parent, element)
            continue
        if element.get("material") == DEFAULT_MATERIAL:
            del element.attrib["material"]
        if element.tag == "geom" and element.get("class") == "floor":
            _remove_element(parent, element)
            continue
        if (
            element.tag == "geom"
            and element.get("type") == "plane"
            and element.get("material") == "groundplane"
        ):
            _remove_element(parent, element)


def _set_mesh_paths(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "assets")

    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is None:
            continue
        normalized = mesh_file.replace("\\", "/")
        if normalized.startswith(TEMP_MESH_PREFIX):
            normalized = normalized[len(TEMP_MESH_PREFIX) :]
        mesh.set("file", normalized)


def _convert_motor_actuators(root: ET.Element) -> None:
    joints_by_name = {
        joint.get("name"): joint
        for joint in root.findall(".//joint")
        if joint.get("name") is not None
    }
    for actuator in root.findall("./actuator/*"):
        if actuator.tag == "motor":
            actuator.tag = "position"
        if actuator.tag != "position":
            continue
        joint = joints_by_name.get(actuator.get("joint"))
        joint_range = joint.get("range") if joint is not None else None
        if joint_range is None:
            continue
        actuator.set("ctrlrange", joint_range)
        actuator.set("ctrllimited", "true")


def _ensure_robot_default_joint(root: ET.Element) -> None:
    default_root = root.find("default")
    if default_root is None:
        default_root = ET.Element("default")
        root.insert(0, default_root)

    robot_default = default_root.find("./default[@class='robot']")
    if robot_default is None:
        robot_default = ET.SubElement(default_root, "default", {"class": "robot"})

    joint = robot_default.find("./joint")
    if joint is None:
        joint = ET.Element("joint")
        robot_default.insert(0, joint)
    joint.set("damping", "2")
    joint.set("armature", "0.01")
    joint.set("frictionloss", "0.2")


def _parse_values(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part) for part in text.split()]


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _format_values(values: Sequence[float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _joint_default(joint: ET.Element | None) -> float:
    return 0.0


def _body_joint_order(body: ET.Element) -> Iterable[ET.Element]:
    for child in body:
        if child.tag in {"freejoint", "joint"}:
            yield child
    for child in body:
        if child.tag == "body":
            yield from _body_joint_order(child)


def _root_body_pose(body: ET.Element) -> list[float]:
    del body
    return [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]


def _scene_keyframe_values(robot_xml: Path) -> tuple[list[float], list[float]]:
    root = ET.parse(robot_xml).getroot()
    root_body = root.find("./worldbody/body")
    if root_body is None:
        raise ValueError("robot XML must define a root body under <worldbody>")

    joints_by_name: dict[str, ET.Element] = {}
    qpos: list[float] = []
    added_free_pose = False
    for joint in _body_joint_order(root_body):
        joint_type = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
        name = joint.get("name")
        if name:
            joints_by_name[name] = joint

        if joint_type == "free":
            if not added_free_pose:
                qpos.extend(_root_body_pose(root_body))
                added_free_pose = True
        elif joint_type == "ball":
            qpos.extend([1.0, 0.0, 0.0, 0.0])
        else:
            qpos.append(_joint_default(joint))

    ctrl: list[float] = []
    for actuator in root.findall("./actuator/*"):
        ctrl.append(_joint_default(joints_by_name.get(actuator.get("joint", ""))))
    return qpos, ctrl


def _write_scene_xml(robot_xml: Path, scene_xml: Path, robot_name: str) -> None:
    qpos, ctrl = _scene_keyframe_values(robot_xml)
    root = ET.Element("mujoco", {"model": f"{robot_name} scene"})

    ET.SubElement(root, "include", {"file": robot_xml.name})

    default = ET.SubElement(root, "default")
    floor_default = ET.SubElement(default, "default", {"class": "floor"})
    ET.SubElement(
        floor_default,
        "geom",
        {"type": "plane", "size": "0 0 0.05", "material": "groundplane"},
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "3840", "offheight": "2160"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(worldbody, "geom", {"name": "floor", "class": "floor", "size": "0 0 0.05"})

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": _format_values(qpos), "ctrl": _format_values(ctrl)},
    )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="unicode")


def _strip_scene_includes_for_fragment(scene_xml: Path) -> Path:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    includes = root.findall("include")
    if not includes:
        return scene_xml
    for include in includes:
        root.remove(include)

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{scene_xml.name}",
        dir=str(scene_xml.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _postprocess_xml(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _set_mesh_paths(root)
    _convert_motor_actuators(root)
    _ensure_robot_default_joint(root)
    _remove_default_material(root)
    _strip_generated_scene_bits(root)
    _ensure_generated_sites_and_sensors(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode")


def _parse_required_values(text: str | None, name: str) -> list[float]:
    if text is None:
        raise ValueError(f"keyframe is missing {name}=...")
    return [float(part) for part in text.split()]


def _replace_free_joint_with_viewer_sliders(
    root: ET.Element, height_range: tuple[float, float]
) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("compiled scene is missing <worldbody>")

    for body in worldbody.findall("body"):
        free_joint = body.find("./freejoint")
        if free_joint is None:
            free_joint = body.find("./joint[@type='free']")
        if free_joint is None:
            continue

        insert_at = list(body).index(free_joint)
        body.remove(free_joint)
        body.set("pos", "0 0 0")
        body.set("quat", "1 0 0 0")
        sliders = [
            ET.Element(
                "joint",
                {
                    "name": _FREE_X_HELPER_JOINT,
                    "type": "slide",
                    "axis": "1 0 0",
                    "range": "-2 2",
                    "limited": "true",
                },
            ),
            ET.Element(
                "joint",
                {
                    "name": _FREE_Y_HELPER_JOINT,
                    "type": "slide",
                    "axis": "0 1 0",
                    "range": "-2 2",
                    "limited": "true",
                },
            ),
            ET.Element(
                "joint",
                {
                    "name": _HEIGHT_HELPER_JOINT,
                    "type": "slide",
                    "axis": "0 0 1",
                    "range": _format_values(height_range),
                    "limited": "true",
                },
            ),
            ET.Element("joint", {"name": _FREE_BALL_HELPER_JOINT, "type": "ball"}),
        ]
        for offset, slider in enumerate(sliders):
            body.insert(insert_at + offset, slider)
        return

    raise ValueError("height helper requires a direct worldbody child with a free joint")


def _append_height_controller(root: ET.Element, height_range: tuple[float, float]) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.Element("actuator")
        root.append(actuator)
    actuator.append(
        ET.Element(
            "position",
            {
                "name": _HEIGHT_HELPER_ACTUATOR,
                "joint": _HEIGHT_HELPER_JOINT,
                "kp": "1000",
                "ctrlrange": _format_values(height_range),
                "ctrllimited": "true",
            },
        )
    )


def _append_height_ctrl_to_keyframes(root: ET.Element) -> None:
    for key in root.findall(".//keyframe/key"):
        qpos = _parse_required_values(key.get("qpos"), "qpos")
        ctrl = _parse_required_values(key.get("ctrl"), "ctrl") if key.get("ctrl") else []
        if len(qpos) < 3:
            raise ValueError("height helper requires a floating-base keyframe qpos")
        key.set("ctrl", _format_values([*ctrl, qpos[2]]))


def _ensure_tuning_scene_visuals(root: ET.Element) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(0, visual)
    if visual.find("headlight") is None:
        visual.append(
            ET.Element(
                "headlight",
                {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
            )
        )
    if visual.find("rgba") is None:
        visual.append(ET.Element("rgba", {"haze": "0.15 0.25 0.35 1"}))
    if visual.find("global") is None:
        visual.append(ET.Element("global", {"azimuth": "120", "elevation": "-20"}))

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1, asset)
    if root.find(".//texture[@name='groundplane']") is None:
        asset.append(
            ET.Element(
                "texture",
                {
                    "type": "2d",
                    "name": "groundplane",
                    "builtin": "checker",
                    "mark": "edge",
                    "rgb1": "0.2 0.3 0.4",
                    "rgb2": "0.1 0.2 0.3",
                    "markrgb": "0.8 0.8 0.8",
                    "width": "300",
                    "height": "300",
                },
            )
        )
    if root.find(".//material[@name='groundplane']") is None:
        asset.append(
            ET.Element(
                "material",
                {
                    "name": "groundplane",
                    "texture": "groundplane",
                    "texuniform": "true",
                    "texrepeat": "5 5",
                    "reflectance": "0.2",
                },
            )
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if root.find(".//light[@name='__unilab_tuning_light']") is None:
        worldbody.append(
            ET.Element(
                "light",
                {
                    "name": "__unilab_tuning_light",
                    "pos": "0 0 1.5",
                    "dir": "0 0 -1",
                    "directional": "true",
                },
            )
        )
    if root.find(".//geom[@name='floor']") is None:
        worldbody.append(
            ET.Element(
                "geom",
                {
                    "name": "floor",
                    "size": "0 0 0.05",
                    "type": "plane",
                    "material": "groundplane",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
        )


def _materialize_tuning_scene(
    robot_xml: Path,
    scene_xml: Path,
    *,
    add_height_joint: bool = True,
    height_range: tuple[float, float] = (-1.0, 1.0),
) -> Path:
    from unilab.base.backend.mujoco.xml import materialize_scene_fragments

    fragment_xml = _strip_scene_includes_for_fragment(scene_xml)
    try:
        merged = materialize_scene_fragments(str(robot_xml), fragment_files=[str(fragment_xml)])
    finally:
        if fragment_xml != scene_xml:
            fragment_xml.unlink(missing_ok=True)
    tree = ET.parse(merged)
    root = tree.getroot()
    _ensure_tuning_scene_visuals(root)
    if add_height_joint:
        _replace_free_joint_with_viewer_sliders(root, height_range)
        _append_height_controller(root, height_range)
        _append_height_ctrl_to_keyframes(root)
    ET.indent(tree, space="  ")
    tree.write(merged, encoding="unicode")
    return Path(merged)


def _compile_tuning_scene(robot_xml: Path, scene_xml: Path) -> Any:
    import mujoco

    mujoco_api: Any = mujoco
    merged = _materialize_tuning_scene(robot_xml, scene_xml)
    try:
        return mujoco_api.MjModel.from_xml_path(str(merged))
    finally:
        try:
            merged.unlink()
        except FileNotFoundError:
            pass


def _load_keyframe(model: Any, key_name: str = "home") -> Any:
    import mujoco

    mujoco_api: Any = mujoco
    key_id = mujoco_api.mj_name2id(model, mujoco_api.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        raise ValueError(f"keyframe '{key_name}' not found in scene.xml")
    data = mujoco_api.MjData(model)
    mujoco_api.mj_resetDataKeyframe(model, data, key_id)
    mujoco_api.mj_forward(model, data)
    return data


def _strip_height_helper_ctrl(model: Any, ctrl: Any) -> Any:
    import mujoco
    import numpy as np

    mujoco_api: Any = mujoco
    helper_id = mujoco_api.mj_name2id(
        model, mujoco_api.mjtObj.mjOBJ_ACTUATOR, _HEIGHT_HELPER_ACTUATOR
    )
    if helper_id < 0:
        return ctrl
    return np.delete(ctrl, helper_id)


def _actuator_qpos_mappings(model: Any) -> list[tuple[int, int]]:
    import mujoco

    mujoco_api: Any = mujoco
    hinge = int(mujoco_api.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco_api.mjtJoint.mjJNT_SLIDE)
    mappings: list[tuple[int, int]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        if int(model.jnt_type[joint_id]) not in {hinge, slide}:
            continue
        mappings.append((actuator_id, int(model.jnt_qposadr[joint_id])))
    return mappings


def _sync_changed_ctrl_to_qpos(
    data: Any, mappings: Sequence[tuple[int, int]], last_ctrl: Any
) -> bool:
    import numpy as np

    changed = False
    for actuator_id, qpos_addr in mappings:
        if np.isclose(data.ctrl[actuator_id], last_ctrl[actuator_id], atol=1e-12):
            continue
        data.qpos[qpos_addr] = data.ctrl[actuator_id]
        changed = True
    last_ctrl[:] = data.ctrl
    return changed


def _open_tuning_viewer(model: Any, data: Any) -> None:
    import mujoco
    import mujoco.viewer

    mujoco_api: Any = mujoco
    actuator_qpos_mappings = _actuator_qpos_mappings(model)
    last_ctrl = data.ctrl.copy()

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        print("MuJoCo viewer opened. Tune qpos/ctrl, then close the window to update scene.xml.")
        try:
            while viewer.is_running():
                if _sync_changed_ctrl_to_qpos(data, actuator_qpos_mappings, last_ctrl):
                    mujoco_api.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass


def _write_tuned_scene_keyframe(
    scene_xml: Path, model: Any, data: Any, key_name: str = "home"
) -> None:
    import mujoco

    mujoco_api: Any = mujoco
    mujoco_api.mj_forward(model, data)
    qpos = data.qpos.copy()
    ctrl = cast(Sequence[float], _strip_height_helper_ctrl(model, data.ctrl.copy()))

    tree = ET.parse(scene_xml)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    key = keyframe.find(f"./key[@name='{key_name}']")
    if key is None:
        key = ET.SubElement(keyframe, "key", {"name": key_name})
    key.set("qpos", _format_values(qpos))
    key.set("ctrl", _format_values(ctrl))
    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="unicode")
    print(f"[unilab-import-robot] wrote tuned keyframe to {scene_xml.relative_to(REPO_ROOT)}")


def _tune_scene_keyframe(robot_xml: Path, scene_xml: Path) -> None:
    model = _compile_tuning_scene(robot_xml, scene_xml)
    data = _load_keyframe(model)
    _open_tuning_viewer(model, data)
    _write_tuned_scene_keyframe(scene_xml, model, data)


def convert(urdf_path: str, robot_name: str | None) -> Path:
    urdf = _resolve_urdf(urdf_path)
    name = _robot_name(urdf, robot_name)
    robot_dir = ROBOT_ASSET_ROOT / name
    output_xml = robot_dir / f"{name}.xml"

    _convert_urdf(urdf, output_xml)
    _preserve_root_link_inertial(urdf, output_xml)
    _move_mesh_assets(robot_dir, output_xml)
    _postprocess_xml(output_xml)
    _write_scene_xml(output_xml, robot_dir / "scene.xml", name)

    print(f"[unilab-import-robot] wrote {output_xml.relative_to(REPO_ROOT)}", flush=True)
    _tune_scene_keyframe(output_xml, robot_dir / "scene.xml")
    return output_xml


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        convert(args.urdf_path, args.robot_name)
    except Exception as exc:
        print(f"[unilab-import-robot] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
